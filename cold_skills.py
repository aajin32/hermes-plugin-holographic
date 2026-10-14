"""Cold skills: keep rarely-used skills out of the prompt, bring them back on demand.

A skill the user parks under ``skills-cold/`` stops loading into every turn, which
is the point — but the agent then cannot see that it exists, so the user has to
remember to move it back by hand. This module closes that loop:

1. :func:`index_cold_skills` reads each cold skill's description once and stores it
   as a fact (``category="skill"``), so the store holds a searchable record of what
   is parked;
2. :func:`check_cold_skills` matches the recent conversation against those records
   at session end;
3. :func:`activate_skill` moves a skill that matched back into ``skills/`` and drops
   its index fact.

Matching runs through the provider's retriever, so with the neural embedding
weights in place it is a semantic match (a Chinese question phrased with
different words still finds the skill), and without a model it degrades to the
lexical path rather than breaking.

Everything here is best-effort: a cold directory that does not exist, an
unreadable SKILL.md or a failing move must never take the session down.
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# The category the index facts are stored under. Kept out of the categories the
# agent stores facts in, so a "skill" fact never mixes into recall of what the
# user knows.
SKILL_CATEGORY = "skill"

# Tag prefix on an index fact, and the marker the skill name is read back from
# ("[skill:name] description").
_TAG = "cold-skill"
_RE_SKILL_FACT = re.compile(r"^\[skill:([^\]]+)\]\s*(.*)$", re.DOTALL)

# Score floor for activation. The retriever's score is embedding similarity
# (shifted to [0,1]) times trust, so a perfect semantic match on a default-trust
# (0.5) fact scores 0.5 at most. Measured with bge-small-zh: an on-topic Chinese
# question against a skill description scores ~0.34 at default trust, unrelated
# text ~0.18 — 0.30 sits in that gap. Set it too high and a park/unpark loop
# silently never fires, which is the failure this module exists to prevent.
ACTIVATE_MIN_SCORE = 0.30

# How much of a message's opening text to keep as the extracted description.
_DESCRIPTION_CHARS = 300
_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_RE_DESCRIPTION = re.compile(r"^description:\s*(.+?)\s*$", re.MULTILINE)


def skills_root(hermes_home: Path) -> Path:
    """The active skills directory for this Hermes home."""
    return hermes_home / "skills"


def cold_skills_root(hermes_home: Path) -> Path:
    """The cold skills directory for this Hermes home."""
    return hermes_home / "skills-cold"


def iter_cold_skills(hermes_home: Path):
    """Yield ``(name, skill_dir)`` for every cold skill, one per SKILL.md.

    A skill is a directory holding a SKILL.md; the directory name is the skill
    name (the same rule the skills loader uses).
    """
    root = cold_skills_root(hermes_home)
    if not root.is_dir():
        return
    for skill_md in sorted(root.rglob("SKILL.md")):
        yield skill_md.parent.name, skill_md.parent


def read_description(skill_dir: Path, name: str) -> str:
    """The skill's one-line description: its frontmatter, else its first prose line.

    Falls back to the directory name so every cold skill still gets an index fact
    (a skill with no readable description must still be discoverable by name).
    """
    try:
        text = (skill_dir / "SKILL.md").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return name

    match = _FRONTMATTER.match(text)
    if match:
        described = _RE_DESCRIPTION.search(match.group(1))
        if described:
            value = described.group(1).strip().strip("'\"")
            if value:
                return value
        body = text[match.end():]
    else:
        body = text

    # No frontmatter description: the first paragraph that is not a heading.
    for block in body.split("\n\n"):
        block = block.strip()
        if block and not block.startswith("#"):
            return " ".join(block.split())[:200]
    return name


def format_skill_fact(name: str, description: str) -> str:
    """The index fact's content, the shape :func:`parse_skill_fact` reads back."""
    return f"[skill:{name}] {description}"


def parse_skill_fact(content: str) -> str:
    """Skill name from an index fact's content; "" when it is not one."""
    match = _RE_SKILL_FACT.match(content or "")
    return match.group(1) if match else ""


def index_cold_skills(store, hermes_home: Path, *, retriever=None) -> int:
    """Store a fact per cold skill so recall can match against them; returns the count.

    Idempotent: skills already indexed are left alone (their name is in the
    store), so a restart neither duplicates facts nor re-reads every file. A
    skill removed from cold storage keeps its fact until
    :func:`activate_skill`/``unindex`` clears it — harmless, and it lets a skill
    moved back by hand still be found.
    """
    known = {parse_skill_fact(row["content"]) for row in store.read(
        "SELECT content FROM facts WHERE category = ?", (SKILL_CATEGORY,))}
    known.discard("")

    indexed = 0
    for name, skill_dir in iter_cold_skills(hermes_home) or ():
        if name in known:
            continue
        try:
            content = format_skill_fact(name, read_description(skill_dir, name))
            store.add_fact(content, category=SKILL_CATEGORY, tags=f"{_TAG},{name}")
            indexed += 1
        except Exception as exc:  # a skill that cannot be indexed must not stop the rest
            logger.debug("holographic: could not index cold skill %s: %s", name, exc)
    if indexed:
        logger.info("holographic: indexed %d cold skill(s) for on-demand activation", indexed)
    return indexed


def unindex_skill(store, name: str) -> None:
    """Drop a skill's index fact (after it was activated, or moved by hand)."""
    try:
        store._write("DELETE FROM facts WHERE category = ? AND content LIKE ?", (SKILL_CATEGORY, f"[skill:{name}]%"))
    except Exception as exc:
        logger.debug("holographic: could not unindex skill %s: %s", name, exc)


def recent_user_text(messages, *, limit: int = 5, chars: int = 200) -> str:
    """The last *limit* user messages joined into one query string.

    The tail of the conversation is what says what the user is working on; older
    turns have already had their chance to activate a skill.
    """
    parts = []
    for message in messages or ():
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            parts.append(content.strip()[:chars])
    return " ".join(parts[-limit:])


def check_cold_skills(store, retriever, messages, hermes_home: Path, *,
                      min_score: float = ACTIVATE_MIN_SCORE, limit: int = 10) -> list[str]:
    """Activate the cold skills the recent conversation is about; returns their names."""
    query = recent_user_text(messages)
    if not query or retriever is None:
        return []
    try:
        # semantic=True: this is the conversational path, where a skill must be
        # found by what it is about, not by sharing a word with the message.
        # min_trust=0: the floor below is what decides, and index facts may sit
        # below the provider's recall trust threshold.
        hits = retriever.search(query, category=SKILL_CATEGORY, limit=limit, semantic=True, min_trust=0.0)
    except Exception as exc:
        logger.debug("holographic: cold-skill match failed: %s", exc)
        return []

    activated = []
    for fact in hits:
        if fact.get("score", 0.0) < min_score:
            continue
        name = parse_skill_fact(fact.get("content", ""))
        if name and activate_skill(name, hermes_home) and store is not None:
            unindex_skill(store, name)
            activated.append(name)
            logger.info("holographic: activated cold skill '%s' (score=%.2f)", name, fact.get("score", 0.0))
    return activated


def activate_skill(name: str, hermes_home: Path) -> bool:
    """Move skill *name* from cold storage into the active skills directory.

    False when it is not in cold storage, is already active, or the move failed —
    never an exception: this runs at session end and must not break the turn.
    """
    source = None
    for candidate_name, skill_dir in iter_cold_skills(hermes_home) or ():
        if candidate_name == name:
            source = skill_dir
            break
    if source is None:
        logger.debug("holographic: cold skill not found: %s", name)
        return False

    target = skills_root(hermes_home) / name
    if target.exists():
        logger.debug("holographic: skill already active: %s", name)
        return False

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        logger.info("holographic: activated cold skill '%s'", name)
        return True
    except Exception as exc:
        logger.warning("holographic: could not activate skill '%s': %s", name, exc)
        return False
