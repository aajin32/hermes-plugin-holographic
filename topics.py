"""Topics: group facts that belong to the same matter, across sessions.

One thing is often recorded over several sessions — a project's constraints
land one fact at a time, each in the conversation where it came up. ``category``
cannot express that (it says what *kind* of fact it is, not what it is *about*),
so this module adds a second axis: a named topic a fact can belong to, plus a
summary that ties its facts together.

The store owns the tables (``fact_topics``, and ``facts.topic_id``); this module
holds the helpers and the summary bookkeeping around them.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# The schema for the topics table. Kept separate from the facts schema so an
# older build reading the file still opens it (it ignores an unknown table).
TOPIC_SCHEMA = """
CREATE TABLE IF NOT EXISTS fact_topics (
    topic_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    summary     TEXT DEFAULT '',
    fact_count  INTEGER DEFAULT 0,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_topics_name ON fact_topics(name);
"""


def _existing_ids(store, sql: str, params=()) -> set:
    """Column of ids from *sql*, as a set (upstream reads columns positionally)."""
    return {row[0] for row in store.read(sql, params)}


def get_or_create_topic(store, name: str) -> int | None:
    """The topic_id for *name*, creating the topic when it is new.

    A blank name is not a topic: returns None so a caller that received an empty
    ``topic`` argument stores the fact ungrouped instead of in a ""-named topic.
    """
    name = (name or "").strip()
    if not name:
        return None
    row = store.read("SELECT topic_id FROM fact_topics WHERE name = ?", (name,))
    if row:
        return int(row[0][0])
    try:
        cur = store._write("INSERT INTO fact_topics (name) VALUES (?)", (name,))
        return int(cur.lastrowid)
    except Exception:  # a concurrent insert won the UNIQUE race: read theirs
        row = store.read("SELECT topic_id FROM fact_topics WHERE name = ?", (name,))
        return int(row[0][0]) if row else None


def list_topics(store, limit: int = 50) -> list[dict]:
    """Topics, the ones holding the most facts first."""
    rows = store.read(
        "SELECT topic_id, name, summary, fact_count, created_at, updated_at "
        "FROM fact_topics ORDER BY fact_count DESC, name LIMIT ?", (int(limit),))
    return [dict(r) for r in rows]


def get_topic(store, topic_id: int) -> dict | None:
    """One topic by id, or None."""
    rows = store.read(
        "SELECT topic_id, name, summary, fact_count, created_at, updated_at "
        "FROM fact_topics WHERE topic_id = ?", (int(topic_id),))
    return dict(rows[0]) if rows else None


def get_topic_by_name(store, name: str) -> dict | None:
    """One topic by name, or None (so the agent can resolve a name it was given)."""
    rows = store.read(
        "SELECT topic_id, name, summary, fact_count, created_at, updated_at "
        "FROM fact_topics WHERE name = ?", ((name or "").strip(),))
    return dict(rows[0]) if rows else None


def set_summary(store, topic: str | int, summary: str) -> bool:
    """Write a topic's summary. *topic* is its id or its name; False when unknown.

    The summary is the point of the feature: facts are what was said, the summary
    is what the matter amounts to, and only a writer (the agent, or a nightly
    job) can produce it — hence a plain setter rather than anything automatic.
    """
    topic_id = topic if isinstance(topic, int) else None
    if topic_id is None:
        found = get_topic_by_name(store, str(topic))
        if found is None:
            return False
        topic_id = found["topic_id"]
    cur = store._write("UPDATE fact_topics SET summary = ?, updated_at = CURRENT_TIMESTAMP "
                       "WHERE topic_id = ?", (summary or "", int(topic_id)))
    return bool(cur.rowcount)


def facts_in_topic(store, topic: str | int, *, min_trust: float = 0.0, limit: int = 200) -> list[dict]:
    """Every fact in a topic, most trusted first (id or name accepted)."""
    topic_id = topic if isinstance(topic, int) else None
    if topic_id is None:
        found = get_topic_by_name(store, str(topic))
        if found is None:
            return []
        topic_id = found["topic_id"]
    rows = store.read(
        "SELECT fact_id, content, category, tags, trust_score, topic_id, created_at, updated_at "
        "FROM facts WHERE topic_id = ? AND trust_score >= ? "
        "ORDER BY trust_score DESC, updated_at DESC, fact_id DESC LIMIT ?",
        (int(topic_id), min_trust, int(limit)))
    return [dict(r) for r in rows]


def assign_fact(store, fact_id: int, topic: str | int) -> bool:
    """Put an existing fact into a topic (id or name); False when either is unknown.

    Counts move with the fact, so ``fact_count`` stays the number of facts the
    topic actually holds rather than the number ever added to it.
    """
    topic_id = topic if isinstance(topic, int) else None
    if topic_id is None:
        topic_id = get_or_create_topic(store, str(topic))
    if topic_id is None:
        return False
    rows = store.read("SELECT topic_id FROM facts WHERE fact_id = ?", (int(fact_id),))
    if not rows:
        return False
    previous = rows[0][0]
    if previous == topic_id:
        return True
    if previous is not None:
        _bump_count(store, previous, -1)
    store._write("UPDATE facts SET topic_id = ?, updated_at = CURRENT_TIMESTAMP WHERE fact_id = ?",
                 (topic_id, int(fact_id)))
    _bump_count(store, topic_id, +1)
    return True


def _bump_count(store, topic_id: int, delta: int) -> None:
    """Adjust a topic's fact_count, never below zero (deleted facts can double-count)."""
    store._write("UPDATE fact_topics SET fact_count = MAX(0, fact_count + ?), "
                 "updated_at = CURRENT_TIMESTAMP WHERE topic_id = ?", (delta, int(topic_id)))


def merge_facts(store, fact_ids: list[int], content: str, *, topic: str | int | None = None) -> int | None:
    """Collapse several facts into one, across sessions. Returns the surviving id.

    The matter this feature exists for: the same point recorded in three
    conversations becomes three near-identical facts. This keeps the most trusted
    of them, rewrites it to *content*, and ends the rest through the normal
    history path so nothing is silently destroyed — a merged-away fact stays in
    ``fact_versions`` and can be reviewed or reverted. Facts carrying a different
    topic are left alone (a merge must not silently regroup the store).
    """
    ids = [int(i) for i in fact_ids if i is not None]
    if len(ids) < 2 or not (content or "").strip():
        return None

    rows = store.read(
        f"SELECT fact_id, topic_id, trust_score FROM facts WHERE fact_id IN ({','.join('?' * len(ids))})", ids)
    if len(rows) < 2:
        return None
    rows.sort(key=lambda r: (r["trust_score"], -r["fact_id"]), reverse=True)
    keep = int(rows[0]["fact_id"])
    drop = [int(r["fact_id"]) for r in rows[1:]]

    if topic is not None:
        assign_fact(store, keep, topic)
    # rewrite the survivor first: the content it takes on is what the others were
    store.change_fact(keep, content, valid_from=None)
    for fact_id in drop:
        try:
            store.remove_fact(fact_id)
        except Exception as exc:  # one failed removal must not strand the rest
            logger.debug("holographic: could not merge away fact #%s: %s", fact_id, exc)
    logger.info("holographic: merged %d fact(s) into #%s", len(drop), keep)
    return keep
