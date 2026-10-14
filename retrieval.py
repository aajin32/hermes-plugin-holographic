"""Fact retrieval for the memory store: FTS5 keyword candidates reranked with Jaccard and HRR similarity, weighted by
trust (ported from KIK memory_agent.py); entity recall straight from the entity index."""

from __future__ import annotations

import logging
import math
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .store import MemoryStore

from . import holographic as hrr
from . import embedding as emb
from . import text as _text
from .history import CLOSED, version_state
from .text import (CJK_NEGATORS, CLAIM_WORDS, CONTRASTIVE, NEGATORS, UNSEGMENTED, analyze, content_chars,
                   content_keys, content_stems, gram_coverage, is_chinese, is_greeting, is_sea, query_grams,
                   query_terms, run_grams, soft_jaccard, soft_overlap, words, zh_char_text)

logger = logging.getLogger(__name__)

_FACT_COLUMNS = "fact_id, content, category, tags, trust_score, retrieval_count, helpful_count, created_at, updated_at"
_F_COLUMNS = ", ".join(f"f.{c.strip()}" for c in _FACT_COLUMNS.split(","))
_ROLE_ENTITY, _ROLE_CONTENT = hrr.ROLE_ENTITY, hrr.ROLE_CONTENT
_PUNCT = ".,;:!?\"'()[]{}#@<>"
# FTS5 operator characters. Inside a token they become spaces of a quoted phrase: "pve-01" -> "pve 01", the same two
# tokens unicode61 indexed, adjacent (deleting them gave "pve01", which matches nothing).
_FTS_OPERATORS = str.maketrans({c: " " for c in '"()*^:-+'})
_RE_UNSEGMENTED = UNSEGMENTED
# The commonest Indonesian function words: a store with none of them holds no Indonesian, and Latin words of a query
# of undetermined language need no word scan (Indonesian prefixes are what hides a word from FTS5).
_INDONESIAN_PROBE = '"yang" OR "dengan" OR "untuk" OR "tidak" OR "adalah" OR "sudah" OR "belum"'
# ... and of German, whose compounds hide their last word from a prefix query ("Katze" in "Familienkatze").
_GERMAN_PROBE = '"und" OR "nicht" OR "ist" OR "sind" OR "der" OR "das" OR "eine" OR "mit"'
# ... Dutch (plurals that change the word: "raam" / "ramen"), Turkish (consonant changes: "kitap" / "kitabı") and
# Polish (ł, and words typed without their marks), with words no other supported language writes as a word of its own.
_DUTCH_PROBE = '"het" OR "een" OR "niet" OR "zijn" OR "wordt" OR "naar"'
_TURKISH_PROBE = '"bir" OR "için" OR "değil" OR "gibi" OR "olarak" OR "çok"'
_POLISH_PROBE = '"się" OR "jest" OR "oraz" OR "który" OR "przez" OR "jak"'
_SCAN_PROBE = " OR ".join((_INDONESIAN_PROBE, _GERMAN_PROBE, _DUTCH_PROBE, _TURKISH_PROBE, _POLISH_PROBE))
# A question told to be in one of these languages scans whatever the store holds: the probe words above are a sample,
# and a store of short facts may have none of them ("Kitabı masada bıraktım." asked as "Kitap nerede?")
_SCAN_LANGUAGES = frozenset({"id", "de", "tr", "pl", "nl"})

# Cosine similarity a fact must reach to be recalled by the neural path alone
# (on the raw [-1, 1] scale, before _shift). bge-small-zh puts unrelated
# sentences around 0.3-0.4 and related ones above 0.6; 0.55 sits in the gap.
# Only the conversational semantic fallback uses it — the fused path trusts its
# lexical candidates, which already passed their own tests.
_EMB_MIN_COS_SIM = 0.55


def _scan_language(query: str) -> bool:
    """Is *query* in a language whose other word forms the FTS index cannot reach (told by its function words)?"""
    if not query:
        return False
    analysis = analyze(query)
    return analysis.determined and not _SCAN_LANGUAGES.isdisjoint(analysis.langs)


class _Corpus:
    """What a search runs over: the current facts, or (``as_of``) the versions that may have held on a date. *source*
    is a table or a subquery with the columns of ``facts``, *key* its row id, *fts* the FTS5 index over it."""

    __slots__ = ("source", "params", "fts", "key", "cache")

    def __init__(self, source: str, params: tuple, fts: str, key: str, cache: dict):
        self.source, self.params, self.fts, self.key, self.cache = source, params, fts, key, cache


# Versions that may have held on a date (as_of): corrected ones never did. Trust: the fact's while the version is
# current, else the snapshot taken when it was closed.
_AS_OF_SOURCE = (
    "(SELECT v.version_id, v.fact_id, v.seq, v.content, v.category, v.tags, v.status, v.valid_from, v.valid_to, "
    "v.valid_to_exact, v.recorded_on, v.recorded_at AS created_at, COALESCE(v.closed_at, v.recorded_at) AS updated_at, "
    "COALESCE(c.trust_score, v.trust_score, ?) AS trust_score, 0 AS retrieval_count, NULL AS hrr_vector "
    "FROM fact_versions v LEFT JOIN facts c ON c.fact_id = v.fact_id AND v.status = 'current' "
    "WHERE v.status != 'corrected' AND (v.valid_from IS NULL OR v.valid_from <= ?) AND (v.valid_to IS NULL OR v.valid_to > ?))")


def _shift(sim: float) -> float:
    """Cosine similarity [-1, 1] -> [0, 1]."""
    return (sim + 1.0) / 2.0


def _related_floor(dim: int) -> float:
    """Residual similarity a fact must reach to count as structurally tied to an entity in `related`. Measured at
    dim=1024 on 400 facts with one to four entities: tied facts >= 0.35, the best of 17k untied pairs 0.088; 0.2 sits
    in the gap with a wide margin. Small dimensions get a proportionally higher floor (8 sd of chance similarity)."""
    return min(0.5, max(0.2, 8.0 * math.sqrt(0.5 / max(dim, 1))))


class FactRetriever:
    """Multi-strategy fact retrieval with trust-weighted scoring."""

    def __init__(self, store: MemoryStore, temporal_decay_half_life: int = 0,  # days, 0 = disabled
                 fts_weight: float = 0.4, jaccard_weight: float = 0.3, hrr_weight: float = 0.3, hrr_dim: int = 1024,
                 embedding_weight: float = 0.7):
        self.store, self.half_life, self.hrr_dim = store, temporal_decay_half_life, hrr_dim
        if hrr_weight > 0 and not hrr._HAS_NUMPY:  # redistribute weights without numpy
            fts_weight, jaccard_weight, hrr_weight = 0.6, 0.4, 0.0
        # Neural embeddings dominate when the model is present: the lexical part
        # (fts + jaccard + hrr) keeps its relative share of whatever is left.
        # Without the model the original lexical weights stand untouched.
        self._emb_available = emb.is_available()
        if self._emb_available and embedding_weight > 0:
            self.embedding_weight = min(float(embedding_weight), 1.0)
            lexical = 1.0 - self.embedding_weight
            total = fts_weight + jaccard_weight + hrr_weight
            if total > 0:
                fts_weight, jaccard_weight, hrr_weight = (fts_weight * lexical / total,
                                                           jaccard_weight * lexical / total,
                                                           hrr_weight * lexical / total)
        else:
            self.embedding_weight = 0.0
        self.fts_weight, self.jaccard_weight, self.hrr_weight = fts_weight, jaccard_weight, hrr_weight
        # row id -> (text, Analysis), for word-level scans: one cache for the current facts, one for versions
        self._current = _Corpus("facts", (), "facts_fts", "fact_id", {})
        self._version_cache: dict[int, tuple[str, object]] = {}

    def _as_of(self, day: str) -> _Corpus:
        return _Corpus(_AS_OF_SOURCE, (self.store.default_trust, day, day), "fact_versions_fts", "version_id",
                       self._version_cache)

    def _atom(self, word: str):
        return hrr.encode_atom(word, self.hrr_dim)

    def _phases(self, blob: bytes):
        """Decode a stored vector; None for a blob of another dimension or a damaged one (logged, row skipped)."""
        try:
            return hrr.bytes_to_phases(blob, dim=self.hrr_dim)
        except (ValueError, TypeError) as exc:
            logger.warning("holographic: skipping an unreadable HRR vector (%s)", exc)
            return None

    def search(self, query: str, category: str | None = None, min_trust: float = 0.3, limit: int = 10,
               corpus: _Corpus | None = None, roots: bool = True, semantic: bool = False) -> list[dict]:
        """FTS5 candidates (limit*3) → Jaccard + HRR rerank → trust weighting → optional temporal decay
        0.5^(age_days / half_life). Returns fact dicts with 'score', sorted desc. *corpus*: the current facts by
        default; search_as_of passes the versions of a date. *roots*=False matches words exactly, as 0.8.0 did (the
        candidates of write_check: a word of the same root must not crowd a conflict out)."""
        corpus = corpus or self._current
        key = corpus.key
        terms = query_terms(query or "")
        candidates = {f[key]: f for f in self._fts_candidates(query, category, min_trust, limit * 3, corpus)
                      if self._contains_a_term(terms, f, exact=not roots)}
        extra = " OR ".join(e for e in (t.fts_extra() for t in terms if t.fts_ok) if e) if roots else ""
        if extra:  # words of the same root: only what the words' own forms missed, ranked below them
            for fact in self._fts_query(extra, category, min_trust, limit * 3, corpus):
                if fact[key] not in candidates and self._contains_a_term(terms, fact):
                    fact["fts_rank"] *= _text._DERIV_WEIGHT
                    candidates[fact[key]] = fact
        for fact in self._ngram_candidates(query, category, min_trust, limit * 3, corpus):
            if fact[key] not in candidates or candidates[fact[key]].get("fts_rank", 0) < fact["fts_rank"]:
                candidates[fact[key]] = fact
        for fact in self._scan_candidates(terms, category, min_trust, limit * 3, corpus, exact=not roots, query=query):
            candidates.setdefault(fact[key], fact)  # fills in what FTS cannot see; FTS ranks what it found
        # Neural fallback, opt-in through ``semantic`` — the conversational recall
        # paths pass it, the structural ones do not. A question phrased with
        # different words than any fact (the usual case in Chinese) still finds it
        # by similarity alone. Off by default because an embedding always has a
        # nearest neighbour: widening the exact-match (roots=False), probe/related
        # and as_of paths, whose empty result IS their answer, would break them.
        if not candidates and semantic and corpus is self._current and self._semantic_fallback_allowed(query):
            found = self._embedding_candidates(query, category, min_trust, limit)
            if found:
                return found
        # BM25 does not see roots: a fact holding "Марина" and "по ночам" ranks with every other fact about Marina for
        # "ночная подработка Марины", and "The walker" matches the prefix "walk"* as fully as a fact with "walk". When
        # any candidate meets any term only through a root (its own or the fact's), all of them are ranked by the
        # weighted share of the query they hold, which counts the root at _DERIV_WEIGHT (as as_of always is).
        if corpus is not self._current:
            self._corpus_rank(query, terms, list(candidates.values()), category, min_trust, corpus)
        elif roots and any(0.0 < t.match(analyze(f'{f.get("content", "")} {f.get("tags") or ""}')) < 1.0
                           for f in candidates.values() for t in terms):
            self._coverage_rank(query, terms, list(candidates.values()))
        return self._rerank(query, list(candidates.values()), limit)

    def _corpus_rank(self, query: str, terms, candidates: list[dict], category, min_trust, corpus: _Corpus) -> None:
        """FTS5's BM25 counts every row of its index; the versions index holds corrected versions and every other
        date too, so for as_of its rank would weigh words by versions that are not in the corpus. Here the weights
        come from the corpus itself: the IDF-weighted share of the query's terms a version contains, or (an
        unsegmented script) the share of the query's n-grams, whichever is higher."""
        if not candidates:
            return
        texts = self._texts(category, min_trust, corpus)
        df = {t: sum(1 for _, _, a in texts if t.found_in(a)) for t in terms}
        weight = {t: math.log(1.0 + len(texts) / df[t]) for t in terms if df[t]}
        # A word no version of the date contains still says what the query is about (as in `relevant`): it counts
        # against coverage, weighted like the commonest present word.
        total = sum(weight.values()) + sum(1 for t in terms if not df[t]) * (min(weight.values()) if weight else 0.0)
        grams, zh = query_grams(query), is_chinese(query or "")
        for fact in candidates:
            text = f'{fact.get("content", "")} {fact.get("tags") or ""}'
            analysis = analyze(text)
            share = sum(w * t.match(analysis) for t, w in weight.items()) / total if total else 0.0
            fact["fts_rank"] = max(share, gram_coverage(grams, analysis, zh) if grams else 0.0)

    def _coverage_rank(self, query: str, terms, candidates: list[dict]) -> None:
        """The same weighted share for the current facts, cheaply: document frequencies from the FTS index (one COUNT
        per term, as the recall gate takes them), the share computed on the candidates only, not on every fact."""
        total_facts = max(self.store.count_facts(), 1)
        df = {t: self._doc_freq(t) for t in terms}
        weight = {t: math.log(1.0 + total_facts / df[t]) for t in terms if df[t]}
        if not weight:
            return
        total = sum(weight.values()) + sum(1 for t in terms if not df[t]) * min(weight.values())
        grams, zh = query_grams(query), is_chinese(query or "")
        for fact in candidates:
            analysis = analyze(f'{fact.get("content", "")} {fact.get("tags") or ""}')
            share = sum(w * t.match(analysis) for t, w in weight.items()) / total
            fact["fts_rank"] = max(share, gram_coverage(grams, analysis, zh) if grams else 0.0)

    def search_as_of(self, query: str, day: str, category: str | None = None, min_trust: float = 0.3,
                     limit: int = 10) -> dict:
        """What held on *day*: the search runs over the versions that may have held then (candidates, IDF, the
        floor and the limit are all computed over them, so limit=1 is the best version for that date). Versions
        known to hold go to 'results', those whose start or end is not known to the day to 'undetermined'."""
        self.store._require_history()
        held, maybe = [], {}
        # A wide pool: FTS5 picks candidates by its global rank, which knows nothing of the date; the corpus rank
        # (_corpus_rank) then orders them by the versions of that date only.
        for v in self.search(query, category=category, min_trust=min_trust, limit=max(limit * 10, 100),
                             corpus=self._as_of(day)):
            state, why = version_state(v, day)
            entry = {"fact_id": v["fact_id"], "version": v["seq"], "content": v["content"], "category": v["category"],
                     "valid_from": v["valid_from"] or "unknown",
                     "valid_to": v["valid_to"] if not v["valid_to"] or v["valid_to_exact"] else f"before {v['valid_to']}",
                     "status": v["status"], "score": round(v["score"], 4)}
            if state == "holds" and all(h["fact_id"] != v["fact_id"] for h in held):
                held.append(entry)
            elif state == "undetermined":
                maybe.setdefault(v["fact_id"], {**entry, "reason": why})
        known = {h["fact_id"] for h in held}
        undetermined = [e for fact_id, e in maybe.items() if fact_id not in known]
        return {"as_of": day, "results": held[:limit], "undetermined": undetermined[:limit]}

    @staticmethod
    def _contains_a_term(terms, fact: dict, exact: bool = False) -> bool:
        """A stem prefix ("contra"*) also matches other words: keep a candidate only when it really contains a query
        term (same word, same stem, the same identifier with word boundaries, or, unless *exact*, a word of the same
        root)."""
        if not terms:
            return True
        analysis = analyze(f'{fact.get("content", "")} {fact.get("tags") or ""}')
        return any(t._exact(analysis) if exact else t.found_in(analysis) for t in terms)

    def _texts(self, category: str | None = None, min_trust: float | None = None,
               corpus: _Corpus | None = None) -> list[tuple[int, float, object]]:
        """(row id, trust, Analysis) of every fact (or version) passing the filters. The rows are read on every call
        (the table is the truth; another process may have written); a row is analysed again only when its text
        changed, so the first scan after start analyses every fact and later ones only what changed."""
        corpus = corpus or self._current
        where = [c for c, on in (("f.trust_score >= ?", min_trust is not None), ("f.category = ?", category)) if on]
        params = [p for p, on in ((min_trust, min_trust is not None), (category, category)) if on]
        rows = self.store.read(f"SELECT f.{corpus.key} AS rid, f.content, f.tags, f.trust_score FROM {corpus.source} f"
                               + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY rid",
                               [*corpus.params, *params])
        cache, out = corpus.cache, []
        for row in rows:
            text = f'{row["content"]} {row["tags"] or ""}'
            hit = cache.get(row["rid"])
            if hit is None or hit[0] != text:
                hit = cache[row["rid"]] = (text, analyze(text))
            out.append((row["rid"], row["trust_score"], hit[1]))
        if corpus is self._current and not where and len(cache) > len(out) + 1024:  # deleted facts: only an
            kept = {fact_id: cache[fact_id] for fact_id, _, _ in out}  # unfiltered read of the facts sees every one
            cache.clear()
            cache.update(kept)
        elif corpus is not self._current and len(cache) > 20_000:
            cache.clear()
        return out

    def _scan_terms(self, terms, corpus: _Corpus | None = None, query: str = "") -> list:
        """The terms matched word by word: Arabic and Indic always; Latin ones (Indonesian, German, Dutch, Turkish,
        Polish, or a query of undetermined language) only when the store holds facts in one of those languages, so an
        English store never pays for a scan."""
        fts = (corpus or self._current).fts
        scan = [t for t in terms if t.scan]
        if any(t.fts_ok for t in scan) and not _scan_language(query):
            try:
                needed = bool(self.store.read(f"SELECT 1 FROM {fts} WHERE {fts} MATCH ? LIMIT 1", (_SCAN_PROBE,)))
            except Exception:  # a damaged index: the scan does not need it
                needed = True
            if not needed:
                scan = [t for t in scan if not t.fts_ok]
        return scan

    def _rows(self, ids: list[int], category: str | None = None, min_trust: float | None = None,
              corpus: _Corpus | None = None) -> list[dict]:
        """Full rows of *ids*, the filters applied again: the fact may have changed since its text was read."""
        if not ids:
            return []
        corpus = corpus or self._current
        sql = (f"SELECT * FROM {corpus.source} f WHERE f.{corpus.key} IN ({','.join('?' * len(ids))})"
               + (" AND f.trust_score >= ?" if min_trust is not None else "") + (" AND f.category = ?" if category else ""))
        params = [*corpus.params, *ids, *([min_trust] if min_trust is not None else []), *([category] if category else [])]
        return [dict(r) for r in self.store.read(sql, params)]

    def _scan_candidates(self, terms, category: str | None, min_trust: float, limit: int,
                         corpus: _Corpus | None = None, exact: bool = False, query: str = "") -> list[dict]:
        """Facts containing the query's scan terms (Arabic, Indic, Indonesian: words whose other forms the FTS5
        index cannot find), matched word by word against the text of every fact and ranked by the IDF-weighted share
        of those terms they contain ('fts_rank'). A scan, like the n-gram path: milliseconds at the size this store
        is meant for."""
        corpus = corpus or self._current
        scan = self._scan_terms(terms, corpus, query)
        if not scan:
            return []
        texts = self._texts(category, min_trust, corpus)
        strength = (lambda t, a: float(t._exact(a))) if exact else (lambda t, a: t.match(a))
        found = [(rid, trust, {t: s for t in scan if (s := strength(t, a))}) for rid, trust, a in texts]
        df = {t: sum(t in hit for _, _, hit in found) for t in scan}
        weight = {t: math.log(1.0 + len(texts) / df[t]) for t in scan if df[t]}
        total = sum(weight.values()) or 1.0
        # ties: the more trusted fact, then the newer one
        ranked = sorted(((sum(weight[t] * s for t, s in hit.items()) / total, trust, rid) for rid, trust, hit in found
                         if hit), reverse=True)[:limit]
        rank = {rid: score for score, _, rid in ranked}
        rows = [f for f in self._rows([rid for _, _, rid in ranked], category, min_trust, corpus)
                if self._contains_a_term(scan, f, exact=exact)]
        for fact in rows:
            fact["fts_rank"] = rank[fact[corpus.key]]
        return rows

    def _ngram_candidates(self, query: str, category: str | None, min_trust: float, limit: int,
                          corpus: _Corpus | None = None) -> list[dict]:
        """Facts sharing character n-grams with the query's unsegmented runs, ranked by the share of n-grams they
        contain ('fts_rank'). A substring scan: no extra index, so nothing to migrate or keep in sync; at the size
        this store is meant for (hundreds to thousands of facts) it takes milliseconds."""
        corpus = corpus or self._current
        grams = query_grams(query)
        if not grams:
            return []
        # Matched against each fact's folded text (content and tags: the agent's keywords are searched like the text),
        # so Traditional and Simplified characters meet; the analysis is cached per fact, as for the word scan. A
        # Chinese query's pairs are matched as they stand in Chinese facts (gram_coverage, inlined: it runs per fact).
        zh, total = is_chinese(query or ""), sum(weight for _, weight in grams)
        plain = [(g.replace(_text.SEAM, ""), w) for g, w in grams]
        texts = self._texts(category, min_trust, corpus)
        held = ((sum(w for g, w in grams if g in a.seamed) if zh and a.seamed else
                 sum(w for g, w in plain if g in a.text), trust, rid) for rid, trust, a in texts)
        ranked = sorted(((share / total, trust, rid) for share, trust, rid in held), reverse=True)
        rank = {rid: score for score, _, rid in ranked[:limit] if score > 0}
        rows = self._rows(list(rank), category, min_trust, corpus)
        for fact in rows:
            fact["fts_rank"] = rank[fact[corpus.key]]
        return sorted(rows, key=lambda f: f["fts_rank"], reverse=True)

    def _rerank(self, query: str, candidates: list[dict], limit: int) -> list[dict]:
        query_tokens = content_stems(query)
        # Query vector is loop-invariant; encode lazily on the first candidate that carries an HRR vector
        # so stores whose hrr_vector was never backfilled don't pay for it.
        query_vec = None  # bind(encode_text(query), ROLE_CONTENT): the shape a fact's content is stored in
        # Same laziness for the neural embedding: encoded once, only if a candidate carries one.
        query_emb = None
        for fact in candidates:
            jaccard = self._jaccard_similarity(query_tokens, content_stems(f'{fact["content"]} {fact.get("tags") or ""}'))
            hrr_sim = 0.5  # neutral
            fact_vec = self._phases(fact["hrr_vector"]) if self.hrr_weight > 0 and fact.get("hrr_vector") else None
            if fact_vec is not None:
                if query_vec is None:
                    query_vec = hrr.bind(hrr.encode_text(query, self.hrr_dim), self._atom(_ROLE_CONTENT))
                hrr_sim = _shift(hrr.similarity(query_vec, fact_vec))
            emb_sim = 0.5  # neutral
            if self.embedding_weight > 0 and fact.get("embedding_vector"):
                if query_emb is None:
                    query_emb = emb.encode_text(query)
                    if query_emb is None:  # the model went away mid-search: fall back for good
                        self.embedding_weight = 0.0
                if query_emb is not None:
                    stored = emb.bytes_to_vector(bytes(fact["embedding_vector"]))
                    if stored is not None and stored.size == query_emb.size:
                        emb_sim = emb.shift(emb.cosine_similarity(query_emb, stored))
            relevance = (self.fts_weight * fact.get("fts_rank", 0.0) + self.jaccard_weight * jaccard
                         + self.hrr_weight * hrr_sim + self.embedding_weight * emb_sim)
            fact["score"] = relevance * fact["trust_score"]
            if self.half_life > 0:
                fact["score"] *= self._temporal_decay(fact.get("updated_at") or fact.get("created_at"))
        results = sorted(candidates, key=lambda x: x["score"], reverse=True)[:limit]
        for fact in results:
            fact.pop("hrr_vector", None)  # callers expect JSON-serializable dicts
            fact.pop("embedding_vector", None)
        return results

    def _semantic_fallback_allowed(self, query: str) -> bool:
        """May the neural path answer this query when the lexical paths found nothing?

        The lexical layer answers ``[]`` for a reason in several cases — a greeting,
        a query of nothing but function words (``" the of``), a message with no
        informative term — and those must stay empty: an embedding always finds its
        nearest neighbour, so without this gate a "thanks!" would recall a random
        fact. Same two rules the lexical path uses: the query must carry an
        informative term, and it must not be a greeting.
        """
        if self.embedding_weight <= 0 or not (query or "").strip():
            return False
        if is_greeting(query):
            return False
        return any(t.fts_ok or t.scan for t in query_terms(query))

    def _embedding_candidates(self, query: str, category: str | None, min_trust: float, limit: int) -> list[dict]:
        """Facts by neural similarity alone, for when the lexical paths found nothing.

        This is the recall the lexical store cannot do: a Chinese question phrased
        with different words than the fact. Scored exactly like the fused path
        (``embedding_weight`` * similarity * trust), so its results interleave with
        lexical ones rather than outranking them.
        """
        if self.embedding_weight <= 0:
            return []
        query_emb = emb.encode_text(query)
        if query_emb is None:
            return []
        where = "WHERE embedding_vector IS NOT NULL AND trust_score >= ?"
        params: list = [min_trust]
        if category:
            where += " AND category = ?"
            params.append(category)
        rows = self.store.read(f"SELECT {_FACT_COLUMNS}, embedding_vector FROM facts {where}", params)
        scored = []
        for row in rows:
            fact = dict(row)
            stored = emb.bytes_to_vector(bytes(fact.get("embedding_vector") or b""))
            if stored is None or stored.size != query_emb.size:
                continue
            fact["score"] = self.embedding_weight * emb.shift(emb.cosine_similarity(query_emb, stored)) * fact["trust_score"]
            if self.half_life > 0:
                fact["score"] *= self._temporal_decay(fact.get("updated_at") or fact.get("created_at"))
            # Internal marker: relevant() keeps these (only when it too was called
            # with semantic=True) instead of testing them against lexical coverage,
            # which they by construction do not have.
            fact["_semantic_hit"] = True
            fact.pop("embedding_vector", None)
            fact.pop("hrr_vector", None)
            scored.append(fact)
        scored.sort(key=lambda f: f["score"], reverse=True)
        # A floor: an embedding always has a nearest neighbour, so without one a fact
        # that merely shares the model's idea of "general text" would be recalled.
        # Scores carry shift(cos) and trust, so the floor is built the same way.
        floor = self.embedding_weight * emb.shift(_EMB_MIN_COS_SIM) * min_trust
        kept = [f for f in scored if f["score"] >= floor]
        return kept[:limit]

    def _linked_facts(self, entity_groups: list[list[int]], category: str | None, min_trust: float, limit: int) -> list[dict]:
        """Facts linked to at least one entity of EVERY group (each group = the ids one name resolves to)."""
        joins = " INTERSECT ".join(
            f"SELECT fact_id FROM fact_entities WHERE entity_id IN ({','.join('?' * len(g))})" for g in entity_groups)
        params = [i for g in entity_groups for i in g] + [min_trust] + ([category] if category else []) + [limit]
        sql = (f"SELECT {_F_COLUMNS} FROM facts f WHERE f.fact_id IN ({joins}) AND f.trust_score >= ? "
               + ("AND f.category = ? " if category else "") + "ORDER BY f.trust_score DESC, f.updated_at DESC, f.fact_id DESC LIMIT ?")
        results = [dict(r) for r in self.store.read(sql, params)]
        for fact in results:
            fact["score"] = fact["trust_score"]
        return results

    def probe(self, entity: str, category: str | None = None, limit: int = 10, min_trust: float = 0.3) -> list[dict]:
        """Every fact linked to the entity (exact, by the entity index; name or alias, any letter case), most trusted
        first. A name never linked as an entity (e.g. a one-word name) falls back to keyword search."""
        ids = self.store.entity_ids(entity)
        if not ids:
            return self.search(entity, category=category, min_trust=min_trust, limit=limit)
        return self._linked_facts([ids], category, min_trust, limit)

    def reason(self, entities: list[str], category: str | None = None, limit: int = 10, min_trust: float = 0.3) -> list[dict]:
        """Facts linked to ALL the given entities. When some name is not a known entity, keyword search that still
        requires every name (FTS5 AND of the names as phrases), never facts about just one of them."""
        names = [e for e in (entities or []) if isinstance(e, str) and e.strip()]
        if not names:
            return []
        groups = [self.store.entity_ids(name) for name in names]
        if all(groups):
            return self._linked_facts(groups, category, min_trust, limit)
        unknown = [name for name, ids in zip(names, groups) if not ids]
        known = [ids for ids in groups if ids]
        spaced = [n for n in unknown if not UNSEGMENTED.search(n)]
        glued = [n.strip().lower() for n in unknown if UNSEGMENTED.search(n)]
        required = []
        for name in spaced:  # every word of the name, in any word form ("allergy" finds "allergic")
            terms = query_terms(name)
            if not terms:
                return []
            required += terms
        if required:  # every constraint first, then the 500 most trusted (newest on ties)
            allowed = {f["fact_id"] for f in self._linked_facts(known, category, min_trust, 10_000)} if known else None
            hits = sorted(((trust, fact_id) for fact_id, trust, a in self._texts(category, min_trust)
                           if (allowed is None or fact_id in allowed) and all(t.found_in(a) for t in required)),
                          reverse=True)[:500]
            lexical = self._rows([fact_id for _, fact_id in hits], category, min_trust)
        elif known:
            lexical = self._linked_facts(known, category, min_trust, 500)
        else:
            lexical = self._ngram_candidates(" ".join(glued), category, min_trust, 500)
        lexical = [f for f in lexical if all(g in (f.get("content") or "").lower() for g in glued)]
        if known:  # known names by identity (aliases included), unknown names by their words
            allowed = {f["fact_id"] for f in self._linked_facts(known, category, min_trust, 10_000)}
            lexical = [f for f in lexical if f["fact_id"] in allowed]
        return self._rerank(" ".join(names), lexical, limit)

    def related(self, entity: str, category: str | None = None, limit: int = 10, min_trust: float = 0.3) -> list[dict]:
        """Facts structurally connected to an entity (the entity's atom unbinds to a role), above a noise floor.
        Keyword search without numpy, vectors, or any fact over the floor."""
        if not hrr._HAS_NUMPY:
            return self.search(entity, category=category, min_trust=min_trust, limit=limit)
        entity_vec = self._atom(entity.strip().lower())  # bare atom, not role-bound: ANY structural match
        roles = (self._atom(_ROLE_ENTITY), self._atom(_ROLE_CONTENT))  # loop-invariant: encode once
        floor = _related_floor(self.hrr_dim)
        scored = []
        for row in self._vector_rows(category, min_trust):
            fact = dict(row)
            fact_vec = self._phases(fact.pop("hrr_vector"))
            if fact_vec is None:
                continue
            sim = max(hrr.similarity(hrr.unbind(fact_vec, entity_vec), role) for role in roles)
            if sim >= floor:
                fact["score"] = _shift(sim) * fact["trust_score"]
                scored.append(fact)
        if not scored:
            return self.search(entity, category=category, min_trust=min_trust, limit=limit)
        return sorted(scored, key=lambda x: x["score"], reverse=True)[:limit]

    def contradict(self, category: str | None = None, threshold: float = 0.3, limit: int = 10,
                   min_trust: float = 0.0) -> list[dict]:
        """Candidate conflicts: pairs about the same subject (shared linked entities) making a similar but not
        identical claim ("Alice lives in Paris" / "Alice lives in London", a changed number, a flipped preference).
        Score = entity overlap x word overlap of the two claims with the entity names left out. Identical claims,
        a claim that only adds detail to the other ("likes apples" / "likes apples and bananas") and different
        topics about the same subject do not qualify. ``content_similarity`` is that word overlap. Word sets, so
        word order is ignored. Works without numpy."""
        sql = ("SELECT fact_id, content, category, tags, trust_score, created_at, updated_at FROM facts "
               "WHERE trust_score >= ?" + (" AND category = ?" if category else "")
               + " ORDER BY COALESCE(updated_at, created_at) DESC LIMIT 500")  # bound: the most recently changed facts
        rows = self.store.read(sql, [min_trust] + ([category] if category else []))
        if len(rows) < 2:
            return []
        ids = [r["fact_id"] for r in rows]
        names_by_fact: dict[int, set[str]] = {}
        for r in self.store.read("SELECT fe.fact_id, e.name FROM fact_entities fe JOIN entities e ON e.entity_id = fe.entity_id "
                                 f"WHERE fe.fact_id IN ({','.join('?' * len(ids))})", ids):
            names_by_fact.setdefault(r["fact_id"], set()).add(r["name"].lower())
        facts = []  # (public dict, lower-cased entity names, claim)
        for row in rows:
            names = names_by_fact.get(row["fact_id"])
            if names:
                entity_words = {w for name in names for w in words(name)}
                facts.append((dict(row), names, self._claim(row["content"], entity_words)))
        # Only pairs that share an entity can qualify: group by entity instead of comparing every pair.
        by_entity: dict[str, list[int]] = {}
        for index, (_, names, _) in enumerate(facts):
            for name in names:
                by_entity.setdefault(name, []).append(index)
        pairs = sorted({(a, b) for group in by_entity.values() for i, a in enumerate(group) for b in group[i + 1:]})
        contradictions = []
        for i, j in pairs:
            (f1, ents1, claim1), (f2, ents2, claim2) = facts[i], facts[j]
            entity_overlap = len(ents1 & ents2) / len(ents1 | ents2)
            if entity_overlap < 0.3:
                continue  # different subjects
            claim_sim = self._conflicting(claim1, claim2)
            contradiction_score = entity_overlap * claim_sim
            if contradiction_score >= threshold:
                contradictions.append({
                    "fact_a": f1, "fact_b": f2,
                    "entity_overlap": round(entity_overlap, 3),
                    "content_similarity": round(claim_sim, 3),
                    "contradiction_score": round(contradiction_score, 3),
                    "shared_entities": sorted(ents1 & ents2),
                })
        return sorted(contradictions, key=lambda x: x["contradiction_score"], reverse=True)[:limit]

    def _vector_rows(self, category: str | None, min_trust: float = 0.0) -> list:
        """All facts that carry an HRR vector and pass the trust filter, optionally within a category."""
        where = "WHERE hrr_vector IS NOT NULL AND trust_score >= ?" + (" AND category = ?" if category else "")
        return self.store.read(f"SELECT {_FACT_COLUMNS}, hrr_vector FROM facts {where}",
                               [min_trust] + ([category] if category else []))

    def _fts_candidates(self, query: str, category: str | None, min_trust: float, limit: int,
                        corpus: _Corpus | None = None) -> list[dict]:
        """FTS5 candidates for a natural-language query; [] when nothing in it is searchable."""
        expression = self._sanitize_fts_query(query)
        return self._fts_query(expression, category, min_trust, limit, corpus) if expression else []

    def _fts_query(self, expression: str, category: str | None, min_trust: float, limit: int,
                   corpus: _Corpus | None = None) -> list[dict]:
        """Raw FTS5 MATCH candidates with rank normalized to [0, 1] as 'fts_rank'."""
        corpus = corpus or self._current
        fts = corpus.fts
        category_clause = "AND f.category = ? " if category else ""
        params = [*corpus.params, expression] + ([category] if category else []) + [min_trust, limit]
        sql = (f"SELECT f.*, {fts}.rank as fts_rank_raw FROM {fts} JOIN {corpus.source} f ON f.{corpus.key} = {fts}.rowid "
               f"WHERE {fts} MATCH ? {category_clause}AND f.trust_score >= ? ORDER BY {fts}.rank LIMIT ?")
        try:
            results = [dict(row) for row in self.store.read(sql, params)]
        except Exception as exc:  # a damaged index (expressions are built to be valid)
            # Logged, not raised: recall must not take the turn down; a silent [] hides a broken index.
            logger.warning("holographic: FTS5 search failed (%s: %s)", type(exc).__name__, exc)
            return []
        # FTS5 rank is negative (lower = better); normalize |rank| / max to [0, 1] (1e-6 floor avoids div by zero)
        max_rank = max([abs(f["fts_rank_raw"]) for f in results] + [1e-6])
        for fact in results:
            fact["fts_rank"] = abs(fact.pop("fts_rank_raw")) / max_rank
        return results

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Lowercase whitespace tokens with surrounding punctuation stripped (no stemming)."""
        return {c for c in (w.strip(_PUNCT) for w in text.lower().split()) if c} if text else set()

    @staticmethod
    def _fts_phrase(text: str) -> str:
        """Text as one FTS5 phrase (operator characters become spaces); "" when nothing is left."""
        words = [w.strip(_PUNCT) for w in text.lower().translate(_FTS_OPERATORS).split()]
        words = [w for w in words if w]
        return f'"{" ".join(words)}"' if words else ""

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Natural-language query -> FTS5-safe OR expression. FTS5 AND-joins a multi-word MATCH by default, which
        tanks recall on prose: informative terms only (English and Russian stopwords dropped), each as its exact
        word OR its stem as a prefix ("маши" -> ("маши" OR "маш"*)), FTS5 operator characters inside a token
        turned into a phrase ("pve-01" -> "pve 01"). Nothing left -> "" (no search, never a raw MATCH)."""
        return " OR ".join(term.fts(_FTS_OPERATORS) for term in query_terms(query or "") if term.fts_ok)

    # -- prefetch relevance --------------------------------------------------------------------------------------

    def _doc_freq(self, term) -> int:
        """Facts that may hold the term: its own forms, and the words of its root when it has any (the population the
        coverage counts it in: "ночные" is weighed by every fact with a night word, not only by the one that says
        "ночные")."""
        extra = term.fts_extra()
        expression = term.fts(_FTS_OPERATORS) if not extra else f"({term.fts(_FTS_OPERATORS)} OR {extra})"
        try:
            return int(self.store.read("SELECT COUNT(*) FROM facts_fts WHERE facts_fts MATCH ?", (expression,))[0][0])
        except Exception:
            return 0

    def relevant(self, query: str, candidates: list[dict], min_coverage: float = 0.3,
                 semantic: bool = False) -> list[dict]:
        """Keep the candidates that answer the query, not merely share a word with it.

        Coverage = IDF-weighted share of the query's informative terms the fact contains (exact word or same stem).
        Terms no fact contains are left out of the total (they cannot be matched by anything), and a matched
        identifier (pve-01, v0.21.6) is decisive on its own. A query with no informative term ("ok, thanks",
        "да, это так") recalls nothing. Order is preserved."""
        if is_greeting(query):  # "good morning", "доброе утро", "早上好": a courtesy, not a question about mornings
            return []
        # Candidates the neural path already validated carry `_semantic_hit`: they
        # cleared its similarity floor and informative-term gate, which is what this
        # lexical coverage test approximates. Only kept when the caller itself asked
        # for semantic recall (the conversational paths), so the exact-match callers
        # see exactly the upstream result.
        if semantic:
            for f in candidates:
                f.pop("_semantic_hit", None)
        else:
            candidates = [f for f in candidates if not f.pop("_semantic_hit", None)]
            if not candidates:
                return []
        grams, zh = query_grams(query or ""), is_chinese(query or "")
        gram_kept = []
        if grams and candidates:
            # Unsegmented script: weighted coverage of the query's n-grams, or (stricter, since one character says
            # less than a word) IDF-weighted coverage of its meaningful characters: in Chinese a one-character
            # word (猫, cat) or a reordered phrase (名字 / 名叫) shares characters, not bigrams.
            char_weights = self._char_weights(content_chars(query))
            pairs = [(g, w) for g, w in grams if len(g.replace(_text.SEAM, "")) > 1]  # not the one-character words
            char_total = sum(char_weights.values())
            for f in candidates:
                analysis = analyze(f'{f.get("content", "")} {f.get("tags") or ""}')  # lower-cased and folded
                chars = zh_char_text(grams, analysis) if zh else analysis.text  # not the halves of a split pair
                char_cov = sum(w for c, w in char_weights.items() if c in chars) / char_total if char_total else 0.0
                # a Chinese one-character word (猫) never recalls alone: a pair of the question is there too, or
                # (characters one by one) two of its characters ("猫几岁了" / "合同里有猫腻" share 猫 only)
                paired = not zh or not pairs or gram_coverage(pairs, analysis, zh, 1.0) > 0
                two = not zh or len(char_weights) < 2 or sum(1 for c in char_weights if c in chars) >= 2
                if (gram_coverage(grams, analysis, zh) >= min_coverage and paired) or \
                        (char_cov >= max(0.5, min_coverage) and two):
                    gram_kept.append(f)
        terms = [t for t in query_terms(_RE_UNSEGMENTED.sub(" ", query or "")) if t.word]
        identifiers = [t for t in terms if t.identifier]
        if identifiers:  # "hr-ovh" inside Chinese text: the character overlap must not stand in for the host
            gram_kept = [f for f in gram_kept
                         if any(t.found_in(analyze(f'{f.get("content", "")} {f.get("tags") or ""}')) for t in identifiers)]
        if not terms or not candidates:
            return gram_kept
        total_facts = max(self.store.count_facts(), 1)
        scan = self._scan_terms(terms, query=query)
        texts = self._texts() if scan else []
        scan_df = {t: sum(1 for _, _, a in texts if t.found_in(a)) for t in scan}
        weights, unknown = {}, []
        for term in terms:
            df = scan_df[term] if term in scan_df else self._doc_freq(term)
            if df:
                weights[term] = math.log(1.0 + total_facts / df)
            else:
                unknown.append(term)
        if not weights:
            return gram_kept
        # A word no fact contains still says what the message is about: it counts against coverage, weighted like the
        # most common known word, so "do submarines deploy on Jupiter?" is not "about" a fact that mentions deploy.
        total = sum(weights.values()) + len(unknown) * min(weights.values())
        kept = []
        for fact in candidates:
            analysis = analyze(f'{fact.get("content", "")} {fact.get("tags") or ""}')
            if identifiers and not any(t.found_in(analysis) for t in identifiers):
                continue  # a message naming pve-01 or v0.21.6 is about that, not about pve-010
            strength = {t: t.match(analysis) for t in weights}  # 1.0 the word itself, _DERIV_WEIGHT its root
            # A root alone never recalls: suffix rules cannot tell a derivation from a look-alike ("войной" / "вой",
            # "полный" / "пол"), so a fact needs a word of the message itself, and its roots only add to that.
            exact = any(s == 1.0 for s in strength.values())
            if any(t.identifier and s for t, s in strength.items()) or fact in gram_kept \
                    or (exact and sum(weights[t] * s for t, s in strength.items()) / total >= min_coverage):
                kept.append(fact)
        return kept

    def _char_weights(self, chars: list[str]) -> dict[str, float]:
        """IDF weight of each character over the store; a character no fact contains weighs like the most common
        one that is present (it still says what the message is about)."""
        if not chars:
            return {}
        texts = [a.text for _, _, a in self._texts()]  # folded: 醫 counts where 医 is written
        total = max(len(texts), 1)
        weights, missing = {}, []
        for c in chars[:16]:
            df = sum(1 for t in texts if c in t)
            if df:
                weights[c] = math.log(1.0 + total / df)
            else:
                missing.append(c)
        if weights:
            floor = min(weights.values())
            weights.update({c: floor for c in missing})
        return weights

    # -- write-time checks and housekeeping ----------------------------------------------------------------------

    @staticmethod
    def _claim(text: str, entity_words: set[str]) -> tuple[tuple, bool]:
        """(the claim without the entity names, negators and function words, negated?). Words compare by their keys
        (word and stems); unsegmented text (Chinese, Thai) by its characters, which is what a short claim shares with
        a contradicting one."""
        claim, negated, contrast = [], False, False
        for word in analyze(text).words:
            w = word.text
            if w in entity_words:
                continue
            if UNSEGMENTED.search(w):
                for name in entity_words:
                    if UNSEGMENTED.search(name):
                        w = w.replace(name, " ")
                if is_sea(w):  # Thai letters mean nothing alone: trigrams, and "ไม่" (not)
                    negated = negated or ("ไม่" in w and not contrast)
                    claim += [frozenset({g}) for run in UNSEGMENTED.findall(w) for g, weight in run_grams(run) if weight]
                    continue
                for ch in w:
                    if ch in CJK_NEGATORS:
                        negated = negated or not contrast
                    elif ch.isalnum():
                        claim.append(frozenset({ch}))
            elif w in CONTRASTIVE:  # "likes cats but not dogs": the negation is another clause's
                contrast = True
            elif w in NEGATORS:
                negated = negated or not contrast
            elif not word.stop or w in CLAIM_WORDS:  # "with" / "without" sugar is the claim
                claim.append(word.keys)
        return tuple(dict.fromkeys(claim)), negated

    @classmethod
    def _conflicting(cls, a: tuple[frozenset, bool], b: tuple[frozenset, bool]) -> float:
        """Claim overlap when two claims about one subject look like a conflict, else 0: the same statement with
        the opposite polarity ("likes apples" / "does not like apples": at least 60 % of the shorter claim is in the
        other, so "eats meat every day" / "does not eat meat" counts and "likes cats" / "does not like cooking" does
        not), or a similar but different statement where neither only adds detail to the other ("lives in Paris" /
        "lives in London"). Lexical candidates for a person to judge; word order (who paid whom) is not modelled."""
        (claim_a, neg_a), (claim_b, neg_b) = a, b
        a_in_b, b_in_a = soft_overlap(list(claim_a), list(claim_b))
        if neg_a != neg_b:
            shorter = min(len(claim_a), len(claim_b))
            contained = min(a_in_b, b_in_a) / shorter if shorter else 0.0
            return contained if contained >= 0.6 else 0.0
        overlap = soft_jaccard(list(claim_a), list(claim_b))
        if a_in_b == len(claim_a) or b_in_a == len(claim_b):  # identical, or one only adds detail
            return 0.0
        return overlap

    def write_check(self, fact_id: int, limit: int = 3) -> dict:
        """Facts that look like a near-duplicate of fact *fact_id* or a possible conflict with it (advisory: the
        agent decides; nothing is changed). Candidates: facts sharing an entity, plus the keyword neighbours."""
        rows = self.store.read("SELECT fact_id, content FROM facts WHERE fact_id = ?", (fact_id,))
        if not rows:
            return {"similar": [], "possible_conflicts": []}
        content = rows[0]["content"]
        own_entities = {r[0] for r in self.store.read("SELECT entity_id FROM fact_entities WHERE fact_id = ?", (fact_id,))}
        candidate_ids = {r[0] for r in self.store.read(
            "SELECT DISTINCT fact_id FROM fact_entities WHERE entity_id IN ({}) AND fact_id != ? ORDER BY fact_id DESC LIMIT 50".format(
                ",".join("?" * len(own_entities)) or "NULL"), [*own_entities, fact_id])}
        candidate_ids |= {f["fact_id"] for f in self.search(content, min_trust=0.0, limit=10, roots=False)} - {fact_id}
        if not candidate_ids:
            return {"similar": [], "possible_conflicts": [],
                    "past_versions": self._past_versions(fact_id, content, content_keys(content), limit)}
        entity_names = {r["entity_id"]: r["name"] for r in self.store.read(
            "SELECT entity_id, name FROM entities WHERE entity_id IN (SELECT entity_id FROM fact_entities WHERE fact_id IN ({}))".format(
                ",".join("?" * (len(candidate_ids) + 1))), [fact_id, *sorted(candidate_ids)])}
        links: dict[int, set[int]] = {}
        for r in self.store.read("SELECT fact_id, entity_id FROM fact_entities WHERE fact_id IN ({})".format(
                ",".join("?" * len(candidate_ids))), sorted(candidate_ids)):
            links.setdefault(r["fact_id"], set()).add(r["entity_id"])
        own_words = {w for e in own_entities for w in words(entity_names.get(e, ""))}
        own_keys, own_claim = content_keys(content), self._claim(content, own_words)
        similar, conflicts = [], []
        for other in self.store.read("SELECT fact_id, content FROM facts WHERE fact_id IN ({})".format(
                ",".join("?" * len(candidate_ids))), sorted(candidate_ids)):
            other_entities = links.get(other["fact_id"], set())
            overlap = soft_jaccard(own_keys, content_keys(other["content"]))
            entry = {"fact_id": other["fact_id"], "content": other["content"], "word_overlap": round(overlap, 2)}
            shared = own_entities & other_entities
            other_words = {w for e in other_entities for w in words(entity_names.get(e, ""))}
            other_claim = self._claim(other["content"], other_words)
            score = self._conflicting(own_claim, other_claim)
            if (shared and score >= 0.3) or (not own_entities and not other_entities and score >= 0.5):
                conflicts.append(entry)  # before duplicates: "budget is 500" / "budget is 600" is an update
            elif overlap >= 0.7 and own_claim[1] == other_claim[1]:
                similar.append(entry)
        by_overlap = lambda xs: sorted(xs, key=lambda e: e["word_overlap"], reverse=True)[:limit]  # noqa: E731
        return {"similar": by_overlap(similar), "possible_conflicts": by_overlap(conflicts),
                "past_versions": self._past_versions(fact_id, content, own_keys, limit)}

    def near_duplicate(self, content: str) -> int | None:
        """A current fact that says nearly what *content* says — write_check's "similar" (word overlap of 0.7 or
        more, the same polarity), checked before anything is stored — or None. A possible conflict is not a duplicate:
        "the budget is 500" / "the budget is 600" is a change."""
        own_keys, own_claim = content_keys(content), self._claim(content, set())
        for other in self.search(content, min_trust=0.0, limit=10, roots=False):
            other_claim = self._claim(other["content"], set())
            if soft_jaccard(own_keys, content_keys(other["content"])) >= 0.7 and own_claim[1] == other_claim[1] \
                    and self._conflicting(own_claim, other_claim) < 0.3:
                return other["fact_id"]
        return None

    def _past_versions(self, fact_id: int, content: str, own_keys, limit: int) -> list[dict]:
        """Closed versions (changed / ended, never corrected) of other facts that say nearly the same: "a fact that
        came back" is a new version of that fact (change), not a fact of its own."""
        expression = self._sanitize_fts_query(content)
        if not expression or not self.store.history_ready():
            return []
        marks = ",".join("?" * len(CLOSED))
        try:
            rows = self.store.read(
                "SELECT v.fact_id, v.seq, v.content, v.status, v.valid_to, v.valid_to_exact FROM fact_versions_fts "
                "JOIN fact_versions v ON v.version_id = fact_versions_fts.rowid WHERE fact_versions_fts MATCH ? "
                f"AND v.status IN ({marks}) AND v.fact_id != ? ORDER BY fact_versions_fts.rank LIMIT 20",
                (expression, *CLOSED, fact_id))
        except Exception as exc:  # a damaged index: the hint is advisory
            logger.warning("holographic: history search failed (%s)", exc)
            return []
        out: dict[int, dict] = {}
        for r in rows:
            overlap = soft_jaccard(own_keys, content_keys(r["content"]))
            if overlap >= 0.7 and r["fact_id"] not in out:
                until = r["valid_to"] if r["valid_to_exact"] else f"before {r['valid_to']}"
                out[r["fact_id"]] = {"fact_id": r["fact_id"], "content": r["content"], "status": r["status"],
                                     "until": until, "word_overlap": round(overlap, 2)}
        return sorted(out.values(), key=lambda e: e["word_overlap"], reverse=True)[:limit]

    def review(self, limit: int = 20, stale_days: int = 60) -> list[dict]:
        """Clean-up candidates with a reason each: low trust; never recalled and older than *stale_days*; a
        near-duplicate of a more trusted fact. Advisory: nothing is removed."""
        out, seen = [], set()
        def add(fact, reason):
            if fact["fact_id"] not in seen:
                seen.add(fact["fact_id"])
                out.append({"fact_id": fact["fact_id"], "content": fact["content"], "trust_score": fact["trust_score"],
                            "retrieval_count": fact["retrieval_count"], "reason": reason})
        for fact in self.store.read(f"SELECT {_FACT_COLUMNS} FROM facts WHERE trust_score < 0.3 ORDER BY trust_score LIMIT ?", (limit,)):
            add(dict(fact), "low trust (rated unhelpful)")
        for fact in self.store.read(
                f"SELECT {_FACT_COLUMNS} FROM facts WHERE retrieval_count = 0 AND julianday('now') - julianday(created_at) > ? "
                "ORDER BY created_at LIMIT ?", (stale_days, limit)):
            add(dict(fact), f"never recalled in {stale_days}+ days")
        if len(out) >= limit:
            return out[:limit]
        # near-duplicates among the most recent 200 facts (bounded pair scan), the more trusted one kept
        facts = sorted((dict(r) for r in self.store.read(f"SELECT {_FACT_COLUMNS} FROM facts ORDER BY fact_id DESC LIMIT 200")),
                       key=lambda f: f["trust_score"], reverse=True)
        keyed = [(f, content_stems(f["content"])) for f in facts]
        for i, (better, a) in enumerate(keyed):
            for worse, b in keyed[i + 1:]:
                if self._jaccard_similarity(a, b) >= 0.8:
                    add(worse, f"near-duplicate of #{better['fact_id']}")
        return out[:limit]

    @staticmethod
    def _jaccard_similarity(set_a: set, set_b: set) -> float:
        """Jaccard similarity coefficient: |A ∩ B| / |A ∪ B|."""
        return len(set_a & set_b) / len(set_a | set_b) if set_a and set_b else 0.0

    def _temporal_decay(self, timestamp_str: str | None) -> float:
        """0.5^(age_days / half_life); 1.0 if disabled, missing, unparseable, or in the future."""
        if not self.half_life or not timestamp_str:
            return 1.0
        try:
            ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00")) if isinstance(timestamp_str, str) else timestamp_str
            age_days = (datetime.now(timezone.utc) - (ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc))).total_seconds() / 86400
            return 1.0 if age_days < 0 else math.pow(0.5, age_days / self.half_life)
        except (ValueError, TypeError):
            return 1.0
