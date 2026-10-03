"""Retrieval metrics: hits@K and product-macro / micro R@K (D20 section 9, eval contract v1).

Task: query image -> ranked, deduplicated list of gallery product_ids. hit@K(query) = 1
if the query's truth product is within the top K, else 0.

- Primary metric is product-macro R@K: average a product's queries' hits, then average
  over products (so a product with many query photos isn't overweighted).
- Secondary metric is micro R@K: average over all queries directly.

Both are reported; they are not required to agree, and generally will not when products
have unequal query counts.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

DEFAULT_KS: tuple[int, ...] = (1, 5, 10, 100)


@dataclass(frozen=True, slots=True)
class QueryResult:
    """One query's ground truth and its ranked, product-deduplicated candidate list."""

    query_id: str
    truth_product_id: str
    ranked_product_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(set(self.ranked_product_ids)) != len(self.ranked_product_ids):
            raise ValueError(f"query {self.query_id!r}: ranked_product_ids contains duplicates")


@dataclass(frozen=True, slots=True)
class ProductHits:
    """Per-product aggregate: hit counts at each K, summed over that product's queries.

    ``hit_sums`` maps K -> number of the product's queries that hit within top K.
    ``n_queries`` is how many queries have this product as their truth product.

    A product with zero queries cannot arise from grouping real ``QueryResult``s (every
    product in the grouping is discovered *because* some query names it as truth), so
    this is rejected defensively: it signals a bug in the caller, not a real eval case.
    """

    product_id: str
    hit_sums: Mapping[int, int]
    n_queries: int

    def __post_init__(self) -> None:
        if self.n_queries <= 0:
            raise ValueError(f"product {self.product_id!r} has zero queries")


def _validate_ks(ks: Sequence[int]) -> tuple[int, ...]:
    ks_t = tuple(ks)
    if not ks_t:
        raise ValueError("ks must not be empty")
    for k in ks_t:
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError(f"k must be a positive integer, got {k!r}")
    return ks_t


def hits_at_k(truth: str, ranked: Sequence[str], ks: Sequence[int] = DEFAULT_KS) -> dict[int, int]:
    """hit@K for a single query: 1 if ``truth`` is within the top K of ``ranked``, else 0.

    Raises ``ValueError`` if ``ranked`` contains the same product_id more than once
    (the contract requires product-level, deduplicated rankings).
    """
    ks_t = _validate_ks(ks)
    ranked_t = tuple(ranked)
    if len(set(ranked_t)) != len(ranked_t):
        raise ValueError("ranked product list contains duplicates")
    return {k: 1 if truth in ranked_t[:k] else 0 for k in ks_t}


def _group_by_product(results: Sequence[QueryResult], ks: Sequence[int]) -> dict[str, ProductHits]:
    ks_t = _validate_ks(ks)
    raw_hits: dict[str, dict[int, int]] = defaultdict(lambda: dict.fromkeys(ks_t, 0))
    raw_counts: dict[str, int] = defaultdict(int)
    for r in results:
        hits = hits_at_k(r.truth_product_id, r.ranked_product_ids, ks_t)
        for k in ks_t:
            raw_hits[r.truth_product_id][k] += hits[k]
        raw_counts[r.truth_product_id] += 1
    return {
        pid: ProductHits(product_id=pid, hit_sums=raw_hits[pid], n_queries=raw_counts[pid])
        for pid in raw_counts
    }


def recall_at_k(results: Sequence[QueryResult], ks: Sequence[int] = DEFAULT_KS) -> dict:
    """Product-macro and micro R@K over ``results`` (D20 section 9).

    Returns ``{"macro": {k: v}, "micro": {k: v}, "n_products": int, "n_queries": int}``.
    Raises ``ValueError`` on empty ``results`` or on any query's duplicated ranking;
    never drops a query silently.
    """
    if not results:
        raise ValueError("recall_at_k: results must not be empty")
    ks_t = _validate_ks(ks)
    groups = _group_by_product(results, ks_t)
    n_products = len(groups)
    n_queries = len(results)

    macro: dict[int, float] = {}
    micro: dict[int, float] = {}
    for k in ks_t:
        per_product_recall = [g.hit_sums[k] / g.n_queries for g in groups.values()]
        macro[k] = sum(per_product_recall) / n_products
        total_hits = sum(g.hit_sums[k] for g in groups.values())
        micro[k] = total_hits / n_queries

    return {"macro": macro, "micro": micro, "n_products": n_products, "n_queries": n_queries}
