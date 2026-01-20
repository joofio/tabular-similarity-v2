"""Proxy metrics for ranking comparison."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy import stats


def _aligned_rank_arrays(ranks_a: Dict[str, float], ranks_b: Dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """Align rank dictionaries by feature name."""
    keys = sorted(ranks_a.keys())
    a = np.array([ranks_a[k] for k in keys], dtype=float)
    b = np.array([ranks_b[k] for k in keys], dtype=float)
    return a, b


def _clamp_unit(value: float) -> float:
    """Clamp a metric to [0, 1]."""
    return float(max(0.0, min(1.0, value)))


def ndcg_at_k(ref_order: List[str], cand_order: List[str], k: int) -> float:
    """Compute NDCG@k with relevance from reference ranking."""
    k = min(k, len(ref_order), len(cand_order))
    if k == 0:
        return 0.0
    relevance = {feat: 1.0 / (idx + 1) for idx, feat in enumerate(ref_order)}

    def dcg(order: List[str]) -> float:
        score = 0.0
        for i, feat in enumerate(order[:k], start=1):
            rel = relevance.get(feat, 0.0)
            score += rel / np.log2(i + 1)
        return score

    ideal = dcg(ref_order)
    if ideal == 0:
        return 0.0
    return _clamp_unit(dcg(cand_order) / ideal)


def kendall_tau_b(ranks_a: Dict[str, float], ranks_b: Dict[str, float]) -> float:
    """Kendall tau-b similarity normalized to [0, 1]."""
    a, b = _aligned_rank_arrays(ranks_a, ranks_b)
    tau, _ = stats.kendalltau(a, b)
    if np.isnan(tau):
        return 0.0
    return _clamp_unit((tau + 1.0) / 2.0)


def weighted_kendall_tau(ranks_a: Dict[str, float], ranks_b: Dict[str, float]) -> float:
    """Weighted Kendall tau similarity with top-weighted weigher."""
    a, b = _aligned_rank_arrays(ranks_a, ranks_b)

    def weigher(rank: int) -> float:
        return 1.0 / (1.0 + rank)

    tau, _ = stats.weightedtau(a, b, weigher=weigher)
    if np.isnan(tau):
        return 0.0
    return _clamp_unit((tau + 1.0) / 2.0)


def rbo(ref_order: List[str], cand_order: List[str], p: float) -> float:
    """Rank-biased overlap for two ordered lists."""
    k = max(len(ref_order), len(cand_order))
    if k == 0:
        return 0.0
    score = 0.0
    for d in range(1, k + 1):
        overlap = len(set(ref_order[:d]) & set(cand_order[:d]))
        score += (overlap / d) * (p ** (d - 1))
    overlap_k = len(set(ref_order[:k]) & set(cand_order[:k]))
    return _clamp_unit((1.0 - p) * score + (overlap_k / k) * (p ** k))


def spearman_footrule_similarity(ranks_a: Dict[str, float], ranks_b: Dict[str, float]) -> float:
    """Spearman footrule similarity based on L1 distance."""
    a, b = _aligned_rank_arrays(ranks_a, ranks_b)
    n = len(a)
    if n == 0:
        return 0.0
    dist = float(np.sum(np.abs(a - b)))
    max_dist = float(np.sum(np.abs(np.arange(1, n + 1) - np.arange(n, 0, -1))))
    if max_dist == 0:
        return 1.0
    return _clamp_unit(1.0 - dist / max_dist)


def spearman_rho_similarity(ranks_a: Dict[str, float], ranks_b: Dict[str, float]) -> float:
    """Spearman rho similarity based on L2 distance."""
    a, b = _aligned_rank_arrays(ranks_a, ranks_b)
    n = len(a)
    if n == 0:
        return 0.0
    dist = float(np.sqrt(np.sum((a - b) ** 2)))
    max_dist = float(np.sqrt(np.sum((np.arange(1, n + 1) - np.arange(n, 0, -1)) ** 2)))
    if max_dist == 0:
        return 1.0
    return _clamp_unit(1.0 - dist / max_dist)


def jaccard_at_k(ref_order: List[str], cand_order: List[str], k: int) -> float:
    """Jaccard overlap for top-k sets."""
    k = min(k, len(ref_order), len(cand_order))
    if k == 0:
        return 0.0
    a = set(ref_order[:k])
    b = set(cand_order[:k])
    denom = len(a | b)
    if denom == 0:
        return 0.0
    return _clamp_unit(len(a & b) / denom)


def overlap_at_k(ref_order: List[str], cand_order: List[str], k: int) -> float:
    """Overlap@k (Recall@k) for top-k sets."""
    k = min(k, len(ref_order), len(cand_order))
    if k == 0:
        return 0.0
    a = set(ref_order[:k])
    b = set(cand_order[:k])
    return _clamp_unit(len(a & b) / k)


def _levenshtein_distance(a: str, b: str) -> int:
    """Compute Levenshtein edit distance."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def levenshtein_similarity(ref_order: List[str], cand_order: List[str]) -> float:
    """String baseline; not ranking-aware."""
    ref_str = "|".join(ref_order)
    cand_str = "|".join(cand_order)
    dist = _levenshtein_distance(ref_str, cand_str)
    denom = max(len(ref_str), len(cand_str), 1)
    return _clamp_unit(1.0 - dist / denom)


def compute_proxy_metrics(
    ref_order: List[str],
    cand_order: List[str],
    ref_ranks: Dict[str, float],
    cand_ranks: Dict[str, float],
    ndcg_ks: List[int],
    rbo_ps: List[float],
    jaccard_ks: List[int],
    overlap_ks: List[int],
) -> Dict[str, float]:
    """Compute the full set of proxy metrics."""
    metrics: Dict[str, float] = {}
    for k in ndcg_ks:
        metrics[f"ndcg@{k}"] = ndcg_at_k(ref_order, cand_order, k)
    metrics["kendall_tau_b"] = kendall_tau_b(ref_ranks, cand_ranks)
    metrics["weighted_kendall_tau"] = weighted_kendall_tau(ref_ranks, cand_ranks)
    for p in rbo_ps:
        metrics[f"rbo_p{p}"] = rbo(ref_order, cand_order, p)
    metrics["spearman_footrule"] = spearman_footrule_similarity(ref_ranks, cand_ranks)
    metrics["spearman_rho"] = spearman_rho_similarity(ref_ranks, cand_ranks)
    for k in jaccard_ks:
        metrics[f"jaccard@{k}"] = jaccard_at_k(ref_order, cand_order, k)
    for k in overlap_ks:
        metrics[f"overlap@{k}"] = overlap_at_k(ref_order, cand_order, k)
    metrics["levenshtein_string_baseline"] = levenshtein_similarity(ref_order, cand_order)
    return metrics
