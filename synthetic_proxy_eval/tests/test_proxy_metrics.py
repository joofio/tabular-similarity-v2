"""Tests for proxy metric implementations."""

from __future__ import annotations

from eval_harness.proxy_metrics import compute_proxy_metrics


def test_proxy_metrics_identity():
    order = ["a", "b", "c", "d", "e"]
    ranks = {name: idx + 1 for idx, name in enumerate(order)}
    metrics = compute_proxy_metrics(
        order,
        order,
        ranks,
        ranks,
        ndcg_ks=[3, 5],
        rbo_ps=[0.9],
        jaccard_ks=[3],
        overlap_ks=[3],
    )
    for value in metrics.values():
        assert abs(value - 1.0) < 1e-6


def test_proxy_metrics_bounds():
    order = ["a", "b", "c", "d", "e"]
    reverse = list(reversed(order))
    ranks_a = {name: idx + 1 for idx, name in enumerate(order)}
    ranks_b = {name: idx + 1 for idx, name in enumerate(reverse)}
    metrics = compute_proxy_metrics(
        order,
        reverse,
        ranks_a,
        ranks_b,
        ndcg_ks=[3, 5],
        rbo_ps=[0.9],
        jaccard_ks=[3],
        overlap_ks=[3],
    )
    for value in metrics.values():
        assert 0.0 <= value <= 1.0
