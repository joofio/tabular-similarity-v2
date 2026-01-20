"""Golden snapshot tests for proxy metrics."""

from __future__ import annotations

from eval_harness.proxy_metrics import compute_proxy_metrics, rbo


def test_golden_proxy_metrics_snapshot():
    ref = ["a", "b", "c"]
    cand = ["a", "c", "b"]
    ranks_ref = {"a": 1, "b": 2, "c": 3}
    ranks_cand = {"a": 1, "b": 3, "c": 2}

    metrics = compute_proxy_metrics(
        ref,
        cand,
        ranks_ref,
        ranks_cand,
        ndcg_ks=[3],
        rbo_ps=[0.9],
        jaccard_ks=[2],
        overlap_ks=[2],
    )

    assert abs(metrics["jaccard@2"] - 1.0 / 3.0) < 1e-6
    assert abs(metrics["overlap@2"] - 0.5) < 1e-6
    assert abs(metrics["levenshtein_string_baseline"] - 0.6) < 1e-6
    assert abs(metrics["spearman_footrule"] - 0.5) < 1e-6
    assert abs(metrics["spearman_rho"] - 0.5) < 1e-6
    assert abs(metrics["rbo_p0.9"] - rbo(ref, cand, 0.9)) < 1e-12
    assert abs(metrics["rbo_p0.9"] - 0.955) < 1e-3
