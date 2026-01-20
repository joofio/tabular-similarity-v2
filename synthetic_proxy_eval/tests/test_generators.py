"""Tests for synthetic data generators."""

from __future__ import annotations

from eval_harness.generators import BootstrapGenerator, GaussianCopulaGenerator


def test_bootstrap_generator(toy_df, toy_schema):
    gen = BootstrapGenerator()
    gen.fit(toy_df, toy_schema, seed=0)
    sample = gen.sample(50, seed=1)
    assert len(sample) == 50
    assert list(sample.columns) == list(toy_df.columns)


def test_gaussian_copula_generator_determinism(toy_df, toy_schema):
    gen = GaussianCopulaGenerator()
    gen.fit(toy_df, toy_schema, seed=0)
    sample_a = gen.sample(30, seed=1)
    sample_b = gen.sample(30, seed=1)
    assert sample_a.equals(sample_b)
