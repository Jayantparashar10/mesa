"""Test to verify Generator reproducibility fix."""

import numpy as np

from mesa.model import Model


def test_generator_reproducibility():
    """Test that passing the same Generator produces reproducible results.

    This test verifies the fix for the issue where:
    - Model(rng=np.random.default_rng(42)) was non-reproducible
    - The code was calling np.random.default_rng() on an already-created Generator

    The fix ensures that if rng is already a Generator, it's used directly.
    """
    # Test 1: Generator reproducibility
    gen1 = np.random.default_rng(42)
    gen2 = np.random.default_rng(42)

    m1 = Model(rng=gen1)
    m2 = Model(rng=gen2)

    # Both should have the same derived seed
    assert m1._seed == m2._seed, (
        "Models with same Generator seed should have same _seed"
    )

    # Both should produce same random values
    assert m1.random.random() == m2.random.random(), (
        "stdlib random should be reproducible"
    )
    assert m1.rng.random() == m2.rng.random(), "numpy rng should be reproducible"

    # Scenario should have the same rng value
    assert m1.scenario.rng == m2.scenario.rng, "Scenario.rng should match"


def test_integer_rng_matches_seed():
    """Test that rng=42 produces identical results to seed=42.

    This verifies the correct migration path from deprecated seed to rng.
    """
    m_seed = Model(seed=42)
    m_rng = Model(rng=42)

    # Both should have seed=42
    assert m_seed._seed == 42
    assert m_rng._seed == 42

    # Both should produce identical random values
    assert m_seed.random.random() == m_rng.random.random()
    assert m_seed.rng.random() == m_rng.rng.random()

    # Scenario should match
    assert m_seed.scenario.rng == m_rng.scenario.rng == 42


def test_generator_not_mutated():
    """Test that passing a Generator doesn't mutate it in unexpected ways.

    The Generator will be consumed once to derive a seed for stdlib random,
    but should be used consistently after that.
    """
    gen = np.random.default_rng(42)

    # Get first few values from the generator
    expected_values = [gen.random() for _ in range(3)]

    # Create a new generator with same seed and pass to model
    gen2 = np.random.default_rng(42)
    model = Model(rng=gen2)

    # The model's rng should produce the same sequence (after the one value used for seed)
    # First value was consumed for deriving stdlib random seed
    for i, expected in enumerate(expected_values[1:], 1):
        actual = model.rng.random()
        assert actual == expected, f"Value {i} should match: {actual} != {expected}"


def test_reset_rng_with_generator():
    """Test that reset_rng() also properly handles Generator instances."""
    # Create two models
    model1 = Model(rng=42)
    model2 = Model(rng=42)

    # Reset both with the same Generator seed
    model1.reset_rng(rng=np.random.default_rng(99))
    model2.reset_rng(rng=np.random.default_rng(99))

    # Both should produce same sequence after reset
    assert model1.rng.random() == model2.rng.random(), (
        "reset_rng with Generator should be reproducible"
    )

    # Verify multiple values match
    for _ in range(3):
        assert model1.rng.random() == model2.rng.random()

    # Test that Generator is reused, not re-wrapped
    gen = np.random.default_rng(55)
    model1.reset_rng(rng=gen)
    # The generator instance should be the same
    assert model1.rng is gen, "reset_rng should reuse Generator instance"


if __name__ == "__main__":
    test_generator_reproducibility()
    test_integer_rng_matches_seed()
    test_generator_not_mutated()
    print("✅ All tests passed!")
