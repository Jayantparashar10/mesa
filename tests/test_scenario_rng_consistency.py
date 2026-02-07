"""Test Scenario.rng consistency across different initialization paths."""

import numpy as np

from mesa.experimental.scenarios import Scenario
from mesa.model import Model


def test_scenario_rng_consistency():
    """Verify that Scenario.rng is consistent after the Generator fix.

    This addresses the issue where Scenario.rng differed between:
    - Model(seed=42) vs Model(rng=42)
    - Multiple Model(rng=np.random.default_rng(42)) instances
    """
    # Test 1: Scenario.rng for integer rng matches seed
    print("=== Test 1: Integer rng vs seed ===")
    m_seed = Model(seed=42)
    m_rng_int = Model(rng=42)

    print(f"Model(seed=42).scenario.rng: {m_seed.scenario.rng}")
    print(f"Model(rng=42).scenario.rng: {m_rng_int.scenario.rng}")
    assert m_seed.scenario.rng == m_rng_int.scenario.rng == 42, (
        "Scenario.rng should be 42 for both seed=42 and rng=42"
    )
    print("✅ Scenario.rng matches for integer rng\n")

    # Test 2: Scenario.rng is reproducible with Generator
    print("=== Test 2: Generator reproducibility ===")
    m_gen1 = Model(rng=np.random.default_rng(42))
    m_gen2 = Model(rng=np.random.default_rng(42))

    print(f"First Model(Generator(42)).scenario.rng: {m_gen1.scenario.rng}")
    print(f"Second Model(Generator(42)).scenario.rng: {m_gen2.scenario.rng}")
    assert m_gen1.scenario.rng == m_gen2.scenario.rng, (
        "Scenario.rng should be identical for same Generator seed"
    )
    assert isinstance(m_gen1.scenario.rng, (int, np.integer)), (
        "Scenario.rng should be an integer"
    )
    print("✅ Scenario.rng is reproducible with Generator\n")

    # Test 3: Scenario.rng propagates to model correctly
    print("=== Test 3: Scenario → Model RNG propagation ===")

    # Create scenario with rng
    scenario = Scenario(rng=123)
    model = Model(scenario=scenario)

    print(f"Scenario.rng: {scenario.rng}")
    print(f"Model scenario.rng: {model.scenario.rng}")
    assert model.scenario.rng == 123, (
        "Model should preserve Scenario.rng when scenario is passed"
    )
    print("✅ Scenario.rng propagates correctly\n")

    # Test 4: Model creates scenario with correct rng when none provided
    print("=== Test 4: Auto-created Scenario ===")
    m = Model(rng=999)
    print(f"Model(rng=999).scenario.rng: {m.scenario.rng}")
    assert m.scenario.rng == 999, (
        "Auto-created Scenario should receive the model's seed"
    )
    print("✅ Auto-created Scenario gets correct rng\n")


if __name__ == "__main__":
    test_scenario_rng_consistency()
    print("\n🎉 All Scenario.rng consistency tests passed!")
