from __future__ import annotations

from dataclasses import asdict

import numpy as np

from src.corrected_numerical_audit import CorrectedAuditConfig, _cov, _family_estimates_from_cov, _ranked_joint_arrays
from src.corrected_pilot_model import PilotConfig
from src.final_numerical_audit import ELC_LEVELS, _params_for_config, _segment, _history
from src.inter_elc_arm import (
    ACTIVE_SOURCE_LEVELS,
    CORRECTED_INTER_ELC_FINAL_SEEDS_V2,
    INTER_ELC_FINAL_SEEDS,
    InterELCBMatrices,
    InterELCSourceMeans,
    approved_inter_elc_source_means,
    build_inter_elc_table,
    estimate_source_state_means,
    final_inter_elc_seeds_are_untouched,
    frozen_epigenetic_only_inter_elc_B,
    inter_elc_b_matrix_inventory,
    inter_elc_epigenetic_isolation_grid,
    inter_elc_family,
    inter_elc_small_micro_eco_grid,
    one_to_one_derangement,
    scaled_epigenetic_B,
    simulate_inter_elc_seed,
    zero_b_initializer_matches_primary,
)
from src.recalibration_audit import calibration_candidates


def tiny_inter_elc_config() -> CorrectedAuditConfig:
    return CorrectedAuditConfig(
        seeds=(9091,),
        n_lineages=24,
        n_generations=6,
        burn_in_generations=2,
        source_tau_start=2,
        source_tau_stop=4,
        history_order=2,
        n_bootstrap=3,
        n_null=3,
        parameter_overrides=calibration_candidates()["C8_stronger_growth_allocation"],
    )


def test_one_to_one_derangement_has_no_self_pairs_and_uses_each_source_once() -> None:
    pairs = one_to_one_derangement(100, 123)
    assert pairs.shape == (100,)
    assert np.unique(pairs).size == 100
    assert np.all(pairs != np.arange(100))


def test_zero_B_initializer_matches_primary_initializer() -> None:
    config = tiny_inter_elc_config().pilot_config(9091)
    par = _params_for_config(tiny_inter_elc_config())
    assert zero_b_initializer_matches_primary(config, par)


def test_B_inventory_documents_only_same_level_active_terms() -> None:
    b = InterELCBMatrices(b_micro=0.15, b_eco=0.20)
    inv = inter_elc_b_matrix_inventory(b)
    active = inv[inv["B_matrix_entry"].fillna(0.0).abs() > 0.0]
    assert set(active["level_l"]) == set(ACTIVE_SOURCE_LEVELS)
    assert len(active[active["level_l"] == "epigenetic"]) == 2
    assert len(active[active["level_l"] == "microbiome"]) == 5
    assert len(active[active["level_l"] == "ecological"]) == 3
    assert "development" in set(inv["level_l"])
    assert "life_history" in set(inv["level_l"])
    assert inv["mathematical_term"].str.contains("mu_source", regex=False).any()


def test_epigenetic_scaling_applies_to_linear_and_nonlinear_terms() -> None:
    b = scaled_epigenetic_B(0.35, b_micro=0.0, b_eco=0.0)
    assert np.isclose(b.b_epi_reg, 0.077)
    assert np.isclose(b.b_epi_stress, 0.070)
    assert np.isclose(b.epi_interaction_reg, 0.014)
    assert np.isclose(b.epi_interaction_stress, 0.0105)


def test_second_calibration_grids_use_requested_values() -> None:
    stage1 = inter_elc_epigenetic_isolation_grid()
    assert [round(scale, 2) for _, scale, _ in stage1] == [0.20, 0.35, 0.50, 0.65]
    selected_scale = 0.35
    stage2 = inter_elc_small_micro_eco_grid(selected_scale)
    assert [(b.b_micro, b.b_eco) for _, _, b in stage2] == [
        (0.01, 0.01),
        (0.025, 0.01),
        (0.01, 0.025),
        (0.025, 0.025),
        (0.05, 0.01),
        (0.01, 0.05),
    ]
    assert all(np.isclose(b.epi_interaction_reg, 0.04 * selected_scale) for _, _, b in stage2)


def test_source_state_means_have_expected_dimensions() -> None:
    cfg = tiny_inter_elc_config()
    data = simulate_inter_elc_seed(9091, cfg, InterELCBMatrices(b_micro=0.0, b_eco=0.0), InterELCSourceMeans.zero())
    means = estimate_source_state_means([data], cfg)
    assert len(means.epigenetic) == 2
    assert len(means.log_microbiome) == 5
    assert len(means.ecological) == 3


def test_frozen_final_inter_elc_B_is_epigenetic_only_35_percent() -> None:
    b = frozen_epigenetic_only_inter_elc_B()
    assert np.isclose(b.b_epi_reg, 0.077)
    assert np.isclose(b.b_epi_stress, 0.070)
    assert np.isclose(b.epi_interaction_reg, 0.014)
    assert np.isclose(b.epi_interaction_stress, 0.0105)
    assert b.b_micro == 0.0
    assert b.b_eco == 0.0
    matrices = b.matrices()
    assert np.allclose(matrices["development"], 0.0)
    assert np.allclose(matrices["life_history"], 0.0)
    assert np.allclose(matrices["microbiome"], 0.0)
    assert np.allclose(matrices["ecological"], 0.0)


def test_final_inter_elc_seeds_are_distinct_from_calibration_seeds() -> None:
    assert tuple(INTER_ELC_FINAL_SEEDS) == (6311, 7541, 8779, 9901)
    assert final_inter_elc_seeds_are_untouched()


def test_corrected_temporal_inter_elc_final_setting_is_epigenetic_only_and_untouched() -> None:
    seeds = tuple(CORRECTED_INTER_ELC_FINAL_SEEDS_V2)
    assert seeds == (16319, 17431, 18637, 19843)
    assert len(set(seeds)) == 4
    assert set(seeds).isdisjoint({101, 202, 303, 404, 6311, 7541, 8779, 9901})
    b = scaled_epigenetic_B(0.65, b_micro=0.0, b_eco=0.0)
    assert np.isclose(b.b_epi_reg, 0.143)
    assert np.isclose(b.b_epi_stress, 0.130)
    assert np.isclose(b.epi_interaction_reg, 0.026)
    assert np.isclose(b.epi_interaction_stress, 0.0195)
    assert np.allclose(b.matrices()["microbiome"], 0.0)
    assert np.allclose(b.matrices()["ecological"], 0.0)


def test_approved_source_means_are_frozen_constants() -> None:
    means = approved_inter_elc_source_means()
    assert np.isclose(means.epigenetic[0], 0.3471842314766994)
    assert np.isclose(means.epigenetic[1], -0.05490154474202076)


def test_inter_elc_table_uses_dprime_source_and_d_history() -> None:
    cfg = tiny_inter_elc_config()
    data = simulate_inter_elc_seed(9091, cfg, InterELCBMatrices(b_micro=0.10, b_eco=0.10))
    table = build_inter_elc_table([data], cfg)
    meta0 = table["meta"].iloc[0]
    d = int(meta0["lineage_id"])
    tau = int(meta0["tau"])
    d_prime = int(data["inter_elc_pairs"][d])
    expected_source = _segment(data, "epigenetic", tau)[d_prime]
    expected_history = _history(data, tau, ELC_LEVELS, cfg.history_order)[d]
    assert np.allclose(table["source_dprime_epigenetic"][0], expected_source)
    assert np.allclose(table["history_full_elc"][0], expected_history)


def test_complete_source_information_nesting_holds_for_inter_elc_family() -> None:
    cfg = tiny_inter_elc_config()
    data = simulate_inter_elc_seed(9091, cfg, InterELCBMatrices(b_micro=0.10, b_eco=0.10))
    table = build_inter_elc_table([data], cfg)
    family = inter_elc_family()
    joint, slices, _ = _ranked_joint_arrays(table, family, include_generation=True)
    values = _family_estimates_from_cov(_cov(joint, 1e-8), slices, family)
    whole = values["whole_source_elc_from_d_prime"]
    assert whole + 1e-9 >= values["dprime_epigenetic_subsystem"]
    assert whole + 1e-9 >= values["dprime_microbiome_subsystem"]
    assert whole + 1e-9 >= values["dprime_ecological_subsystem"]
