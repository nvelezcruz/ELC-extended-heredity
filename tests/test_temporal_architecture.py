from __future__ import annotations

from dataclasses import replace
import inspect

import numpy as np

from src.corrected_pilot_model import (
    PilotConfig,
    _params,
    apply_parameter_overrides,
    build_event_schedule,
    effective_nonzero_cross_level_edges,
    extract_analysis_segment,
    level_T,
    level_m,
    level_segment_indices,
    level_steps,
    level_timestamps,
    reproductive_indices,
    phenotype_states,
)
from src.final_numerical_audit import (
    BACKGROUND_LEVEL,
    ELC_LEVELS,
    FinalAuditConfig,
    _integrate_generation_vec,
    build_analysis_table,
    load_saved_seed,
    save_seed_archive,
    simulate_final_seed,
)


ALL_LEVELS = ELC_LEVELS + (BACKGROUND_LEVEL,)


def tiny_config() -> FinalAuditConfig:
    return replace(
        FinalAuditConfig(),
        seeds=(17,),
        n_lineages=6,
        n_generations=4,
        burn_in_generations=1,
        source_tau_start=1,
        source_tau_stop=2,
        history_order=2,
        n_bootstrap=2,
        n_pid_bootstrap=2,
        n_null=2,
        n_contexts=6,
        n_contexts_per_seed=6,
        n_intervention_draws=2,
    )


def test_full_retained_lengths_and_asynchronous_timestamps() -> None:
    config = PilotConfig()
    assert level_T(config) == {
        "development": 57,
        "microbiome": 61,
        "life_history": 41,
        "epigenetic": 33,
        "ecological": 33,
        "background": 33,
    }
    timestamp_lengths = {level: len(values) for level, values in level_timestamps(config).items()}
    assert timestamp_lengths == level_T(config)
    assert len(set(timestamp_lengths.values())) > 1
    for values in level_timestamps(config).values():
        assert np.isclose(values[0], 0.0)
        assert np.isclose(values[-1], 1.0)
        assert np.all(np.diff(values) > 0.0)


def test_segments_are_proper_subsets_extracted_from_full_series() -> None:
    config = PilotConfig()
    expected = {
        "development": np.arange(6, 14),
        "microbiome": np.arange(36, 46),
        "life_history": np.arange(20, 25),
        "epigenetic": np.arange(20, 25),
        "ecological": np.arange(20, 25),
        "background": np.arange(20, 25),
    }
    for level, indices in level_segment_indices(config).items():
        assert np.array_equal(indices, expected[level])
        assert len(indices) == level_m(config)[level]
        assert len(indices) < level_T(config)[level]
        assert np.all(indices >= 0)
        assert np.all(indices < level_T(config)[level])
        full = np.arange(level_T(config)[level] * 2, dtype=float).reshape(level_T(config)[level], 2)
        extracted = extract_analysis_segment(full, level, config)
        assert np.array_equal(extracted, full[indices])
        assert np.all(level_timestamps(config)[level][indices] <= config.reproductive_u + 1e-12)


def test_reproductive_state_is_separate_from_segment_and_terminal_state() -> None:
    config = PilotConfig()
    expected = {
        "development": 42,
        "microbiome": 45,
        "life_history": 30,
        "epigenetic": 24,
        "ecological": 24,
        "background": 24,
    }
    assert reproductive_indices(config) == expected
    for level, index in expected.items():
        timestamps = level_timestamps(config)[level]
        assert np.isclose(timestamps[index], 0.75)
        assert index != len(timestamps) - 1


def test_solver_grid_retained_grid_and_segment_lengths_are_distinct() -> None:
    config = PilotConfig()
    for level in ALL_LEVELS:
        assert level_steps(config)[level] != level_T(config)[level]
        assert level_T(config)[level] != level_m(config)[level]
        assert level_steps(config)[level] != level_m(config)[level]


def test_vectorized_simulator_saves_full_segment_and_reproductive_arrays() -> None:
    config = tiny_config()
    data = simulate_final_seed(17, config)
    for level in ALL_LEVELS:
        full = np.asarray(data["full_time_series"][level])
        segment = np.asarray(data[level])
        reproductive = np.asarray(data["reproductive_states"][level])
        assert full.shape[2] == level_T(data["config"])[level]
        assert segment.shape[2] == level_m(data["config"])[level]
        assert np.array_equal(segment, full[:, :, data["segment_indices"][level], :])
        assert np.array_equal(reproductive, full[:, :, data["reproductive_indices"][level], :])


def test_asynchronous_updates_use_one_pre_event_state_without_interpolation() -> None:
    source = inspect.getsource(_integrate_generation_vec)
    assert "pre = {k: v.copy() for k, v in state.items()}" in source
    assert "UPDATE_VEC[level](pre" in source
    assert source.index("state.update(updates)") > source.index("UPDATE_VEC[level](pre")
    schedule = build_event_schedule(PilotConfig())
    assert any(len(levels) > 1 for _, levels in schedule)


def test_corrected_archive_round_trip_and_legacy_archive_rejection(tmp_path) -> None:
    config = tiny_config()
    data = simulate_final_seed(17, config)
    corrected = tmp_path / "corrected_temporal_architecture_seed_17.npz"
    save_seed_archive(corrected, data)
    loaded = load_saved_seed(17, config, tmp_path)
    assert loaded is not None
    for level in ALL_LEVELS:
        assert np.array_equal(loaded[level], data[level])
        assert np.array_equal(loaded["full_time_series"][level], data["full_time_series"][level])

    corrected.unlink()
    np.savez_compressed(
        tmp_path / "corrected_temporal_architecture_seed_17.npz",
        **{level: np.asarray(data[level]) for level in ALL_LEVELS},
    )
    assert load_saved_seed(17, config, tmp_path) is None


def test_analysis_dimensions_are_computed_from_selected_segments() -> None:
    config = tiny_config()
    data = simulate_final_seed(17, config)
    table = build_analysis_table([data], config)
    expected_epi = level_m(data["config"])["epigenetic"] * data["config"].epi_dim
    expected_full = sum(
        level_m(data["config"])[level]
        * {
            "development": data["config"].dev_dim,
            "microbiome": data["config"].micro_dim,
            "life_history": data["config"].life_dim,
            "epigenetic": data["config"].epi_dim,
            "ecological": data["config"].eco_dim,
        }[level]
        for level in ELC_LEVELS
    )
    assert table["source_epigenetic"].shape[1] == expected_epi
    assert table["target_full_elc"].shape[1] == expected_full


def test_early_maturation_occurs_before_reproductive_transition() -> None:
    config = replace(PilotConfig(), n_lineages=2, n_generations=1, burn_in_generations=0)
    life = np.zeros((2, 2, config.T_life, config.life_dim), dtype=float)
    life[..., 1] = 0.8
    life[0, 1, 24:, 0] = config.theta_maturation
    life[1, 1, 23:, 0] = config.theta_maturation
    data = {
        "full_time_series": {"life_history": life},
        "timestamps": {"life_history": np.arange(config.T_life, dtype=float) / 40.0},
    }
    states = phenotype_states(data, config).sort_values("d")
    assert states.iloc[0]["phenotype_state"] == "non_early_maturation_high_growth"
    assert states.iloc[1]["phenotype_state"] == "nu_early_maturation_high_growth"
    assert np.isclose(config.early_maturation_u, 0.60)
    assert config.early_maturation_u < config.reproductive_u


def test_effective_edge_inventory_uses_calibrated_coefficients_and_microbiome_notation() -> None:
    par = apply_parameter_overrides(
        _params(),
        (
            ("epi_effect_life_timing_scale", 4.5),
            ("epi_effect_life_maturation_scale", 3.0),
            ("epi_effect_life_growth_scale", 7.0),
            ("epi_effect_life_allocation_scale", 3.5),
            ("development_effect_microbiome_scale", 1.25),
        ),
    )
    edges = effective_nonzero_cross_level_edges(par)
    epi_life = edges[
        (edges["process"] == "within_generation")
        & (edges["source_level"] == "epigenetic")
        & (edges["target_level"] == "life_history")
    ].set_index("target_component")
    assert np.isclose(epi_life.loc["growth_capacity", "coefficient"], 1.68)
    assert np.isclose(epi_life.loc["reproductive_allocation", "coefficient"], 0.77)
    assert "0.75 p_reg^circ" in epi_life.loc["maturation_progress", "mathematical_term"]
    assert "-0.315 p_reg^circ" in epi_life.loc["maturation_progress", "mathematical_term"]
    dev_micro = edges[
        (edges["process"] == "within_generation")
        & (edges["source_level"] == "development")
        & (edges["target_level"] == "microbiome")
    ]
    assert np.allclose(
        np.sort(dev_micro["coefficient"].to_numpy(dtype=float)),
        np.sort(np.array([0.225, 0.10, 0.20])),
    )
    assert not edges["mathematical_term"].str.contains(r"\by_[1-5]\b", regex=True).any()
