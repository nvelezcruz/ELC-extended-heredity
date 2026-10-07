from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def test_c13_final_graph_recovers_every_encoded_dependency() -> None:
    summary = pd.read_csv(
        FIXTURES / "graph_reconstruction" / "graph_reconstruction_recovery_summary.csv"
    ).iloc[0]
    assert int(summary["true_positives"]) == 28
    assert int(summary["false_negatives"]) == 0
    assert np.isclose(float(summary["recall"]), 1.0)


def test_factor_and_complex_location_profiles_cover_all_elc_times() -> None:
    profile = pd.read_csv(
        FIXTURES
        / "location_profiles"
        / "factor_and_complex_location_summary.csv"
    )
    expected_lengths = {
        "development": 57,
        "microbiome": 61,
        "life_history": 41,
        "epigenetic": 33,
        "ecological": 33,
    }
    assert set(profile["source_profile"]) == {
        "epigenetic_factor",
        "joint_epigenetic_ecological",
    }
    for source in sorted(set(profile["source_profile"])):
        source_rows = profile[profile["source_profile"] == source]
        for level, expected in expected_lengths.items():
            level_rows = source_rows[source_rows["target_level"] == level]
            assert len(level_rows) == expected
            assert np.array_equal(
                level_rows.sort_values("target_t_l")["target_t_l"].to_numpy(dtype=int),
                np.arange(expected),
            )
    numeric = profile[
        [
            "raw_estimate_bits",
            "randomized_source_mean_bits",
            "information_beyond_randomized_mean_bits",
            "ci_lower_bits",
            "ci_upper_bits",
            "randomized_source_p_value",
        ]
    ].to_numpy(dtype=float)
    assert np.all(np.isfinite(numeric))


def test_c13_robust_fisher_diagonals_are_positive() -> None:
    robust = FIXTURES / "fisher"
    for source, coordinates in {
        "epigenetic": ("theta_reg", "theta_stress"),
        "joint": (
            "theta_reg",
            "theta_stress",
            "theta_soil",
            "theta_resource",
            "theta_microclimate",
        ),
    }.items():
        matrix = pd.read_csv(
            robust / f"{source}_fisher_plugin_and_split_half_matrix.csv"
        )
        for estimator in ("ordinary_M1000_plugin", "split_half_cross_product"):
            rows = matrix[matrix["estimator"] == estimator]
            diagonal = []
            for coordinate in coordinates:
                diagonal.append(
                    float(
                        rows[
                            (rows["row_coordinate"] == coordinate)
                            & (rows["column_coordinate"] == coordinate)
                        ]["value"].iloc[0]
                    )
                )
            assert np.all(np.asarray(diagonal) > 0.0)
