from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / "outputs" / "c13_final_untouched"
MC_DIR = PRIMARY / "intervention_fisher_mc_robustness"
SPLIT_DIR = PRIMARY / "intervention_fisher_split_half_cross_product"
OUTPUT_DIR = PRIMARY / "intervention_fisher_robustness"
COORDINATES = {
    "epigenetic": ("theta_reg", "theta_stress"),
    "joint": (
        "theta_reg",
        "theta_stress",
        "theta_soil",
        "theta_resource",
        "theta_microclimate",
    ),
}


def _matrix_outputs(analysis: str) -> None:
    plugin_long = pd.read_csv(MC_DIR / "fisher_matrix_by_M.csv")
    plugin_long = plugin_long[
        (plugin_long["analysis"] == analysis) & (plugin_long["M"] == 1000)
    ]
    split_long = pd.read_csv(SPLIT_DIR / "split_half_cross_product_fisher_matrix_comparison.csv")
    split_long = split_long[split_long["analysis"] == analysis]
    coords = list(COORDINATES[analysis])
    plugin = (
        plugin_long.pivot(
            index="row_component", columns="col_component", values="fisher_information"
        )
        .loc[coords, coords]
        .to_numpy(dtype=float)
    )
    split = (
        split_long.pivot(
            index="row_component",
            columns="col_component",
            values="split_half_cross_product_fisher",
        )
        .loc[coords, coords]
        .to_numpy(dtype=float)
    )
    matrix_rows: list[dict[str, object]] = []
    eigen_rows: list[dict[str, object]] = []
    for estimator, matrix in (
        ("ordinary_M1000_plugin", plugin),
        ("split_half_cross_product", split),
    ):
        eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
        for index, eigenvalue in enumerate(eigenvalues):
            eigen_rows.append(
                {
                    "analysis": analysis,
                    "estimator": estimator,
                    "eigenvalue_index": index,
                    "eigenvalue": float(eigenvalue),
                    "negative": bool(eigenvalue < 0.0),
                }
            )
        for i, row_coordinate in enumerate(coords):
            for j, column_coordinate in enumerate(coords):
                plugin_value = float(plugin[i, j])
                value = float(matrix[i, j])
                difference = value - plugin_value
                matrix_rows.append(
                    {
                        "analysis": analysis,
                        "estimator": estimator,
                        "row_coordinate": row_coordinate,
                        "column_coordinate": column_coordinate,
                        "value": value,
                        "absolute_difference_from_plugin": abs(difference),
                        "relative_difference_from_plugin": (
                            difference / plugin_value if plugin_value != 0.0 else np.nan
                        ),
                        "minimum_eigenvalue": float(eigenvalues.min()),
                        "maximum_eigenvalue": float(eigenvalues.max()),
                        "numerical_rank_tolerance_1e_minus_10": int(
                            np.sum(eigenvalues > 1e-10)
                        ),
                    }
                )
    pd.DataFrame(matrix_rows).to_csv(
        OUTPUT_DIR / f"{analysis}_fisher_plugin_and_split_half_matrix.csv",
        index=False,
    )
    pd.DataFrame(eigen_rows).to_csv(
        OUTPUT_DIR / f"{analysis}_fisher_plugin_and_split_half_eigenvalues.csv",
        index=False,
    )


def _gradient_output(analysis: str) -> None:
    gradients = pd.read_csv(MC_DIR / "gradient_summary.csv")
    gradients = gradients[
        (gradients["analysis"] == analysis) & (gradients["M"] == 1000)
    ].copy()
    gradients = gradients.rename(
        columns={
            "sd": "standard_deviation",
            "min": "minimum",
            "max": "maximum",
            "pct_positive_point": "percentage_positive_point",
            "pct_negative_point": "percentage_negative_point",
            "pct_positive_interval": "percentage_positive_interval",
            "pct_negative_interval": "percentage_negative_interval",
        }
    )
    gradients["n_contexts"] = np.where(
        gradients["seed"].astype(str) == "all", 1000, 250
    )
    columns = [
        "analysis",
        "seed",
        "coordinate",
        "n_contexts",
        "mean",
        "median",
        "standard_deviation",
        "minimum",
        "q025",
        "q25",
        "q75",
        "q975",
        "maximum",
        "percentage_positive_point",
        "percentage_negative_point",
        "percentage_positive_interval",
        "percentage_negative_interval",
    ]
    gradients[columns].to_csv(
        OUTPUT_DIR / f"{analysis}_local_gradient_summary.csv", index=False
    )


def _context_outputs(analysis: str) -> None:
    context = pd.read_csv(MC_DIR / "context_specific_qualification_M1000_by_context.csv")
    context = context[context["analysis"] == analysis].copy()
    context["focal_nu_log_ratio_positive"] = context["A_log_ratio_positive"]
    context["same_context_point_supported_any_coordinate"] = context[
        "ABC_permitted_direction_point_any_coordinate"
    ]
    context["same_context_interval_supported_any_coordinate"] = context[
        "ABC_permitted_direction_interval_any_coordinate"
    ]
    context.to_csv(
        OUTPUT_DIR / f"{analysis}_same_context_robustness_by_context.csv", index=False
    )
    rows: list[dict[str, object]] = []
    groups = [("all", context)] + [
        (str(seed), group) for seed, group in context.groupby("seed", sort=True)
    ]
    for seed, group in groups:
        positive = group["focal_nu_log_ratio_positive"].astype(bool)
        point = group["same_context_point_supported_any_coordinate"].astype(bool)
        interval = group["same_context_interval_supported_any_coordinate"].astype(bool)
        rows.append(
            {
                "analysis": analysis,
                "seed": seed,
                "n_contexts": len(group),
                "n_positive_focal_nu_log_ratio": int(positive.sum()),
                "percentage_positive_focal_nu_log_ratio": float(100.0 * positive.mean()),
                "n_same_context_point_supported": int(point.sum()),
                "percentage_same_context_point_supported": float(100.0 * point.mean()),
                "n_same_context_interval_supported": int(interval.sum()),
                "percentage_same_context_interval_supported": float(
                    100.0 * interval.mean()
                ),
            }
        )
    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / f"{analysis}_same_context_robustness_summary.csv", index=False
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for analysis in COORDINATES:
        _matrix_outputs(analysis)
        _gradient_output(analysis)
        _context_outputs(analysis)


if __name__ == "__main__":
    main()
