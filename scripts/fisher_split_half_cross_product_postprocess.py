from __future__ import annotations

from pathlib import Path
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.corrected_numerical_audit import _smooth_probabilities  # noqa: E402
from src.final_draft32_completeness_audit import JOINT_THETA_COMPONENTS  # noqa: E402
from src.final_numerical_audit import PHENOTYPE_LABELS  # noqa: E402


RUN_OUTPUT_NAME = os.environ.get("ROBUSTNESS_RUN_OUTPUT", "final_c8_untouched_audit")
RUN_ROOT = ROOT / "outputs" / RUN_OUTPUT_NAME
ROBUSTNESS_DIR = RUN_ROOT / "intervention_fisher_mc_robustness"
OUTPUT_DIR = RUN_ROOT / "intervention_fisher_split_half_cross_product"
M_HALF = 500
M_FULL = 1000
H = 0.1
ALPHA = 0.5
RANK_TOL = 1e-10
COUNT_COLS = [f"count_{label}" for label in PHENOTYPE_LABELS]
ANALYSIS_COORDS = {
    "epigenetic": ("theta_reg", "theta_stress"),
    "joint": JOINT_THETA_COMPONENTS,
}


def _counts(table: pd.DataFrame, analysis: str, m: int, theta_label: str) -> np.ndarray:
    sub = table[(table["analysis"] == analysis) & (table["M"] == m) & (table["theta_label"] == theta_label)]
    sub = sub.sort_values("context_id")
    if len(sub) != 1000:
        raise AssertionError(f"{analysis} M={m} {theta_label} has {len(sub)} rows, expected 1000")
    return sub[COUNT_COLS].to_numpy(dtype=float)


def _meta(table: pd.DataFrame, analysis: str) -> pd.DataFrame:
    sub = table[(table["analysis"] == analysis) & (table["M"] == M_FULL) & (table["theta_label"] == "zero")]
    return sub[["seed", "lineage_id", "tau", "context_id"]].sort_values("context_id").reset_index(drop=True)


def _safe_relative(diff: float, reference: float) -> float:
    if abs(reference) <= 1e-15:
        return np.nan
    return diff / reference


def split_half_cross_product(
    table: pd.DataFrame,
    analysis: str,
    coords: tuple[str, ...],
) -> dict[str, object]:
    p0 = _smooth_probabilities(_counts(table, analysis, M_FULL, "zero") / float(M_FULL), M_FULL, ALPHA)
    gradients_a = []
    gradients_b = []
    half_checks = []
    for coord in coords:
        plus_a = _counts(table, analysis, M_HALF, f"plus_{coord}")
        minus_a = _counts(table, analysis, M_HALF, f"minus_{coord}")
        plus_b = _counts(table, analysis, M_FULL, f"plus_{coord}") - plus_a
        minus_b = _counts(table, analysis, M_FULL, f"minus_{coord}") - minus_a
        for label, arr in [(f"plus_{coord}_half_B", plus_b), (f"minus_{coord}_half_B", minus_b)]:
            if np.any(arr < 0):
                raise AssertionError(f"negative split-half count in {analysis} {label}")
            sums = arr.sum(axis=1)
            if not np.allclose(sums, M_HALF):
                raise AssertionError(f"{analysis} {label} row sums are not {M_HALF}")
        p_plus_a = _smooth_probabilities(plus_a / float(M_HALF), M_HALF, ALPHA)
        p_minus_a = _smooth_probabilities(minus_a / float(M_HALF), M_HALF, ALPHA)
        p_plus_b = _smooth_probabilities(plus_b / float(M_HALF), M_HALF, ALPHA)
        p_minus_b = _smooth_probabilities(minus_b / float(M_HALF), M_HALF, ALPHA)
        gradients_a.append((p_plus_a - p_minus_a) / (2.0 * H))
        gradients_b.append((p_plus_b - p_minus_b) / (2.0 * H))
        half_checks.append(
            {
                "analysis": analysis,
                "coordinate": coord,
                "half_A_draws": M_HALF,
                "half_B_draws": M_HALF,
                "half_B_min_count": int(min(plus_b.min(), minus_b.min())),
                "half_B_max_row_sum": int(max(plus_b.sum(axis=1).max(), minus_b.sum(axis=1).max())),
                "half_B_min_row_sum": int(min(plus_b.sum(axis=1).min(), minus_b.sum(axis=1).min())),
            }
        )
    g_a = np.stack(gradients_a, axis=1)
    g_b = np.stack(gradients_b, axis=1)
    mats_ab = np.einsum("cav,cbv,cv->cab", g_a, g_b, 1.0 / p0)
    mats_ba = np.einsum("cav,cbv,cv->cab", g_b, g_a, 1.0 / p0)
    context_mats = 0.5 * (mats_ab + mats_ba)
    context_mats = 0.5 * (context_mats + np.swapaxes(context_mats, 1, 2))
    average = 0.5 * (context_mats.mean(axis=0) + context_mats.mean(axis=0).T)
    eig = np.linalg.eigvalsh(average)
    context_eig = np.linalg.eigvalsh(context_mats)
    return {
        "p0": p0,
        "gradients_a": g_a,
        "gradients_b": g_b,
        "context_mats": context_mats,
        "average": average,
        "eigenvalues": eig,
        "context_eigenvalues": context_eig,
        "half_checks": pd.DataFrame(half_checks),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(ROBUSTNESS_DIR / "probability_by_context.csv")
    plugin = pd.read_csv(ROBUSTNESS_DIR / "fisher_matrix_by_M.csv")
    plugin_eigs = pd.read_csv(ROBUSTNESS_DIR / "fisher_eigenvalues_by_M.csv")
    gradient_summary = pd.read_csv(ROBUSTNESS_DIR / "gradient_summary.csv")
    qualification = pd.read_csv(ROBUSTNESS_DIR / "context_specific_qualification_M1000_summary.csv")

    matrix_rows = []
    eigen_rows = []
    diag_rows = []
    context_rows = []
    half_check_frames = []
    gradient_rows = []
    qualification_rows = []

    for analysis, coords in ANALYSIS_COORDS.items():
        result = split_half_cross_product(table, analysis, coords)
        cross = result["average"]
        eig = result["eigenvalues"]
        context_eig = result["context_eigenvalues"]
        half_check_frames.append(result["half_checks"])

        plugin_matrix = (
            plugin[(plugin["analysis"] == analysis) & (plugin["M"] == M_FULL)]
            .pivot(index="row_component", columns="col_component", values="fisher_information")
            .loc[list(coords), list(coords)]
            .to_numpy(dtype=float)
        )
        plugin_eval = (
            plugin_eigs[(plugin_eigs["analysis"] == analysis) & (plugin_eigs["M"] == M_FULL)]
            .sort_values("eigenvalue_index")["eigenvalue"]
            .to_numpy(dtype=float)
        )

        for i, row_component in enumerate(coords):
            for j, col_component in enumerate(coords):
                diff = cross[i, j] - plugin_matrix[i, j]
                abs_diff = abs(diff)
                matrix_rows.append(
                    {
                        "analysis": analysis,
                        "row_component": row_component,
                        "col_component": col_component,
                        "split_half_cross_product_fisher": float(cross[i, j]),
                        "ordinary_M1000_plugin_fisher": float(plugin_matrix[i, j]),
                        "signed_difference_cross_minus_plugin": float(diff),
                        "absolute_difference": float(abs_diff),
                        "relative_difference_signed_vs_plugin": float(_safe_relative(diff, plugin_matrix[i, j])),
                        "relative_difference_absolute_vs_plugin": float(_safe_relative(abs_diff, abs(plugin_matrix[i, j]))),
                    }
                )
        for i, coord in enumerate(coords):
            diff = cross[i, i] - plugin_matrix[i, i]
            diag_rows.append(
                {
                    "analysis": analysis,
                    "coordinate": coord,
                    "split_half_cross_product_diagonal": float(cross[i, i]),
                    "ordinary_M1000_plugin_diagonal": float(plugin_matrix[i, i]),
                    "signed_difference_cross_minus_plugin": float(diff),
                    "absolute_difference": float(abs(diff)),
                    "relative_difference_absolute_vs_plugin": float(_safe_relative(abs(diff), abs(plugin_matrix[i, i]))),
                    "cross_product_diagonal_positive": bool(cross[i, i] > 0.0),
                    "plugin_diagonal_positive": bool(plugin_matrix[i, i] > 0.0),
                }
            )
        for i, value in enumerate(eig):
            diff = value - plugin_eval[i]
            eigen_rows.append(
                {
                    "analysis": analysis,
                    "eigenvalue_index": i,
                    "split_half_cross_product_eigenvalue": float(value),
                    "ordinary_M1000_plugin_eigenvalue": float(plugin_eval[i]),
                    "signed_difference_cross_minus_plugin": float(diff),
                    "absolute_difference": float(abs(diff)),
                    "relative_difference_absolute_vs_plugin": float(_safe_relative(abs(diff), abs(plugin_eval[i]))),
                    "cross_product_eigenvalue_negative": bool(value < 0.0),
                }
            )
        min_eig = float(eig[0])
        max_eig = float(eig[-1])
        context_rows.append(
            {
                "analysis": analysis,
                "n_contexts": int(context_eig.shape[0]),
                "cross_product_any_negative_average_eigenvalue": bool(np.any(eig < 0.0)),
                "cross_product_min_average_eigenvalue": min_eig,
                "cross_product_max_average_eigenvalue": max_eig,
                "cross_product_condition_number": float(max_eig / min_eig) if min_eig > 0.0 else np.nan,
                "ordinary_plugin_condition_number": float(plugin_eval[-1] / plugin_eval[0]) if plugin_eval[0] > 0.0 else np.nan,
                "ordinary_plugin_rank_tol_1e_minus_10": int(np.sum(plugin_eval > RANK_TOL)),
                "cross_product_n_eigenvalues_gt_1e_minus_10": int(np.sum(eig > RANK_TOL)),
                "context_specific_pct_with_negative_min_eigenvalue": float(100.0 * np.mean(context_eig[:, 0] < 0.0)),
                "context_specific_min_eigenvalue_min": float(np.min(context_eig[:, 0])),
                "context_specific_min_eigenvalue_q025": float(np.quantile(context_eig[:, 0], 0.025)),
                "context_specific_min_eigenvalue_median": float(np.median(context_eig[:, 0])),
                "context_specific_min_eigenvalue_q975": float(np.quantile(context_eig[:, 0], 0.975)),
            }
        )

        gs = gradient_summary[
            (gradient_summary["analysis"] == analysis)
            & (gradient_summary["M"] == M_FULL)
            & (gradient_summary["seed"].astype(str) == "all")
        ]
        for _, row in gs.iterrows():
            gradient_rows.append(
                {
                    "analysis": analysis,
                    "coordinate": row["coordinate"],
                    "M": M_FULL,
                    "context_averaged_gradient_p_nu": float(row["mean"]),
                    "median_context_gradient_p_nu": float(row["median"]),
                    "pct_positive_point": float(row["pct_positive_point"]),
                    "pct_negative_point": float(row["pct_negative_point"]),
                    "pct_interval_excludes_zero": float(row["pct_interval_excludes_zero"]),
                }
            )

        q = qualification[
            (qualification["analysis"] == analysis)
            & (qualification["seed"].astype(str) == "all")
            & qualification["classification"].isin(
                [
                    "positive_focal_nu_log_ratio_A",
                    "all_three_same_context_any_coordinate_interval_supported",
                    "all_three_same_context_any_coordinate_point_estimate",
                ]
            )
        ].copy()
        qualification_rows.append(q)

    matrix_df = pd.DataFrame(matrix_rows)
    eigen_df = pd.DataFrame(eigen_rows)
    diag_df = pd.DataFrame(diag_rows)
    context_df = pd.DataFrame(context_rows)
    half_df = pd.concat(half_check_frames, ignore_index=True)
    gradients_df = pd.DataFrame(gradient_rows)
    qual_df = pd.concat(qualification_rows, ignore_index=True)

    matrix_df.to_csv(OUTPUT_DIR / "split_half_cross_product_fisher_matrix_comparison.csv", index=False)
    eigen_df.to_csv(OUTPUT_DIR / "split_half_cross_product_fisher_eigenvalue_comparison.csv", index=False)
    diag_df.to_csv(OUTPUT_DIR / "split_half_cross_product_fisher_diagonal_comparison.csv", index=False)
    context_df.to_csv(OUTPUT_DIR / "split_half_cross_product_fisher_diagnostics.csv", index=False)
    half_df.to_csv(OUTPUT_DIR / "split_half_cross_product_half_count_checks.csv", index=False)
    gradients_df.to_csv(OUTPUT_DIR / "recommended_local_gradients_M1000.csv", index=False)
    qual_df.to_csv(OUTPUT_DIR / "recommended_context_qualification_percentages.csv", index=False)

    metadata = {
        "source_counts": str(ROBUSTNESS_DIR / "probability_by_context.csv"),
        "ordinary_plugin_source": str(ROBUSTNESS_DIR / "fisher_matrix_by_M.csv"),
        "half_A": "first 500 draws, stored as M=500 counts",
        "half_B": "draws 501-1000, computed as M=1000 counts minus M=500 counts",
        "half_gradient_smoothing_alpha": ALPHA,
        "denominator": "M=1000 no-change phenotype probabilities after alpha=0.5 smoothing",
        "finite_difference_step": H,
        "rank_tolerance": RANK_TOL,
    }
    (OUTPUT_DIR / "split_half_cross_product_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"wrote split-half Fisher post-processing outputs to {OUTPUT_DIR}")
    print(context_df.to_string(index=False))
    print(eigen_df.to_string(index=False))


if __name__ == "__main__":
    main()
