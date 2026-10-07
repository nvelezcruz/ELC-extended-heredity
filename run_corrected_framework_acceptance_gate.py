from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _append(rows: list[dict[str, object]], group: str, criterion: str, observed: object, target: str, passed: bool) -> None:
    rows.append(
        {
            "group": group,
            "criterion": criterion,
            "observed": observed,
            "target": target,
            "passes": bool(passed),
        }
    )


def main() -> None:
    root = Path(__file__).resolve().parent
    primary_root = root / "outputs" / "c13_final_untouched"
    primary = primary_root / "corrected_numerical_audit" / "data"
    completeness = primary_root / "draft32_completeness_audit" / "data"
    robustness = primary_root / "intervention_fisher_robustness"
    inter_elc = root / "outputs" / "c13_inter_elc_final_untouched" / "data"
    rows: list[dict[str, object]] = []

    graph = pd.read_csv(primary_root / "graph_reconstruction" / "graph_reconstruction_information_matrix.csv")
    expected = graph[graph["known_dependency"].astype(bool)]
    _append(
        rows,
        "coupling recovery",
        "all encoded level-level dependencies recovered",
        f"{int(expected['recovered_dependency'].sum())}/{len(expected)}",
        f"{len(expected)}/{len(expected)} with zero missed",
        bool(expected["recovered_dependency"].all()),
    )

    phenotype_checks = pd.read_csv(primary_root / "final_phenotype_acceptance_checks.csv")
    for _, row in phenotype_checks.iterrows():
        _append(rows, "phenotype", str(row["criterion"]), row["observed_value"], str(row["target"]), bool(row["passes"]))

    central = pd.read_csv(primary / "corrected_central_contrast_check.csv")
    for _, row in central.iterrows():
        _append(rows, "aggregate hereditary factor", str(row["criterion"]), row.get("observed_value", row.get("value", "see source")), str(row.get("target", "predeclared contrast")), bool(row["passes"]))

    stability = pd.read_csv(primary / "corrected_intergenerational_stability_summary.csv")
    for _, row in stability.iterrows():
        passed = float(row["null_excess_bits"]) > 0.0 and float(row["generation_preserving_surrogate_p_value"]) <= 0.05
        _append(rows, "stability", f"horizon rho={int(row['rho'])}", float(row["null_excess_bits"]), ">0 bits beyond baseline with p<=0.05", passed)

    location = pd.read_csv(primary / "corrected_level_time_specific_transfer_entropy_summary.csv")
    for target_level in ("development", "microbiome", "life_history"):
        target = location[location["target_level"] == target_level]
        significant = (target["null_excess_bits"] > 0.0) & (
            target["generation_preserving_surrogate_p_value"] <= 0.05
        )
        _append(
            rows,
            "location",
            f"{target_level} contribution above baseline at every recorded time",
            f"{int(significant.sum())}/{len(target)} recorded times",
            f"{len(target)}/{len(target)}",
            bool(significant.all()),
        )
    ecology = location[location["target_level"] == "ecological"]
    ecology_significant = (ecology["null_excess_bits"] > 0.0) & (
        ecology["generation_preserving_surrogate_p_value"] <= 0.05
    )
    _append(
        rows,
        "location",
        "ecological level-wise profile contains information above baseline",
        f"{int(ecology_significant.sum())}/{len(ecology)} recorded times",
        "characterization only; no minimum ecological magnitude",
        bool(ecology_significant.any()),
    )

    intervention = pd.read_csv(primary / "intervention_generation_stratified_response_surface_summary.csv")
    p_col = "p_nu_early_maturation_high_growth"
    zero = intervention[np.isclose(intervention["theta_reg"], 0.0) & np.isclose(intervention["theta_stress"], 0.0)].iloc[0]
    positive_direction = intervention[np.isclose(intervention["theta_reg"], 1.5) & np.isclose(intervention["theta_stress"], -1.5)].iloc[0]
    delta = float(positive_direction[p_col] - zero[p_col])
    _append(rows, "epigenetic intervention", "predeclared positive direction increases p(nu)", delta, ">0", delta > 0.0)

    for analysis in ("epigenetic", "joint"):
        context_summary = pd.read_csv(robustness / f"{analysis}_same_context_robustness_summary.csv")
        context_all = context_summary[context_summary["seed"].astype(str) == "all"].iloc[0]
        _append(
            rows,
            "context-specific focal variant",
            f"{analysis} positive focal-nu information, interval-supported intervention coordinate, and positive Fisher diagonal coincide",
            int(context_all["n_same_context_interval_supported"]),
            ">0 matched contexts",
            int(context_all["n_same_context_interval_supported"]) > 0,
        )

    pid_summary = pd.read_csv(primary / "corrected_epigenetic_ecological_delta_g_pid_summary.csv").iloc[0]
    pid_seed = pd.read_csv(primary / "corrected_epigenetic_ecological_delta_g_pid_by_seed.csv")
    pid_ci = pd.read_csv(primary / "corrected_epigenetic_ecological_delta_g_pid_centered_intervals.csv")
    synergy_ci = pid_ci[pid_ci["quantity"] == "synergy_bits"].iloc[0]
    _append(rows, "complex source", "pooled PID synergy", float(pid_summary["synergy_bits"]), ">0 bits", float(pid_summary["synergy_bits"]) > 0.0)
    _append(rows, "complex source", "PID synergy centered interval", float(synergy_ci["ci_lower"]), "lower bound >0", float(synergy_ci["ci_lower"]) > 0.0)
    _append(rows, "complex source", "seed-specific PID synergy", float(pid_seed["synergy_bits"].min()), ">0 bits in every seed", bool((pid_seed["synergy_bits"] > 0.0).all()))

    multiple_parent_nesting = pd.read_csv(primary / "corrected_multiple_parent_nesting_checks.csv")
    for _, row in multiple_parent_nesting.iterrows():
        _append(
            rows,
            "multiple-parent extension",
            str(row["check"]),
            float(row["joint_bits"]),
            f">= {float(row['single_bits']):.12g} bits",
            bool(row["passes"]),
        )

    joint_qual = pd.read_csv(completeness / "complex_unit_qualification_table.csv")
    for _, row in joint_qual.iterrows():
        if str(row["requirement"]) == "Joint causal specificity":
            continue
        passed = str(row["pass_fail"]).strip().lower() == "pass"
        _append(rows, "joint source", str(row["requirement"]), str(row["result"]), "pass", passed)

    for analysis, expected_coordinates in (("epigenetic", 2), ("joint", 5)):
        matrix = pd.read_csv(robustness / f"{analysis}_fisher_plugin_and_split_half_matrix.csv")
        eigen = pd.read_csv(robustness / f"{analysis}_fisher_plugin_and_split_half_eigenvalues.csv")
        plugin = matrix[matrix["estimator"] == "ordinary_M1000_plugin"]
        plugin_diag = plugin[plugin["row_coordinate"] == plugin["column_coordinate"]]
        plugin_eigen = eigen[eigen["estimator"] == "ordinary_M1000_plugin"]["eigenvalue"].to_numpy(dtype=float)
        split = matrix[matrix["estimator"] == "split_half_cross_product"]
        split_diag = split[split["row_coordinate"] == split["column_coordinate"]]
        _append(
            rows,
            f"{analysis} causal specificity",
            "positive M=1000 Fisher diagonals",
            float(plugin_diag["value"].min()),
            f">0 for all {expected_coordinates} coordinates",
            len(plugin_diag) == expected_coordinates and bool((plugin_diag["value"] > 0.0).all()),
        )
        _append(
            rows,
            f"{analysis} causal specificity",
            "positive-semidefinite M=1000 averaged Fisher matrix",
            float(plugin_eigen.min()),
            "minimum eigenvalue >= -1e-10",
            bool(plugin_eigen.min() >= -1e-10),
        )
        _append(
            rows,
            f"{analysis} causal specificity",
            "positive split-half cross-product diagonals",
            float(split_diag["value"].min()),
            f">0 for all {expected_coordinates} coordinates",
            len(split_diag) == expected_coordinates and bool((split_diag["value"] > 0.0).all()),
        )

    qualitative = pd.read_csv(primary / "qualitative_acceptance_criteria_by_seed_frozen_model.csv")
    _append(
        rows,
        "numerical health",
        "all seed-level qualitative and boundedness checks",
        f"{int(qualitative['passes'].sum())}/{len(qualitative)}",
        f"{len(qualitative)}/{len(qualitative)}",
        bool(qualitative["passes"].all()),
    )
    convergence = pd.read_csv(primary / "integration_step_convergence_confirmation.csv")
    _append(
        rows,
        "numerical health",
        "all integration-convergence checks",
        f"{int(convergence['passes'].sum())}/{len(convergence)}",
        f"{len(convergence)}/{len(convergence)}",
        bool(convergence["passes"].all()),
    )

    inter_checks = pd.read_csv(inter_elc / "inter_elc_final_acceptance_checks.csv")
    for _, row in inter_checks.iterrows():
        _append(rows, "additional source lineage", str(row["criterion"]), row["observed_value"], str(row["target"]), bool(row["passes"]))

    result = pd.DataFrame(rows)
    output = root / "outputs" / "c13_framework_acceptance_gate.csv"
    result.to_csv(output, index=False)
    failed = result[~result["passes"]]
    print(result.to_string(index=False), flush=True)
    if len(failed):
        raise AssertionError(f"framework acceptance gate failed: {failed.to_dict(orient='records')}")
    print(f"all {len(result)} framework acceptance checks passed", flush=True)


if __name__ == "__main__":
    main()
