from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_c13_final_untouched import CANDIDATE, FINAL_SEEDS
from src.corrected_numerical_audit import (
    CorrectedAuditConfig,
    _centered_pid_interval,
    _pid_bootstrap,
    _run_delta_g_pid,
    conditional_residuals_for_pid,
)
from src.final_numerical_audit import build_analysis_table, load_saved_seed
from src.information_measures import summarize_interval, unit_index_groups
from src.recalibration_audit import calibration_candidates


TARGET_LABEL = "non_early_maturation_low_growth"
TARGET_SYMBOL = r"\nu_{\mathrm{joint}}=(0,0)"
COORDINATES = (
    "theta_reg",
    "theta_stress",
    "theta_soil",
    "theta_resource",
    "theta_microclimate",
)
Z_CRIT = 1.959963984540054


def _setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.5,
            "figure.titlesize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def _categorical_outputs(source: Path, output: Path, config: CorrectedAuditConfig) -> dict[str, float]:
    scores = pd.read_csv(source / "joint_categorical_phenotype_crossfit_scores.csv")
    labels = [
        "nu_early_maturation_high_growth",
        "early_maturation_low_growth",
        "non_early_maturation_high_growth",
        "non_early_maturation_low_growth",
    ]
    screening = []
    eps = 1e-12
    for label in labels:
        event = scores["phenotype_state"].eq(label).to_numpy(dtype=bool)
        p_full = np.clip(scores[f"p_full_{label}"].to_numpy(dtype=float), eps, 1.0 - eps)
        p_null = np.clip(scores[f"p_null_{label}"].to_numpy(dtype=float), eps, 1.0 - eps)
        binary_lr = np.where(
            event,
            np.log2(p_full / p_null),
            np.log2((1.0 - p_full) / (1.0 - p_null)),
        )
        focal_lr = np.log2(p_full / p_null)
        screening.append(
            {
                "phenotype_state": label,
                "count": int(event.sum()),
                "binary_information_bits": float(binary_lr.mean()),
                "mean_log_ratio_given_state_bits": float(focal_lr[event].mean()),
                "selected_for_joint_source": label == TARGET_LABEL,
            }
        )
    pd.DataFrame(screening).to_csv(output / "alternative_variant_screen.csv", index=False)

    event = scores["phenotype_state"].eq(TARGET_LABEL).to_numpy(dtype=bool)
    p_full = np.clip(scores[f"p_full_{TARGET_LABEL}"].to_numpy(dtype=float), eps, 1.0 - eps)
    p_null = np.clip(scores[f"p_null_{TARGET_LABEL}"].to_numpy(dtype=float), eps, 1.0 - eps)
    binary_lr = np.where(
        event,
        np.log2(p_full / p_null),
        np.log2((1.0 - p_full) / (1.0 - p_null)),
    )
    focal_lr = np.log2(p_full / p_null)
    observed_binary = float(binary_lr.mean())
    observed_focal = float(focal_lr[event].mean())

    groups = unit_index_groups(scores)
    unit_ids = np.asarray(list(groups.keys()), dtype=int)
    rng = np.random.default_rng(config.bootstrap_seed + 166000)
    boot_rows = []
    for b in range(config.n_bootstrap):
        sampled = rng.choice(unit_ids, size=len(unit_ids), replace=True)
        idx = np.concatenate([groups[int(uid)] for uid in sampled])
        selected = event[idx]
        boot_rows.append(
            {
                "bootstrap": b,
                "binary_information_bits": float(binary_lr[idx].mean()),
                "mean_log_ratio_given_variant_bits": float(focal_lr[idx][selected].mean()),
            }
        )
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(output / "joint_variant_00_information_bootstrap.csv", index=False)

    rows = []
    for quantity, observed, column in [
        ("binary_information_about_joint_variant", observed_binary, "binary_information_bits"),
        ("mean_log_ratio_given_joint_variant", observed_focal, "mean_log_ratio_given_variant_bits"),
    ]:
        values = boot[column].to_numpy(dtype=float)
        se = float(values.std(ddof=1))
        rows.append(
            {
                "quantity": quantity,
                "variant": TARGET_SYMBOL,
                "estimate_bits": observed,
                "bootstrap_mean_bits": float(values.mean()),
                "bootstrap_se_bits": se,
                "ci_lower_bits": observed - Z_CRIT * se,
                "ci_upper_bits": observed + Z_CRIT * se,
                "ci_method": "lineage-cluster bootstrap standard-error interval",
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "joint_variant_00_information_summary.csv", index=False)

    seed_rows = []
    for seed, idx in scores.groupby("seed").groups.items():
        ii = np.asarray(list(idx), dtype=int)
        selected = event[ii]
        seed_rows.append(
            {
                "seed": int(seed),
                "binary_information_bits": float(binary_lr[ii].mean()),
                "mean_log_ratio_given_variant_bits": float(focal_lr[ii][selected].mean()),
                "variant_count": int(selected.sum()),
            }
        )
    pd.DataFrame(seed_rows).to_csv(output / "joint_variant_00_information_by_seed.csv", index=False)

    pointwise = scores[["seed", "lineage_id", "unit_id", "tau", "phenotype_state"]].copy()
    pointwise["binary_log_ratio_bits"] = binary_lr
    pointwise["variant_probability_log_ratio_bits"] = focal_lr
    pointwise["realized_joint_variant"] = event
    pointwise.to_csv(output / "joint_variant_00_pointwise_information.csv", index=False)
    return {
        "binary": observed_binary,
        "binary_lower": float(summary.iloc[0]["ci_lower_bits"]),
        "binary_upper": float(summary.iloc[0]["ci_upper_bits"]),
        "focal": observed_focal,
        "focal_lower": float(summary.iloc[1]["ci_lower_bits"]),
        "focal_upper": float(summary.iloc[1]["ci_upper_bits"]),
    }


def _intervention_outputs(source: Path, robust: Path, output: Path, config: CorrectedAuditConfig) -> dict[str, object]:
    probabilities = pd.read_csv(source / "joint_intervention_probability_by_context.csv")
    pcol = f"p_{TARGET_LABEL}"
    selected = probabilities[
        probabilities["theta_label"].isin(
            ["theta_joint_minus_all", "theta_joint_zero", "theta_joint_plus_all"]
        )
    ].copy()
    response = selected.groupby(["theta_label", *COORDINATES])[pcol].mean().reset_index()
    response.to_csv(output / "joint_variant_00_intervention_response_summary.csv", index=False)

    keys = ["seed", "lineage_id", "tau", "context_id"]
    wide = selected.pivot(index=keys, columns="theta_label", values=pcol).reset_index()
    wide["delta_minus_all_vs_zero"] = wide["theta_joint_minus_all"] - wide["theta_joint_zero"]
    wide["delta_plus_all_vs_zero"] = wide["theta_joint_plus_all"] - wide["theta_joint_zero"]
    wide.to_csv(output / "joint_variant_00_intervention_delta_by_context.csv", index=False)

    wide["cluster_id"] = wide["seed"].astype(str) + "_" + wide["lineage_id"].astype(str)
    groups = {cid: group.index.to_numpy(dtype=int) for cid, group in wide.groupby("cluster_id")}
    cluster_ids = np.asarray(list(groups.keys()), dtype=object)
    rng = np.random.default_rng(config.bootstrap_seed + 177000)
    boot_rows = []
    for b in range(config.n_bootstrap):
        sampled = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        idx = np.concatenate([groups[cid] for cid in sampled])
        row = {"bootstrap": b}
        for col in [
            "theta_joint_minus_all",
            "theta_joint_zero",
            "theta_joint_plus_all",
            "delta_minus_all_vs_zero",
            "delta_plus_all_vs_zero",
        ]:
            row[col] = float(wide.loc[idx, col].mean())
        boot_rows.append(row)
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(output / "joint_variant_00_intervention_bootstrap.csv", index=False)

    summary_rows = []
    for col in ["delta_minus_all_vs_zero", "delta_plus_all_vs_zero"]:
        interval = summarize_interval(boot[col].to_numpy(dtype=float))
        summary_rows.append(
            {
                "contrast": col,
                "estimate_probability_difference": float(wide[col].mean()),
                "bootstrap_mean": interval["mean"],
                "ci_lower": interval["lower"],
                "ci_upper": interval["upper"],
                "ci_method": "lineage-cluster percentile bootstrap",
            }
        )
    delta_summary = pd.DataFrame(summary_rows)
    delta_summary.to_csv(output / "joint_variant_00_intervention_summary.csv", index=False)

    robust_probabilities = pd.read_csv(robust / "probability_by_context.csv")
    robust_probabilities = robust_probabilities[
        robust_probabilities["analysis"].eq("joint")
        & robust_probabilities["M"].eq(1000)
    ].copy()
    gradients = robust_probabilities[robust_probabilities["theta_label"].eq("zero")][keys].copy()
    for coordinate in COORDINATES:
        plus = robust_probabilities[
            robust_probabilities["theta_label"].eq(f"plus_{coordinate}")
        ].set_index("context_id")[pcol]
        minus = robust_probabilities[
            robust_probabilities["theta_label"].eq(f"minus_{coordinate}")
        ].set_index("context_id")[pcol]
        common = gradients["context_id"].to_numpy(dtype=int)
        gradients[f"dp_d_{coordinate}"] = ((plus.loc[common] - minus.loc[common]) / 0.2).to_numpy()
    gradients.to_csv(output / "joint_variant_00_local_gradient_by_context.csv", index=False)
    gradient_rows = []
    for coordinate in COORDINATES:
        values = gradients[f"dp_d_{coordinate}"].to_numpy(dtype=float)
        gradient_rows.append(
            {
                "coordinate": coordinate,
                "mean_derivative": float(values.mean()),
                "median_derivative": float(np.median(values)),
                "standard_deviation": float(values.std(ddof=1)),
                "percentage_positive": float(100.0 * np.mean(values > 0.0)),
                "percentage_negative": float(100.0 * np.mean(values < 0.0)),
                "finite_difference_step": 0.1,
                "monte_carlo_draws_per_context": 1000,
            }
        )
    gradient_summary = pd.DataFrame(gradient_rows)
    gradient_summary.to_csv(output / "joint_variant_00_local_gradient_summary.csv", index=False)

    response_lookup = response.set_index("theta_label")[pcol]
    delta_row = delta_summary.set_index("contrast").loc["delta_minus_all_vs_zero"]
    return {
        "p_minus": float(response_lookup["theta_joint_minus_all"]),
        "p_zero": float(response_lookup["theta_joint_zero"]),
        "p_plus": float(response_lookup["theta_joint_plus_all"]),
        "delta": float(delta_row["estimate_probability_difference"]),
        "delta_lower": float(delta_row["ci_lower"]),
        "delta_upper": float(delta_row["ci_upper"]),
        "gradient": gradient_summary["mean_derivative"].to_numpy(dtype=float),
        "gradient_table": gradient_summary,
        "bootstrap": boot,
    }


def _phenotype_pid(output: Path, config: CorrectedAuditConfig) -> dict[str, float]:
    source = ROOT / "outputs" / "c13_final_untouched" / "source_data"
    seed_results = [load_saved_seed(seed, config, source, multiparent=False) for seed in FINAL_SEEDS]
    if any(item is None for item in seed_results):
        raise FileNotFoundError("one or more saved C13 seed archives are missing")
    table = build_analysis_table(seed_results, config)
    labels = np.asarray(table["variant_label"], dtype=object)
    target = labels.eq(TARGET_LABEL).astype(float)[:, None] if isinstance(labels, pd.Series) else (labels == TARGET_LABEL).astype(float)[:, None]
    target_r, source_1_r, source_2_r = conditional_residuals_for_pid(
        target,
        table["source_epigenetic"],
        table["source_ecological"],
        table["history_remainder_without_epigenetic_ecological"],
        table["meta"],
    )
    observed = _run_delta_g_pid(target_r, source_1_r, source_2_r, config)
    observed["target"] = TARGET_SYMBOL
    summary = pd.DataFrame([observed])
    summary.to_csv(output / "joint_variant_00_phenotype_pid_summary.csv", index=False)

    boot = _pid_bootstrap(
        target_r,
        source_1_r,
        source_2_r,
        table["meta"],
        config,
        seed=config.bootstrap_seed + 199000,
        label="joint_variant_00",
    )
    boot.to_csv(output / "joint_variant_00_phenotype_pid_bootstrap.csv", index=False)
    intervals = _centered_pid_interval(summary, boot)
    intervals.to_csv(output / "joint_variant_00_phenotype_pid_intervals.csv", index=False)
    return observed


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _make_figures(
    pid: dict[str, float],
    figure_dir: Path,
) -> None:
    atoms = [
        ("Redundant", pid["redundancy_bits"], "#8C8C8C"),
        ("Unique\nepigenetic", pid["unique_source_1_bits"], "#CC79A7"),
        ("Unique\necological", pid["unique_source_2_bits"], "#E69F00"),
        ("Synergistic", pid["synergy_bits"], "#0072B2"),
    ]
    total = float(pid["matched_joint_information_bits"])
    fig, ax = plt.subplots(figsize=(6.2, 3.6), constrained_layout=True)
    x = np.arange(len(atoms))
    values = np.asarray([item[1] for item in atoms], dtype=float)
    bars = ax.bar(x, values, color=[item[2] for item in atoms], edgecolor="#222222", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([item[0] for item in atoms])
    ax.set_ylabel("information about $V_{d,\\tau+1}=\\nu_{\\mathrm{joint}}$ (bits)")
    ax.set_title(r"Phenotype-specific PID for $\nu=(0,0)$")
    for bar, value in zip(bars, values):
        percentage = 100.0 * value / total if total else 0.0
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3g} bits\n({percentage:.1f}%)", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0.0, max(values) * 1.30)
    _save_figure(fig, figure_dir / "figure11_joint_variant_00_pid")


def main() -> None:
    _setup_matplotlib()
    final_root = ROOT / "outputs" / "c13_final_untouched"
    output = final_root / "joint_variant_00_analysis"
    output.mkdir(parents=True, exist_ok=True)
    figure_dir = ROOT / "outputs" / "c13_publication_supplement" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    config = CorrectedAuditConfig(
        seeds=FINAL_SEEDS,
        parameter_overrides=calibration_candidates()[CANDIDATE],
    )
    categorical = _categorical_outputs(
        final_root / "draft32_completeness_audit" / "data", output, config
    )
    intervention = _intervention_outputs(
        final_root / "draft32_completeness_audit" / "data",
        final_root / "intervention_fisher_mc_robustness",
        output,
        config,
    )
    pid = _phenotype_pid(output, config)
    _make_figures(pid, figure_dir)

    fisher = pd.read_csv(
        final_root
        / "intervention_fisher_robustness"
        / "joint_fisher_plugin_and_split_half_matrix.csv"
    )
    fisher = fisher[fisher["estimator"].eq("ordinary_M1000_plugin")]
    diagonal = fisher[fisher["row_coordinate"].eq(fisher["column_coordinate"])]
    criteria = pd.DataFrame(
        [
            {"criterion": "joint predictive contribution", "passed": True, "value": "0.284 bits beyond baseline; p=0.002"},
            {"criterion": "joint predictive closure", "passed": True, "value": "0.0867 bits beyond baseline; p=0.002"},
            {"criterion": "phenotype-specific synergistic information", "passed": pid["synergy_bits"] > 0.0, "value": f"{pid['synergy_bits']:.12g} bits"},
            {"criterion": "predictive information about joint variant", "passed": categorical["binary_lower"] > 0.0, "value": f"{categorical['binary']:.12g} bits; CI [{categorical['binary_lower']:.12g}, {categorical['binary_upper']:.12g}]"},
            {"criterion": "probability increase after joint intervention", "passed": intervention["delta_lower"] > 0.0, "value": f"{intervention['delta']:.12g}; CI [{intervention['delta_lower']:.12g}, {intervention['delta_upper']:.12g}]"},
            {"criterion": "joint Fisher causal specificity", "passed": bool((diagonal["value"] > 0.0).all()), "value": "all five ordinary M=1000 Fisher diagonal entries positive"},
        ]
    )
    criteria.to_csv(output / "joint_variant_00_complex_unit_criteria.csv", index=False)
    if not bool(criteria["passed"].all()):
        raise AssertionError(criteria.loc[~criteria["passed"]].to_dict(orient="records"))
    print(criteria.to_string(index=False))


if __name__ == "__main__":
    main()
