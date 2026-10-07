from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.corrected_numerical_audit import (  # noqa: E402
    CorrectedAuditConfig,
    _CommonRandomThetaBatch,
    _smooth_probabilities,
)
from src.corrected_pilot_model import PilotConfig, build_event_schedule, level_timestamps, reproductive_state  # noqa: E402
from src.final_draft32_completeness_audit import JOINT_THETA_COMPONENTS, NU_LABEL  # noqa: E402
from src.final_numerical_audit import (  # noqa: E402
    BACKGROUND_LEVEL,
    ELC_LEVELS,
    PHENOTYPE_LABELS,
    _integrate_generation_vec,
    _next_generation_start_vec,
    load_saved_seed,
)
from src.recalibration_audit import calibration_candidates  # noqa: E402


SEEDS = (31415927, 27182818, 16180339, 14142135)
M = 1000
HALF_M = M // 2
POINT_TOL = 1e-12
EIGEN_TOL = 1e-10
Z_CRIT = 1.959963984540054
PHENOTYPE_INDEX = {label: i for i, label in enumerate(PHENOTYPE_LABELS)}
NU_INDEX = PHENOTYPE_INDEX[NU_LABEL]
COORDINATES = {
    "epigenetic": ("theta_reg", "theta_stress"),
    "joint": JOINT_THETA_COMPONENTS,
}


def final_config() -> CorrectedAuditConfig:
    return CorrectedAuditConfig(
        seeds=SEEDS,
        n_intervention_draws=M,
        parameter_overrides=calibration_candidates()["C11_pre_reproductive_phenotype"],
    )


def theta_table(analysis: str, h: float) -> pd.DataFrame:
    coordinates = COORDINATES[analysis]
    rows: list[dict[str, float | str]] = []
    zero = {coordinate: 0.0 for coordinate in coordinates}
    rows.append({"theta_label": "zero", **zero})
    for coordinate in coordinates:
        plus = dict(zero)
        minus = dict(zero)
        plus[coordinate] = h
        minus[coordinate] = -h
        rows.append({"theta_label": f"plus_{coordinate}", **plus})
        rows.append({"theta_label": f"minus_{coordinate}", **minus})
    return pd.DataFrame(rows)


def phenotype_codes(life_history: np.ndarray, config: CorrectedAuditConfig) -> np.ndarray:
    maturation = life_history[:, :, 0]
    crosses = maturation >= config.theta_maturation
    crossing = np.full(maturation.shape[0], -1, dtype=int)
    for t_l in range(maturation.shape[1]):
        crossing[(crossing < 0) & crosses[:, t_l]] = t_l
    times = level_timestamps(config.pilot_config(config.seeds[0]))["life_history"]
    crossing_u = np.full(crossing.shape, np.nan, dtype=float)
    valid = crossing >= 0
    crossing_u[valid] = times[crossing[valid]]
    early = valid & (crossing_u < config.pilot_config(config.seeds[0]).early_maturation_u)
    high = life_history[:, -1, 1] >= config.theta_growth
    codes = np.empty(maturation.shape[0], dtype=np.int8)
    codes[early & high] = PHENOTYPE_INDEX["nu_early_maturation_high_growth"]
    codes[early & ~high] = PHENOTYPE_INDEX["early_maturation_low_growth"]
    codes[~early & high] = PHENOTYPE_INDEX["non_early_maturation_high_growth"]
    codes[~early & ~high] = PHENOTYPE_INDEX["non_early_maturation_low_growth"]
    return codes


def _counts(codes: np.ndarray) -> np.ndarray:
    result = np.zeros(codes.shape[:2] + (len(PHENOTYPE_LABELS),), dtype=np.int16)
    for index in range(len(PHENOTYPE_LABELS)):
        result[:, :, index] = np.sum(codes == index, axis=2, dtype=np.int16)
    return result


def simulate_codes(
    analysis: str,
    seed_results: list[dict[str, object]],
    contexts: pd.DataFrame,
    config: CorrectedAuditConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    theta = theta_table(analysis, config.fisher_step)
    coordinates = COORDINATES[analysis]
    seed_map = {int(result["seed"]): result for result in seed_results}
    schedule_cache: dict[int, list[tuple[float, tuple[str, ...]]]] = {}
    meta_parts: list[pd.DataFrame] = []
    code_parts: list[np.ndarray] = []
    grouped = list(contexts.groupby(["seed", "tau"], sort=True))
    for batch, ((seed, tau), group) in enumerate(grouped, start=1):
        print(
            f"{analysis} paired local intervention: seed={int(seed)} tau={int(tau)} "
            f"batch={batch}/{len(grouped)} contexts={len(group)}",
            flush=True,
        )
        seed_data = seed_map[int(seed)]
        lineages = group["lineage_id"].to_numpy(dtype=int)
        repeated_lineages = np.repeat(lineages, M)
        base_n = repeated_lineages.size
        n_theta = len(theta)
        expanded_n = base_n * n_theta
        expanded_config: PilotConfig = config.pilot_config(int(seed), n_lineages=expanded_n)
        if expanded_n not in schedule_cache:
            schedule_cache[expanded_n] = build_event_schedule(expanded_config)
        start = {
            level: np.tile(
                np.asarray(seed_data["full_time_series"][level])[repeated_lineages, int(tau), 0].copy(),
                (n_theta, 1),
            )
            for level in ELC_LEVELS + (BACKGROUND_LEVEL,)
        }
        theta_values = theta[list(coordinates)].to_numpy(dtype=float)
        theta_by_row = np.repeat(theta_values, base_n, axis=0)
        start["epigenetic"][:, :2] += theta_by_row[:, :2]
        if analysis == "joint":
            start["ecological"][:, :3] += theta_by_row[:, 2:5]
            seed_offset = 700_000 + int(seed) * 100_000 + int(tau) * 1009
        else:
            seed_offset = int(seed) * 100_000 + int(tau) * 1009 + 10_000_019
        expanded_lineages = np.tile(repeated_lineages, n_theta)
        rng = _CommonRandomThetaBatch(
            config.intervention_seed + seed_offset,
            n_theta=n_theta,
            base_n=base_n,
        )
        source_series = _integrate_generation_vec(
            start,
            int(tau),
            rng,
            expanded_config,
            seed_data["parameters"],
            schedule_cache[expanded_n],
            d_idx=expanded_lineages,
        )
        reproductive = {
            level: reproductive_state(source_series[level], level, expanded_config)
            for level in ELC_LEVELS + (BACKGROUND_LEVEL,)
        }
        future_start = _next_generation_start_vec(
            reproductive,
            rng,
            expanded_config,
            seed_data["parameters"],
        )
        future_series = _integrate_generation_vec(
            future_start,
            int(tau) + 1,
            rng,
            expanded_config,
            seed_data["parameters"],
            schedule_cache[expanded_n],
            d_idx=expanded_lineages,
        )
        life = future_series["life_history"].reshape(
            n_theta,
            base_n,
            future_series["life_history"].shape[1],
            future_series["life_history"].shape[2],
        )
        codes = np.stack([phenotype_codes(life[i], config) for i in range(n_theta)], axis=0)
        codes = codes.reshape(n_theta, len(group), M)
        meta_parts.append(group[["seed", "lineage_id", "tau", "context_id"]].copy())
        code_parts.append(codes)
    meta = pd.concat(meta_parts, ignore_index=True)
    codes = np.concatenate(code_parts, axis=1)
    order = np.argsort(meta["context_id"].to_numpy(dtype=int))
    return theta, meta.iloc[order].reset_index(drop=True), codes[:, order]


def _smoothed(counts: np.ndarray, draws: int, alpha: float) -> np.ndarray:
    return _smooth_probabilities(counts / float(draws), draws, alpha)


def fisher_matrices(
    counts: np.ndarray,
    theta: pd.DataFrame,
    coordinates: tuple[str, ...],
    config: CorrectedAuditConfig,
    draws: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = {label: i for i, label in enumerate(theta["theta_label"])}
    p0 = _smoothed(counts[labels["zero"]], draws, config.intervention_probability_smoothing_alpha)
    gradients = []
    for coordinate in coordinates:
        plus = _smoothed(counts[labels[f"plus_{coordinate}"]], draws, config.intervention_probability_smoothing_alpha)
        minus = _smoothed(counts[labels[f"minus_{coordinate}"]], draws, config.intervention_probability_smoothing_alpha)
        gradients.append((plus - minus) / (2.0 * config.fisher_step))
    gradient = np.stack(gradients, axis=1)
    fisher = np.einsum("cav,cbv,cv->cab", gradient, gradient, 1.0 / p0)
    fisher = 0.5 * (fisher + np.swapaxes(fisher, 1, 2))
    return gradient, p0, fisher


def split_half_cross_fisher(
    counts_a: np.ndarray,
    counts_b: np.ndarray,
    counts_full: np.ndarray,
    theta: pd.DataFrame,
    coordinates: tuple[str, ...],
    config: CorrectedAuditConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gradient_a, _, _ = fisher_matrices(counts_a, theta, coordinates, config, HALF_M)
    gradient_b, _, _ = fisher_matrices(counts_b, theta, coordinates, config, HALF_M)
    labels = {label: i for i, label in enumerate(theta["theta_label"])}
    p0 = _smoothed(counts_full[labels["zero"]], M, config.intervention_probability_smoothing_alpha)
    ab = np.einsum("cav,cbv,cv->cab", gradient_a, gradient_b, 1.0 / p0)
    ba = np.einsum("cav,cbv,cv->cab", gradient_b, gradient_a, 1.0 / p0)
    result = 0.5 * (ab + ba)
    result = 0.5 * (result + np.swapaxes(result, 1, 2))
    return gradient_a, gradient_b, result


def paired_interval(diff: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    estimate = diff.mean(axis=1) / (2.0 * h)
    standard_error = diff.std(axis=1, ddof=1) / (2.0 * h * np.sqrt(diff.shape[1]))
    return estimate, estimate - Z_CRIT * standard_error, estimate + Z_CRIT * standard_error


def summarize(
    analysis: str,
    theta: pd.DataFrame,
    meta: pd.DataFrame,
    codes: np.ndarray,
    config: CorrectedAuditConfig,
    focal_scores: pd.DataFrame,
    output_dir: Path,
) -> None:
    coordinates = COORDINATES[analysis]
    labels = {label: i for i, label in enumerate(theta["theta_label"])}
    full_counts = _counts(codes)
    counts_a = _counts(codes[:, :, :HALF_M])
    counts_b = full_counts - counts_a
    gradient, _, fisher = fisher_matrices(full_counts, theta, coordinates, config, M)
    gradient_a, gradient_b, fisher_cross = split_half_cross_fisher(
        counts_a,
        counts_b,
        full_counts,
        theta,
        coordinates,
        config,
    )
    fisher_average = 0.5 * (fisher.mean(axis=0) + fisher.mean(axis=0).T)
    cross_average = 0.5 * (fisher_cross.mean(axis=0) + fisher_cross.mean(axis=0).T)
    eig = np.linalg.eigvalsh(fisher_average)
    cross_eig = np.linalg.eigvalsh(cross_average)

    matrix_rows = []
    for estimator, matrix, eigenvalues in (
        ("ordinary_M1000_plugin", fisher_average, eig),
        ("split_half_cross_product", cross_average, cross_eig),
    ):
        for i, row_coordinate in enumerate(coordinates):
            for j, column_coordinate in enumerate(coordinates):
                plugin_value = float(fisher_average[i, j])
                value = float(matrix[i, j])
                matrix_rows.append(
                    {
                        "analysis": analysis,
                        "estimator": estimator,
                        "row_coordinate": row_coordinate,
                        "column_coordinate": column_coordinate,
                        "value": value,
                        "absolute_difference_from_plugin": value - plugin_value,
                        "relative_difference_from_plugin": (value - plugin_value) / plugin_value if plugin_value else np.nan,
                        "minimum_eigenvalue": float(eigenvalues[0]),
                        "maximum_eigenvalue": float(eigenvalues[-1]),
                        "numerical_rank_tolerance_1e_minus_10": int(np.sum(eigenvalues > EIGEN_TOL)),
                    }
                )
    pd.DataFrame(matrix_rows).to_csv(output_dir / f"{analysis}_fisher_plugin_and_split_half_matrix.csv", index=False)
    pd.DataFrame(
        [
            {
                "analysis": analysis,
                "estimator": estimator,
                "eigenvalue_index": index,
                "eigenvalue": float(value),
                "negative": bool(value < 0.0),
            }
            for estimator, values in (
                ("ordinary_M1000_plugin", eig),
                ("split_half_cross_product", cross_eig),
            )
            for index, value in enumerate(values)
        ]
    ).to_csv(output_dir / f"{analysis}_fisher_plugin_and_split_half_eigenvalues.csv", index=False)

    context = meta.copy()
    score_columns = ["seed", "lineage_id", "tau", "focal_nu_probability_log_ratio_bits"]
    context = context.merge(focal_scores[score_columns], on=["seed", "lineage_id", "tau"], how="left", validate="one_to_one")
    if context["focal_nu_probability_log_ratio_bits"].isna().any():
        raise AssertionError(f"missing focal-nu information rows for {analysis}")
    context["focal_nu_log_ratio_positive"] = context["focal_nu_probability_log_ratio_bits"] > 0.0
    point_flags = []
    interval_flags = []
    gradient_rows = []
    for index, coordinate in enumerate(coordinates):
        plus = codes[labels[f"plus_{coordinate}"]] == NU_INDEX
        minus = codes[labels[f"minus_{coordinate}"]] == NU_INDEX
        difference = plus.astype(np.int8) - minus.astype(np.int8)
        paired_estimate, ci_lower, ci_upper = paired_interval(difference, config.fisher_step)
        point = gradient[:, index, NU_INDEX]
        diag = fisher[:, index, index]
        cross_diag = fisher_cross[:, index, index]
        interval_supported = (ci_lower > 0.0) | (ci_upper < 0.0)
        point_supported = (
            context["focal_nu_log_ratio_positive"].to_numpy(bool)
            & (np.abs(point) > POINT_TOL)
            & (diag > 0.0)
        )
        interval_joint = (
            context["focal_nu_log_ratio_positive"].to_numpy(bool)
            & interval_supported
            & (diag > 0.0)
        )
        point_flags.append(point_supported)
        interval_flags.append(interval_joint)
        context[f"gradient_{coordinate}"] = point
        context[f"paired_gradient_{coordinate}"] = paired_estimate
        context[f"paired_ci_lower_{coordinate}"] = ci_lower
        context[f"paired_ci_upper_{coordinate}"] = ci_upper
        context[f"plugin_fisher_diagonal_{coordinate}"] = diag
        context[f"cross_fisher_diagonal_{coordinate}"] = cross_diag
        context[f"same_context_point_supported_{coordinate}"] = point_supported
        context[f"same_context_interval_supported_{coordinate}"] = interval_joint
        for seed_label, values in [("all", np.arange(len(context))), *[(str(seed), np.asarray(indices, dtype=int)) for seed, indices in context.groupby("seed").groups.items()]]:
            selected = point[values]
            selected_low = ci_lower[values]
            selected_high = ci_upper[values]
            gradient_rows.append(
                {
                    "analysis": analysis,
                    "seed": seed_label,
                    "coordinate": coordinate,
                    "n_contexts": len(values),
                    "mean": float(np.mean(selected)),
                    "median": float(np.median(selected)),
                    "standard_deviation": float(np.std(selected, ddof=1)),
                    "minimum": float(np.min(selected)),
                    "q025": float(np.quantile(selected, 0.025)),
                    "q25": float(np.quantile(selected, 0.25)),
                    "q75": float(np.quantile(selected, 0.75)),
                    "q975": float(np.quantile(selected, 0.975)),
                    "maximum": float(np.max(selected)),
                    "percentage_positive_point": float(100.0 * np.mean(selected > POINT_TOL)),
                    "percentage_negative_point": float(100.0 * np.mean(selected < -POINT_TOL)),
                    "percentage_positive_interval": float(100.0 * np.mean(selected_low > 0.0)),
                    "percentage_negative_interval": float(100.0 * np.mean(selected_high < 0.0)),
                }
            )
    context["same_context_point_supported_any_coordinate"] = np.column_stack(point_flags).any(axis=1)
    context["same_context_interval_supported_any_coordinate"] = np.column_stack(interval_flags).any(axis=1)
    context.to_csv(output_dir / f"{analysis}_same_context_robustness_by_context.csv", index=False)
    pd.DataFrame(gradient_rows).to_csv(output_dir / f"{analysis}_local_gradient_summary.csv", index=False)

    summary_rows = []
    for seed_label, group in [("all", context), *[(str(seed), group) for seed, group in context.groupby("seed")]]:
        summary_rows.append(
            {
                "analysis": analysis,
                "seed": seed_label,
                "n_contexts": len(group),
                "n_positive_focal_nu_log_ratio": int(group["focal_nu_log_ratio_positive"].sum()),
                "percentage_positive_focal_nu_log_ratio": float(100.0 * group["focal_nu_log_ratio_positive"].mean()),
                "n_same_context_point_supported": int(group["same_context_point_supported_any_coordinate"].sum()),
                "percentage_same_context_point_supported": float(100.0 * group["same_context_point_supported_any_coordinate"].mean()),
                "n_same_context_interval_supported": int(group["same_context_interval_supported_any_coordinate"].sum()),
                "percentage_same_context_interval_supported": float(100.0 * group["same_context_interval_supported_any_coordinate"].mean()),
            }
        )
    pd.DataFrame(summary_rows).to_csv(output_dir / f"{analysis}_same_context_robustness_summary.csv", index=False)


def main() -> None:
    config = final_config()
    primary_root = ROOT / "outputs" / "corrected_pre_reproductive_final_untouched"
    corrected = primary_root / "corrected_numerical_audit" / "data"
    completeness = primary_root / "draft32_completeness_audit" / "data"
    source_dir = primary_root / "source_data"
    output_dir = primary_root / "intervention_fisher_robustness"
    output_dir.mkdir(parents=True, exist_ok=True)
    contexts = pd.read_csv(corrected / "intervention_generation_stratified_contexts.csv")
    if len(contexts) != config.n_contexts:
        raise AssertionError(f"expected {config.n_contexts} contexts, found {len(contexts)}")
    seed_results = []
    for seed in config.seeds:
        result = load_saved_seed(int(seed), config, source_dir)
        if result is None:
            raise FileNotFoundError(f"missing corrected temporal-architecture archive for seed {seed}")
        seed_results.append(result)
    metadata = {
        "configuration": asdict(config),
        "monte_carlo_draws": M,
        "finite_difference_step": config.fisher_step,
        "paired_interval": "normal 95 percent interval for paired Bernoulli outcome differences under common random numbers",
        "ordinary_fisher": "M=1000 plug-in estimator",
        "robustness_fisher": "independent 500-draw split-half cross-product estimator",
        "point_tolerance": POINT_TOL,
        "eigenvalue_tolerance": EIGEN_TOL,
        "trajectory_architecture": "full retained series; pre-reproductive analysis segments; reproductive state at u_R=0.75; early-maturation boundary u=0.60",
    }
    (output_dir / "corrected_temporal_fisher_robustness_metadata.json").write_text(json.dumps(metadata, indent=2))
    for analysis, score_path in (
        ("epigenetic", corrected / "categorical_phenotype_crossfit_scores.csv"),
        ("joint", completeness / "joint_categorical_phenotype_crossfit_scores.csv"),
    ):
        if not score_path.exists():
            raise FileNotFoundError(score_path)
        theta, meta, codes = simulate_codes(analysis, seed_results, contexts, config)
        summarize(analysis, theta, meta, codes, config, pd.read_csv(score_path), output_dir)
    print(f"wrote corrected temporal Fisher robustness outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
