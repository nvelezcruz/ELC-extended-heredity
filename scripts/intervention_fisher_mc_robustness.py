from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import os
import sys
from typing import Mapping

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
from src.corrected_pilot_model import (  # noqa: E402
    PilotConfig,
    build_event_schedule,
    level_timestamps,
    reproductive_state,
)
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


RUN_OUTPUT_NAME = os.environ.get("ROBUSTNESS_RUN_OUTPUT", "final_c8_untouched_audit")
CANDIDATE_NAME = os.environ.get(
    "ROBUSTNESS_CANDIDATE", "C8_stronger_growth_allocation"
)
SEEDS = tuple(
    int(value)
    for value in os.environ.get("ROBUSTNESS_SEEDS", "1207,2411,3613,4817").split(",")
)
PRIMARY_ROOT = ROOT / "outputs" / RUN_OUTPUT_NAME
M_VALUES = (100, 250, 500, 1000)
MAX_M = max(M_VALUES)
TOL = 1e-12
RANK_TOL = 1e-10
Z_CRIT = 1.959963984540054
PHENO_INDEX = {label: i for i, label in enumerate(PHENOTYPE_LABELS)}
NU_INDEX = PHENO_INDEX[NU_LABEL]
ANALYSIS_COORDS = {
    "epigenetic": ("theta_reg", "theta_stress"),
    "joint": JOINT_THETA_COMPONENTS,
}


def final_config() -> CorrectedAuditConfig:
    return CorrectedAuditConfig(
        seeds=SEEDS,
        n_intervention_draws=MAX_M,
        parameter_overrides=calibration_candidates()[CANDIDATE_NAME],
    )


def phenotype_codes_from_life(life: np.ndarray, config: CorrectedAuditConfig) -> np.ndarray:
    maturation = life[:, :, 0]
    crosses = maturation >= config.theta_maturation
    crossing = np.full(maturation.shape[0], -1, dtype=int)
    for t_l in range(maturation.shape[1]):
        crossing[(crossing < 0) & crosses[:, t_l]] = t_l
    life_times = level_timestamps(config.pilot_config(config.seeds[0]))["life_history"]
    crossing_u = np.full(crossing.shape, np.nan, dtype=float)
    valid = crossing >= 0
    crossing_u[valid] = life_times[crossing[valid]]
    early = valid & (crossing_u < config.pilot_config(config.seeds[0]).early_maturation_u)
    high = life[:, -1, 1] >= config.theta_growth
    codes = np.empty(maturation.shape[0], dtype=np.int16)
    codes[early & high] = PHENO_INDEX["nu_early_maturation_high_growth"]
    codes[early & ~high] = PHENO_INDEX["early_maturation_low_growth"]
    codes[~early & high] = PHENO_INDEX["non_early_maturation_high_growth"]
    codes[~early & ~high] = PHENO_INDEX["non_early_maturation_low_growth"]
    return codes


def theta_table(analysis: str, h: float) -> tuple[pd.DataFrame, dict[str, tuple[str, str]]]:
    if analysis == "epigenetic":
        coords = ANALYSIS_COORDS[analysis]
        rows = [("zero", (0.0, 0.0))]
        for j, coord in enumerate(coords):
            plus = np.zeros(2, dtype=float)
            minus = np.zeros(2, dtype=float)
            plus[j] = h
            minus[j] = -h
            rows.append((f"plus_{coord}", tuple(plus)))
            rows.append((f"minus_{coord}", tuple(minus)))
        df_rows = []
        for label, vector in rows:
            df_rows.append({"theta_label": label, "theta_reg": vector[0], "theta_stress": vector[1]})
        pair_map = {coord: (f"plus_{coord}", f"minus_{coord}") for coord in coords}
        return pd.DataFrame(df_rows), pair_map
    if analysis == "joint":
        coords = ANALYSIS_COORDS[analysis]
        rows = [("zero", tuple(np.zeros(5, dtype=float)))]
        for j, coord in enumerate(coords):
            plus = np.zeros(5, dtype=float)
            minus = np.zeros(5, dtype=float)
            plus[j] = h
            minus[j] = -h
            rows.append((f"plus_{coord}", tuple(plus)))
            rows.append((f"minus_{coord}", tuple(minus)))
        df_rows = []
        for label, vector in rows:
            row = {"theta_label": label}
            row.update({coord: float(vector[i]) for i, coord in enumerate(coords)})
            df_rows.append(row)
        pair_map = {coord: (f"plus_{coord}", f"minus_{coord}") for coord in coords}
        return pd.DataFrame(df_rows), pair_map
    raise ValueError(analysis)


def counts_for_codes(codes: np.ndarray, m: int) -> np.ndarray:
    # codes shape: n_theta x n_contexts x MAX_M
    prefix = codes[:, :, :m]
    counts = np.zeros(prefix.shape[:2] + (len(PHENOTYPE_LABELS),), dtype=np.int16)
    for k in range(len(PHENOTYPE_LABELS)):
        counts[:, :, k] = np.sum(prefix == k, axis=2, dtype=np.int16)
    return counts


def fisher_from_counts(
    count_by_theta: dict[str, np.ndarray],
    pair_map: Mapping[str, tuple[str, str]],
    *,
    m: int,
    h: float,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p0 = _smooth_probabilities(count_by_theta["zero"] / float(m), m, alpha)
    derivs = []
    for coord, (plus_label, minus_label) in pair_map.items():
        p_plus = _smooth_probabilities(count_by_theta[plus_label] / float(m), m, alpha)
        p_minus = _smooth_probabilities(count_by_theta[minus_label] / float(m), m, alpha)
        derivs.append((p_plus - p_minus) / (2.0 * h))
    grad = np.stack(derivs, axis=1)
    mats = np.einsum("cav,cbv,cv->cab", grad, grad, 1.0 / p0)
    mats = 0.5 * (mats + np.swapaxes(mats, 1, 2))
    return grad, p0, mats


def paired_interval(diff: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # diff shape: n_contexts x m, entries are +1, 0, or -1 for paired outcomes.
    m = diff.shape[1]
    raw_mean = diff.mean(axis=1) / (2.0 * h)
    raw_sd = diff.std(axis=1, ddof=1) / (2.0 * h)
    se = raw_sd / np.sqrt(float(m))
    return raw_mean, raw_mean - Z_CRIT * se, raw_mean + Z_CRIT * se


def simulate_analysis(
    analysis: str,
    seed_results: list[dict[str, object]],
    contexts: pd.DataFrame,
    config: CorrectedAuditConfig,
    output_dir: Path,
) -> dict[str, object]:
    theta_df, pair_map = theta_table(analysis, config.fisher_step)
    coords = ANALYSIS_COORDS[analysis]
    seed_map = {int(sd["seed"]): sd for sd in seed_results}
    schedule_cache: dict[int, list[tuple[float, tuple[str, ...]]]] = {}
    n_theta = len(theta_df)
    theta_index = {str(label): i for i, label in enumerate(theta_df["theta_label"])}

    counts_by_m: dict[int, dict[str, list[np.ndarray]]] = {
        m: {label: [] for label in theta_df["theta_label"]} for m in M_VALUES
    }
    meta_parts: list[pd.DataFrame] = []
    pair_diffs: dict[int, dict[str, list[np.ndarray]]] = {
        m: {coord: [] for coord in coords} for m in M_VALUES
    }

    grouped_contexts = list(contexts.groupby(["seed", "tau"], sort=True))
    for batch_index, ((seed, tau), group) in enumerate(grouped_contexts, start=1):
        print(
            f"  {analysis}: seed {int(seed)} tau {int(tau)} "
            f"({batch_index}/{len(grouped_contexts)}), contexts={len(group)}",
            flush=True,
        )
        seed_data = seed_map[int(seed)]
        lineages = group["lineage_id"].to_numpy(dtype=int)
        context_ids = group["context_id"].to_numpy(dtype=int)
        repeated_lineages = np.repeat(lineages, MAX_M)
        base_n = repeated_lineages.size
        expanded_n = n_theta * base_n
        expanded_config: PilotConfig = config.pilot_config(int(seed), n_lineages=expanded_n)
        if expanded_n not in schedule_cache:
            schedule_cache[expanded_n] = build_event_schedule(expanded_config)
        start = {
            level: np.tile(
                np.asarray(seed_data["full_time_series"][level])[
                    repeated_lineages, int(tau), 0
                ].copy(),
                (n_theta, 1),
            )
            for level in ELC_LEVELS + (BACKGROUND_LEVEL,)
        }
        if analysis == "epigenetic":
            theta_mat = theta_df[["theta_reg", "theta_stress"]].to_numpy(dtype=float)
            start["epigenetic"][:, :2] += np.repeat(theta_mat, base_n, axis=0)
            seed_offset = int(seed) * 100_000 + int(tau) * 1009
        else:
            theta_mat = theta_df[list(coords)].to_numpy(dtype=float)
            theta_by_row = np.repeat(theta_mat, base_n, axis=0)
            start["epigenetic"][:, :2] += theta_by_row[:, :2]
            start["ecological"][:, :3] += theta_by_row[:, 2:5]
            seed_offset = 700_000 + int(seed) * 100_000 + int(tau) * 1009
        expanded_lineages = np.tile(repeated_lineages, n_theta)
        rng = _CommonRandomThetaBatch(
            config.intervention_seed + seed_offset,
            n_theta=n_theta,
            base_n=base_n,
        )
        source_retained = _integrate_generation_vec(
            start,
            int(tau),
            rng,
            expanded_config,
            seed_data["parameters"],
            schedule_cache[expanded_n],
            d_idx=expanded_lineages,
        )
        prev_final = {
            level: reproductive_state(source_retained[level], level, expanded_config)
            for level in ELC_LEVELS + (BACKGROUND_LEVEL,)
        }
        future_start = _next_generation_start_vec(prev_final, rng, expanded_config, seed_data["parameters"])
        future_retained = _integrate_generation_vec(
            future_start,
            int(tau) + 1,
            rng,
            expanded_config,
            seed_data["parameters"],
            schedule_cache[expanded_n],
            d_idx=expanded_lineages,
        )
        life = future_retained["life_history"].reshape(
            n_theta,
            base_n,
            future_retained["life_history"].shape[1],
            future_retained["life_history"].shape[2],
        )
        codes = np.stack([phenotype_codes_from_life(life[i], config) for i in range(n_theta)], axis=0)
        codes = codes.reshape(n_theta, len(context_ids), MAX_M)
        meta_parts.append(group[["seed", "lineage_id", "tau", "context_id"]].copy())
        for m in M_VALUES:
            counts = counts_for_codes(codes, m)
            for label, idx in theta_index.items():
                counts_by_m[m][label].append(counts[idx])
            for coord, (plus_label, minus_label) in pair_map.items():
                diff = (
                    (codes[theta_index[plus_label], :, :m] == NU_INDEX).astype(np.int8)
                    - (codes[theta_index[minus_label], :, :m] == NU_INDEX).astype(np.int8)
                )
                pair_diffs[m][coord].append(diff)

    meta = pd.concat(meta_parts, ignore_index=True).sort_values("context_id").reset_index(drop=True)
    all_counts: dict[int, dict[str, np.ndarray]] = {}
    all_diffs: dict[int, dict[str, np.ndarray]] = {}
    for m in M_VALUES:
        all_counts[m] = {label: np.vstack(parts) for label, parts in counts_by_m[m].items()}
        all_diffs[m] = {coord: np.vstack(parts) for coord, parts in pair_diffs[m].items()}

    return {
        "analysis": analysis,
        "theta_df": theta_df,
        "pair_map": pair_map,
        "coords": coords,
        "meta": meta,
        "counts": all_counts,
        "diffs": all_diffs,
    }


def summarize_analysis(sim: dict[str, object], config: CorrectedAuditConfig) -> dict[str, pd.DataFrame]:
    analysis = str(sim["analysis"])
    coords = tuple(sim["coords"])
    pair_map = dict(sim["pair_map"])
    meta = sim["meta"].copy()
    count_by_m = sim["counts"]
    diffs_by_m = sim["diffs"]

    prob_rows = []
    gradient_rows = []
    paired_rows = []
    fisher_long_rows = []
    fisher_eig_rows = []
    context_rank_rows = []
    context_diag_rows = []
    gradient_summary_rows = []
    convergence_rows = []
    by_context_for_qualification = {}
    sign_by_m: dict[int, pd.DataFrame] = {}

    for m in M_VALUES:
        counts = count_by_m[m]
        for theta_label, arr in counts.items():
            probs = arr / float(m)
            for i, row in meta.iterrows():
                out = {
                    "analysis": analysis,
                    "M": int(m),
                    "seed": int(row["seed"]),
                    "lineage_id": int(row["lineage_id"]),
                    "tau": int(row["tau"]),
                    "context_id": int(row["context_id"]),
                    "theta_label": theta_label,
                }
                out.update({f"count_{label}": int(arr[i, PHENO_INDEX[label]]) for label in PHENOTYPE_LABELS})
                out.update({f"p_{label}": float(probs[i, PHENO_INDEX[label]]) for label in PHENOTYPE_LABELS})
                prob_rows.append(out)

        grad, p0, mats = fisher_from_counts(
            counts,
            pair_map,
            m=m,
            h=config.fisher_step,
            alpha=config.intervention_probability_smoothing_alpha,
        )
        eig_context = np.linalg.eigvalsh(mats)
        rank_context = np.sum(eig_context > RANK_TOL, axis=1)
        f_avg = 0.5 * (mats.mean(axis=0) + mats.mean(axis=0).T)
        eig_avg = np.linalg.eigvalsh(f_avg)
        rank_avg = int(np.sum(eig_avg > RANK_TOL))

        grad_nu = grad[:, :, NU_INDEX]
        sign_df = meta.copy()
        sign_df["analysis"] = analysis
        sign_df["M"] = int(m)
        sign_df["rank_tol_1e_minus_10"] = rank_context.astype(int)
        sign_df["min_eigenvalue"] = eig_context[:, 0]
        sign_df["context_matrix_psd_tol_1e_minus_10"] = eig_context[:, 0] >= -RANK_TOL
        for j, coord in enumerate(coords):
            raw_mean, ci_low, ci_high = paired_interval(diffs_by_m[m][coord], config.fisher_step)
            point = grad_nu[:, j]
            diag = mats[:, j, j]
            sign_df[f"dp_nu_d_{coord}"] = point
            sign_df[f"positive_direction_{coord}"] = point > TOL
            sign_df[f"negative_direction_{coord}"] = point < -TOL
            sign_df[f"zero_derivative_{coord}"] = np.abs(point) <= TOL
            sign_df[f"paired_raw_dp_nu_d_{coord}"] = raw_mean
            sign_df[f"paired_ci_lower_{coord}"] = ci_low
            sign_df[f"paired_ci_upper_{coord}"] = ci_high
            sign_df[f"paired_interval_positive_{coord}"] = ci_low > 0.0
            sign_df[f"paired_interval_negative_{coord}"] = ci_high < 0.0
            sign_df[f"paired_interval_excludes_zero_{coord}"] = (ci_low > 0.0) | (ci_high < 0.0)
            sign_df[f"fisher_diag_{coord}"] = diag
            sign_df[f"fisher_diag_gt_zero_{coord}"] = diag > 0.0
            sign_df[f"fisher_diag_gt_tol_{coord}"] = diag > RANK_TOL
            for i, row in meta.iterrows():
                gradient_rows.append(
                    {
                        "analysis": analysis,
                        "M": int(m),
                        "seed": int(row["seed"]),
                        "lineage_id": int(row["lineage_id"]),
                        "tau": int(row["tau"]),
                        "context_id": int(row["context_id"]),
                        "coordinate": coord,
                        "dp_nu_d_theta": float(point[i]),
                        "paired_raw_dp_nu_d_theta": float(raw_mean[i]),
                        "paired_ci_lower": float(ci_low[i]),
                        "paired_ci_upper": float(ci_high[i]),
                        "positive_point": bool(point[i] > TOL),
                        "negative_point": bool(point[i] < -TOL),
                        "zero_point": bool(abs(point[i]) <= TOL),
                        "positive_interval": bool(ci_low[i] > 0.0),
                        "negative_interval": bool(ci_high[i] < 0.0),
                        "interval_excludes_zero": bool((ci_low[i] > 0.0) or (ci_high[i] < 0.0)),
                    }
                )
                dvals = diffs_by_m[m][coord][i]
                paired_rows.append(
                    {
                        "analysis": analysis,
                        "M": int(m),
                        "seed": int(row["seed"]),
                        "lineage_id": int(row["lineage_id"]),
                        "tau": int(row["tau"]),
                        "context_id": int(row["context_id"]),
                        "coordinate": coord,
                        "n_plus_minus_pairs": int(m),
                        "n_plus_gt_minus": int(np.sum(dvals > 0)),
                        "n_plus_eq_minus": int(np.sum(dvals == 0)),
                        "n_plus_lt_minus": int(np.sum(dvals < 0)),
                    }
                )
            values = point
            gradient_summary_rows.append(
                {
                    "analysis": analysis,
                    "M": int(m),
                    "seed": "all",
                    "coordinate": coord,
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "sd": float(np.std(values, ddof=1)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "q025": float(np.quantile(values, 0.025)),
                    "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                    "q975": float(np.quantile(values, 0.975)),
                    "pct_positive_point": float(100.0 * np.mean(values > TOL)),
                    "pct_negative_point": float(100.0 * np.mean(values < -TOL)),
                    "pct_zero_point": float(100.0 * np.mean(np.abs(values) <= TOL)),
                    "pct_positive_interval": float(100.0 * np.mean(ci_low > 0.0)),
                    "pct_negative_interval": float(100.0 * np.mean(ci_high < 0.0)),
                    "pct_interval_excludes_zero": float(100.0 * np.mean((ci_low > 0.0) | (ci_high < 0.0))),
                }
            )
            for seed, seed_idx in meta.groupby("seed").groups.items():
                idx = np.asarray(list(seed_idx), dtype=int)
                seed_values = point[idx]
                seed_ci_low = ci_low[idx]
                seed_ci_high = ci_high[idx]
                gradient_summary_rows.append(
                    {
                        "analysis": analysis,
                        "M": int(m),
                        "seed": int(seed),
                        "coordinate": coord,
                        "mean": float(np.mean(seed_values)),
                        "median": float(np.median(seed_values)),
                        "sd": float(np.std(seed_values, ddof=1)),
                        "min": float(np.min(seed_values)),
                        "max": float(np.max(seed_values)),
                        "q025": float(np.quantile(seed_values, 0.025)),
                        "q25": float(np.quantile(seed_values, 0.25)),
                        "q75": float(np.quantile(seed_values, 0.75)),
                        "q975": float(np.quantile(seed_values, 0.975)),
                        "pct_positive_point": float(100.0 * np.mean(seed_values > TOL)),
                        "pct_negative_point": float(100.0 * np.mean(seed_values < -TOL)),
                        "pct_zero_point": float(100.0 * np.mean(np.abs(seed_values) <= TOL)),
                        "pct_positive_interval": float(100.0 * np.mean(seed_ci_low > 0.0)),
                        "pct_negative_interval": float(100.0 * np.mean(seed_ci_high < 0.0)),
                        "pct_interval_excludes_zero": float(100.0 * np.mean((seed_ci_low > 0.0) | (seed_ci_high < 0.0))),
                    }
                )

        sign_df["any_positive_direction"] = np.column_stack([sign_df[f"positive_direction_{coord}"].to_numpy(bool) for coord in coords]).any(axis=1)
        sign_df["any_negative_direction"] = np.column_stack([sign_df[f"negative_direction_{coord}"].to_numpy(bool) for coord in coords]).any(axis=1)
        sign_df["any_tested_direction_increases_p_nu"] = (
            sign_df["any_positive_direction"] | sign_df["any_negative_direction"]
        )
        sign_df["all_derivatives_zero"] = np.column_stack([sign_df[f"zero_derivative_{coord}"].to_numpy(bool) for coord in coords]).all(axis=1)
        sign_df["any_interval_supported_direction"] = np.column_stack(
            [sign_df[f"paired_interval_excludes_zero_{coord}"].to_numpy(bool) for coord in coords]
        ).any(axis=1)
        sign_by_m[m] = sign_df
        if m == 1000:
            by_context_for_qualification = {
                "sign_df": sign_df.copy(),
                "mats": mats.copy(),
                "eig_context": eig_context.copy(),
                "grad": grad.copy(),
                "p0": p0.copy(),
            }

        for i, row_component in enumerate(coords):
            for j, col_component in enumerate(coords):
                fisher_long_rows.append(
                    {
                        "analysis": analysis,
                        "M": int(m),
                        "row_component": row_component,
                        "col_component": col_component,
                        "fisher_information": float(f_avg[i, j]),
                    }
                )
        for idx, eig in enumerate(eig_avg):
            fisher_eig_rows.append(
                {
                    "analysis": analysis,
                    "M": int(m),
                    "eigenvalue_index": int(idx),
                    "eigenvalue": float(eig),
                    "rank_tol_1e_minus_10": rank_avg,
                    "minimum_eigenvalue": float(eig_avg[0]),
                }
            )
        for rank_value, n_rank in pd.Series(rank_context).value_counts().sort_index().items():
            context_rank_rows.append(
                {
                    "analysis": analysis,
                    "M": int(m),
                    "rank_tol_1e_minus_10": int(rank_value),
                    "n_contexts": int(n_rank),
                    "percentage": float(100.0 * n_rank / len(rank_context)),
                }
            )
        for j, coord in enumerate(coords):
            diag = mats[:, j, j]
            context_diag_rows.append(
                {
                    "analysis": analysis,
                    "M": int(m),
                    "coordinate": coord,
                    "pct_diag_gt_zero": float(100.0 * np.mean(diag > 0.0)),
                    "pct_diag_gt_tol_1e_minus_10": float(100.0 * np.mean(diag > RANK_TOL)),
                }
            )
        convergence_rows.append(
            {
                "analysis": analysis,
                "M": int(m),
                "n_contexts": int(len(meta)),
                "context_averaged_gradient": json.dumps({coord: float(np.mean(grad_nu[:, j])) for j, coord in enumerate(coords)}),
                "averaged_fisher_rank_tol_1e_minus_10": rank_avg,
                "averaged_fisher_min_eigenvalue": float(eig_avg[0]),
                "averaged_fisher_max_eigenvalue": float(eig_avg[-1]),
                "pct_any_positive_coordinate_point": float(100.0 * sign_df["any_positive_direction"].mean()),
                "pct_any_negative_coordinate_point": float(100.0 * sign_df["any_negative_direction"].mean()),
                "pct_any_tested_direction_point": float(100.0 * sign_df["any_tested_direction_increases_p_nu"].mean()),
                "pct_any_interval_supported_direction": float(100.0 * sign_df["any_interval_supported_direction"].mean()),
                "pct_all_derivatives_zero_point": float(100.0 * sign_df["all_derivatives_zero"].mean()),
            }
        )

    stability_rows = []
    for coord in coords:
        signs = []
        for m in (250, 500, 1000):
            values = sign_by_m[m][f"dp_nu_d_{coord}"].to_numpy(dtype=float)
            signs.append(np.where(values > TOL, 1, np.where(values < -TOL, -1, 0)))
        stable = (signs[0] == signs[1]) & (signs[1] == signs[2])
        stability_rows.append(
            {
                "analysis": analysis,
                "coordinate": coord,
                "pct_sign_unchanged_across_M_250_500_1000": float(100.0 * np.mean(stable)),
                "n_contexts_unchanged": int(np.sum(stable)),
                "n_contexts": int(len(stable)),
            }
        )
    return {
        "probability_by_context": pd.DataFrame(prob_rows),
        "gradient_by_context": pd.DataFrame(gradient_rows),
        "paired_difference_counts_by_context": pd.DataFrame(paired_rows),
        "fisher_matrix_by_M": pd.DataFrame(fisher_long_rows),
        "fisher_eigenvalues_by_M": pd.DataFrame(fisher_eig_rows),
        "fisher_context_rank_distribution": pd.DataFrame(context_rank_rows),
        "fisher_context_diagonal_summary": pd.DataFrame(context_diag_rows),
        "gradient_summary": pd.DataFrame(gradient_summary_rows),
        "convergence_summary": pd.DataFrame(convergence_rows),
        "derivative_sign_stability": pd.DataFrame(stability_rows),
        "qualification_inputs": by_context_for_qualification,
    }


def qualification_tables(
    summaries: Mapping[str, dict[str, object]],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if os.environ.get("ROBUSTNESS_USE_CURRENT_LOG_RATIOS", "0") == "1":
        contexts = pd.read_csv(
            PRIMARY_ROOT
            / "corrected_numerical_audit"
            / "data"
            / "intervention_generation_stratified_contexts.csv"
        )
        score_paths = {
            "epigenetic": PRIMARY_ROOT
            / "corrected_numerical_audit"
            / "data"
            / "categorical_phenotype_crossfit_scores.csv",
            "joint_epigenetic_ecological": PRIMARY_ROOT
            / "draft32_completeness_audit"
            / "data"
            / "joint_categorical_phenotype_crossfit_scores.csv",
        }
        prior_parts = []
        for analysis_label, score_path in score_paths.items():
            scores = pd.read_csv(score_path)
            matched = contexts.merge(
                scores[
                    [
                        "seed",
                        "lineage_id",
                        "tau",
                        "focal_nu_probability_log_ratio_bits",
                    ]
                ],
                on=["seed", "lineage_id", "tau"],
                how="left",
                validate="one_to_one",
            )
            if matched["focal_nu_probability_log_ratio_bits"].isna().any():
                raise AssertionError(
                    f"missing current focal-nu log-ratio rows for {analysis_label}"
                )
            matched["analysis"] = analysis_label
            matched["A_log_ratio_positive"] = (
                matched["focal_nu_probability_log_ratio_bits"] > 0.0
            )
            prior_parts.append(matched)
        prior = pd.concat(prior_parts, ignore_index=True)
    else:
        prior = pd.read_csv(
            ROOT
            / "outputs"
            / "main39_code_audit"
            / "context_specific_qualification_by_context.csv"
        )
    rows_by_context = []
    summary_rows = []
    for analysis in ("epigenetic", "joint"):
        q = summaries[analysis]["qualification_inputs"]
        sign_df = q["sign_df"].copy()
        prior_analysis = "joint_epigenetic_ecological" if analysis == "joint" else analysis
        prior_a = prior[prior["analysis"] == prior_analysis][
            ["seed", "lineage_id", "tau", "context_id", "focal_nu_probability_log_ratio_bits", "A_log_ratio_positive"]
        ].copy()
        merged = sign_df.merge(prior_a, on=["seed", "lineage_id", "tau", "context_id"], how="left", validate="one_to_one")
        if merged["A_log_ratio_positive"].isna().any():
            raise AssertionError(f"missing matched focal-nu log-ratio rows for {analysis}")
        coords = ANALYSIS_COORDS[analysis]
        point_coord_flags = []
        interval_coord_flags = []
        for coord in coords:
            point = (
                merged["A_log_ratio_positive"].to_numpy(bool)
                & (np.abs(merged[f"dp_nu_d_{coord}"].to_numpy(float)) > TOL)
                & merged[f"fisher_diag_gt_zero_{coord}"].to_numpy(bool)
            )
            interval = (
                merged["A_log_ratio_positive"].to_numpy(bool)
                & merged[f"paired_interval_excludes_zero_{coord}"].to_numpy(bool)
                & merged[f"fisher_diag_gt_zero_{coord}"].to_numpy(bool)
            )
            merged[f"ABC_permitted_direction_point_{coord}"] = point
            merged[f"ABC_permitted_direction_interval_{coord}"] = interval
            point_coord_flags.append(point)
            interval_coord_flags.append(interval)
        merged["ABC_permitted_direction_point_any_coordinate"] = np.column_stack(point_coord_flags).any(axis=1)
        merged["ABC_permitted_direction_interval_any_coordinate"] = np.column_stack(interval_coord_flags).any(axis=1)
        rows_by_context.append(merged)

        for seed_label, group in [("all", merged), *[(int(s), g) for s, g in merged.groupby("seed")]]:
            n = len(group)
            A = group["A_log_ratio_positive"].to_numpy(bool)
            point_any = group["ABC_permitted_direction_point_any_coordinate"].to_numpy(bool)
            interval_any = group["ABC_permitted_direction_interval_any_coordinate"].to_numpy(bool)
            intervention_point = group["any_tested_direction_increases_p_nu"].to_numpy(bool)
            intervention_interval = group["any_interval_supported_direction"].to_numpy(bool)
            summary_rows.extend(
                [
                    {
                        "analysis": analysis,
                        "seed": seed_label,
                        "classification": "all_three_same_context_any_coordinate_point_estimate",
                        "n_contexts": int(n),
                        "n_satisfying": int(np.sum(point_any)),
                        "percentage": float(100.0 * np.mean(point_any)),
                    },
                    {
                        "analysis": analysis,
                        "seed": seed_label,
                        "classification": "all_three_same_context_any_coordinate_interval_supported",
                        "n_contexts": int(n),
                        "n_satisfying": int(np.sum(interval_any)),
                        "percentage": float(100.0 * np.mean(interval_any)),
                    },
                    {
                        "analysis": analysis,
                        "seed": seed_label,
                        "classification": "positive_focal_nu_log_ratio_A",
                        "n_contexts": int(n),
                        "n_satisfying": int(np.sum(A)),
                        "percentage": float(100.0 * np.mean(A)),
                    },
                    {
                        "analysis": analysis,
                        "seed": seed_label,
                        "classification": "A_only_no_same_coordinate_point_B_and_C",
                        "n_contexts": int(n),
                        "n_satisfying": int(np.sum(A & ~point_any)),
                        "percentage": float(100.0 * np.mean(A & ~point_any)),
                    },
                    {
                        "analysis": analysis,
                        "seed": seed_label,
                        "classification": "intervention_condition_point_but_not_A",
                        "n_contexts": int(n),
                        "n_satisfying": int(np.sum(intervention_point & ~A)),
                        "percentage": float(100.0 * np.mean(intervention_point & ~A)),
                    },
                    {
                        "analysis": analysis,
                        "seed": seed_label,
                        "classification": "A_but_no_point_derivative_in_any_permitted_direction",
                        "n_contexts": int(n),
                        "n_satisfying": int(np.sum(A & ~intervention_point)),
                        "percentage": float(100.0 * np.mean(A & ~intervention_point)),
                    },
                    {
                        "analysis": analysis,
                        "seed": seed_label,
                        "classification": "intervention_condition_interval_but_not_A",
                        "n_contexts": int(n),
                        "n_satisfying": int(np.sum(intervention_interval & ~A)),
                        "percentage": float(100.0 * np.mean(intervention_interval & ~A)),
                    },
                    {
                        "analysis": analysis,
                        "seed": seed_label,
                        "classification": "A_but_no_interval_supported_derivative_in_any_permitted_direction",
                        "n_contexts": int(n),
                        "n_satisfying": int(np.sum(A & ~intervention_interval)),
                        "percentage": float(100.0 * np.mean(A & ~intervention_interval)),
                    },
                ]
            )
    by_context = pd.concat(rows_by_context, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    by_context.to_csv(output_dir / "context_specific_qualification_M1000_by_context.csv", index=False)
    summary.to_csv(output_dir / "context_specific_qualification_M1000_summary.csv", index=False)
    return by_context, summary


def old_new_comparisons(
    all_tables: Mapping[str, pd.DataFrame],
    qualification_summary: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    rows = []
    old_grad = pd.read_csv(ROOT / "outputs" / "main39_code_audit" / "local_derivative_p_nu_gradient_summary.csv")
    old_fisher = pd.read_csv(ROOT / "outputs" / "main39_code_audit" / "fisher_context_diagnostics.csv")
    old_qual = pd.read_csv(ROOT / "outputs" / "main39_code_audit" / "context_specific_qualification_rate_summary.csv")

    grad_summary = all_tables["gradient_summary"]
    for analysis in ("epigenetic", "joint"):
        prior_analysis = "joint_epigenetic_ecological" if analysis == "joint" else analysis
        for coord in ANALYSIS_COORDS[analysis]:
            new = grad_summary[
                (grad_summary["analysis"] == analysis)
                & (grad_summary["M"] == 1000)
                & (grad_summary["seed"].astype(str) == "all")
                & (grad_summary["coordinate"] == coord)
            ].iloc[0]
            old_subset = old_grad[(old_grad["analysis"] == prior_analysis) & (old_grad["coordinate"] == coord)]
            if not old_subset.empty:
                old = old_subset.iloc[0]
                rows.append(
                    {
                        "analysis": analysis,
                        "quantity": f"context_averaged_gradient_{coord}",
                        "old_preliminary_M100": float(old["mean"]),
                        "new_M1000": float(new["mean"]),
                        "note": "new value uses nested common random numbers and 1000 Monte Carlo draws per context",
                    }
                )
    eig = all_tables["fisher_eigenvalues_by_M"]
    for analysis in ("epigenetic", "joint"):
        prior_analysis = "joint_epigenetic_ecological" if analysis == "joint" else analysis
        new_min = eig[(eig["analysis"] == analysis) & (eig["M"] == 1000)]["minimum_eigenvalue"].iloc[0]
        old_min = old_fisher[old_fisher["analysis"] == prior_analysis]
        if not old_min.empty and "avg_matrix_eigenvalues" in old_min:
            old_eigs = [float(x) for x in str(old_min.iloc[0]["avg_matrix_eigenvalues"]).split(";")]
            rows.append(
                {
                    "analysis": analysis,
                    "quantity": "averaged_fisher_min_eigenvalue",
                    "old_preliminary_M100": float(min(old_eigs)),
                    "new_M1000": float(new_min),
                    "note": "Fisher matrix recomputed from M=1000 probability evaluations",
                }
            )
        new_rank = eig[(eig["analysis"] == analysis) & (eig["M"] == 1000)]["rank_tol_1e_minus_10"].iloc[0]
        old_rank = old_fisher[old_fisher["analysis"] == prior_analysis]
        if not old_rank.empty and "avg_matrix_rank_tol_1e_minus_10" in old_rank:
            rows.append(
                {
                    "analysis": analysis,
                    "quantity": "averaged_fisher_rank",
                    "old_preliminary_M100": float(old_rank.iloc[0]["avg_matrix_rank_tol_1e_minus_10"]),
                    "new_M1000": float(new_rank),
                    "note": "rank computed with tolerance 1e-10",
                }
            )
        qnew_point = qualification_summary[
            (qualification_summary["analysis"] == analysis)
            & (qualification_summary["seed"].astype(str) == "all")
            & (qualification_summary["classification"] == "all_three_same_context_any_coordinate_point_estimate")
        ].iloc[0]
        qnew_interval = qualification_summary[
            (qualification_summary["analysis"] == analysis)
            & (qualification_summary["seed"].astype(str) == "all")
            & (qualification_summary["classification"] == "all_three_same_context_any_coordinate_interval_supported")
        ].iloc[0]
        qold = old_qual[
            (old_qual["analysis"] == prior_analysis)
            & (old_qual["seed"].astype(str) == "all")
            & (old_qual["category"] == "all_three_same_context_any_coordinate")
        ]
        if not qold.empty:
            rows.append(
                {
                    "analysis": analysis,
                    "quantity": "same_context_qualification_rate_point_estimate",
                    "old_preliminary_M100": float(qold.iloc[0]["percentage"]),
                    "new_M1000": float(qnew_point["percentage"]),
                    "note": "new point-estimate rate treats positive and negative coordinate changes as permitted directions",
                }
            )
            rows.append(
                {
                    "analysis": analysis,
                    "quantity": "same_context_qualification_rate_interval_supported",
                    "old_preliminary_M100": float(qold.iloc[0]["percentage"]),
                    "new_M1000": float(qnew_interval["percentage"]),
                    "note": "requires paired 95% derivative interval to exclude zero in an increasing direction",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "old_vs_new_robustness_comparison.csv", index=False)
    return out


def write_outputs(summaries: Mapping[str, dict[str, pd.DataFrame]], output_dir: Path) -> None:
    combined: dict[str, list[pd.DataFrame]] = {}
    for tables in summaries.values():
        for key, df in tables.items():
            if isinstance(df, pd.DataFrame):
                combined.setdefault(key, []).append(df)
    for key, dfs in combined.items():
        pd.concat(dfs, ignore_index=True).to_csv(output_dir / f"{key}.csv", index=False)


def main() -> None:
    config = final_config()
    output_dir = PRIMARY_ROOT / "intervention_fisher_mc_robustness"
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = PRIMARY_ROOT / "source_data"
    contexts = pd.read_csv(
        PRIMARY_ROOT
        / "corrected_numerical_audit"
        / "data"
        / "intervention_generation_stratified_contexts.csv"
    )
    if len(contexts) != config.n_contexts:
        raise AssertionError(f"expected {config.n_contexts} contexts, found {len(contexts)}")
    seed_results = []
    for seed in config.seeds:
        data = load_saved_seed(int(seed), config, source_dir)
        if data is None:
            raise FileNotFoundError(f"missing saved final trajectory archive for seed {seed}")
        seed_results.append(data)
    metadata = {
        "seeds": list(config.seeds),
        "M_values": list(M_VALUES),
        "max_M": MAX_M,
        "n_contexts": int(len(contexts)),
        "finite_difference_step": float(config.fisher_step),
        "common_random_numbers": "theta values share the same random streams within each seed-by-generation batch; M prefixes are nested",
        "probability_smoothing_alpha": float(config.intervention_probability_smoothing_alpha),
        "point_estimate_tolerance": TOL,
        "rank_tolerance": RANK_TOL,
        "candidate": CANDIDATE_NAME,
        "contexts_file": str(
            PRIMARY_ROOT
            / "corrected_numerical_audit"
            / "data"
            / "intervention_generation_stratified_contexts.csv"
        ),
    }
    (output_dir / "robustness_metadata.json").write_text(json.dumps(metadata, indent=2))

    sim_summaries: dict[str, dict[str, pd.DataFrame]] = {}
    qualification_inputs: dict[str, dict[str, object]] = {}
    for analysis in ("epigenetic", "joint"):
        print(f"simulating {analysis} nested Monte Carlo intervention probabilities", flush=True)
        sim = simulate_analysis(analysis, seed_results, contexts, config, output_dir)
        tables = summarize_analysis(sim, config)
        qualification_inputs[analysis] = {"qualification_inputs": tables.pop("qualification_inputs")}
        sim_summaries[analysis] = tables
    write_outputs(sim_summaries, output_dir)
    q_inputs = {analysis: qualification_inputs[analysis] for analysis in ("epigenetic", "joint")}
    _, qual_summary = qualification_tables(q_inputs, output_dir)
    combined_tables = {
        key: pd.concat([tables[key] for tables in sim_summaries.values()], ignore_index=True)
        for key in sim_summaries["epigenetic"].keys()
        if key in sim_summaries["joint"]
    }
    combined_tables["gradient_summary"] = pd.read_csv(output_dir / "gradient_summary.csv")
    combined_tables["fisher_eigenvalues_by_M"] = pd.read_csv(output_dir / "fisher_eigenvalues_by_M.csv")
    if os.environ.get("ROBUSTNESS_SKIP_OLD_COMPARISON", "0") != "1":
        old_new_comparisons(combined_tables, qual_summary, output_dir)
    print(f"wrote robustness outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
