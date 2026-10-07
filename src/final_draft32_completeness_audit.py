from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import json
import platform
import shutil

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .corrected_numerical_audit import (
    CorrectedAuditConfig,
    _CommonRandomThetaBatch,
    _as2d,
    _smooth_probabilities,
    gc_rank,
    generation_design,
)
from .corrected_pilot_model import PilotConfig, _params, build_event_schedule, level_timestamps, reproductive_state
from .final_numerical_audit import (
    BACKGROUND_LEVEL,
    ELC_LEVELS,
    PHENOTYPE_LABELS,
    _integrate_generation_vec,
    _next_generation_start_vec,
    build_analysis_table,
    load_saved_seed,
)
from .information_measures import summarize_interval, unit_index_groups


JOINT_THETA_COMPONENTS = (
    "theta_reg",
    "theta_stress",
    "theta_soil",
    "theta_resource",
    "theta_microclimate",
)
NU_LABEL = "nu_early_maturation_high_growth"
Z_CRIT = 1.959963984540054


def _label_codes(labels: np.ndarray) -> np.ndarray:
    lookup = {label: i for i, label in enumerate(PHENOTYPE_LABELS)}
    return np.asarray([lookup[str(label)] for label in labels], dtype=int)


def _joint_source(table: dict[str, object]) -> np.ndarray:
    return np.hstack([_as2d(table["source_epigenetic"]), _as2d(table["source_ecological"])])


def _joint_phenotype_design(table: dict[str, object], *, include_source: bool) -> np.ndarray:
    z = gc_rank(_as2d(table["history_remainder_without_epigenetic_ecological"]))
    gen = generation_design(table["meta"])
    parts = [z]
    if include_source:
        parts.append(gc_rank(_joint_source(table)))
    if gen.size:
        parts.append(gen)
    return np.hstack(parts)


def run_joint_categorical_phenotype_information(
    table: dict[str, object],
    config: CorrectedAuditConfig,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = np.asarray(table["variant_label"], dtype=object)
    y = _label_codes(labels)
    x_null = _joint_phenotype_design(table, include_source=False)
    x_full = _joint_phenotype_design(table, include_source=True)
    groups = table["meta"]["unit_id"].to_numpy(dtype=int)

    p_null = np.zeros((len(y), len(PHENOTYPE_LABELS)), dtype=float)
    p_full = np.zeros_like(p_null)
    fold_rows = []
    for fold, (train, test) in enumerate(GroupKFold(n_splits=config.categorical_folds).split(x_full, y, groups=groups)):
        null_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=config.categorical_logistic_c,
                penalty="l2",
                solver="lbfgs",
                max_iter=900,
            ),
        )
        full_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=config.categorical_logistic_c,
                penalty="l2",
                solver="lbfgs",
                max_iter=900,
            ),
        )
        null_model.fit(x_null[train], y[train])
        full_model.fit(x_full[train], y[train])
        p_null[test] = null_model.predict_proba(x_null[test])
        p_full[test] = full_model.predict_proba(x_full[test])
        fold_rows.append(
            {
                "fold": int(fold),
                "n_train": int(train.size),
                "n_test": int(test.size),
                "lineage_grouping": "all generations from a lineage remain in one fold",
            }
        )

    eps = 1e-12
    p_null = np.clip(p_null, eps, 1.0)
    p_full = np.clip(p_full, eps, 1.0)
    p_null = p_null / p_null.sum(axis=1, keepdims=True)
    p_full = p_full / p_full.sum(axis=1, keepdims=True)

    idx = np.arange(len(y))
    state_log_ratio = np.log2(p_full[idx, y] / p_null[idx, y])
    complete_te = float(np.mean(state_log_ratio))

    nu_index = PHENOTYPE_LABELS.index(NU_LABEL)
    event = y == nu_index
    p_full_event = np.where(event, p_full[:, nu_index], 1.0 - p_full[:, nu_index])
    p_null_event = np.where(event, p_null[:, nu_index], 1.0 - p_null[:, nu_index])
    binary_log_ratio = np.log2(np.clip(p_full_event, eps, 1.0) / np.clip(p_null_event, eps, 1.0))
    binary_te = float(np.mean(binary_log_ratio))
    focal_log_ratio_all_contexts = np.log2(p_full[:, nu_index] / p_null[:, nu_index])
    focal_log_ratio_nu = float(np.mean(focal_log_ratio_all_contexts[event]))

    decomp_rows = []
    for label_index, label in enumerate(PHENOTYPE_LABELS):
        mask = y == label_index
        mean_lr = float(np.mean(state_log_ratio[mask])) if np.any(mask) else np.nan
        contribution = float(mean_lr * np.mean(mask)) if np.any(mask) else 0.0
        decomp_rows.append(
            {
                "phenotype_state": label,
                "proportion": float(np.mean(mask)),
                "mean_state_specific_log_ratio_bits": mean_lr,
                "weighted_contribution_bits": contribution,
            }
        )
    decomp = pd.DataFrame(decomp_rows)
    decomp.to_csv(output_dir / "joint_categorical_phenotype_state_specific_decomposition.csv", index=False)

    meta = table["meta"].reset_index(drop=True)
    score_df = meta[["seed", "lineage_id", "unit_id", "tau"]].copy()
    score_df["phenotype_state"] = labels
    score_df["complete_state_log_ratio_bits"] = state_log_ratio
    score_df["binary_nu_log_ratio_bits"] = binary_log_ratio
    score_df["focal_nu_probability_log_ratio_bits"] = focal_log_ratio_all_contexts
    for label_index, label in enumerate(PHENOTYPE_LABELS):
        score_df[f"p_full_{label}"] = p_full[:, label_index]
        score_df[f"p_null_{label}"] = p_null[:, label_index]
    score_df.to_csv(output_dir / "joint_categorical_phenotype_crossfit_scores.csv", index=False)

    groups_index = unit_index_groups(meta)
    unit_ids = np.array(list(groups_index.keys()), dtype=int)
    rng = np.random.default_rng(config.bootstrap_seed + 166000)
    boot_rows = []
    for b in range(config.n_bootstrap):
        sampled = rng.choice(unit_ids, size=len(unit_ids), replace=True)
        boot_idx = np.concatenate([groups_index[int(uid)] for uid in sampled])
        nu_mask = event[boot_idx]
        boot_rows.append(
            {
                "bootstrap": int(b),
                "complete_phenotype_transfer_entropy_bits": float(np.mean(state_log_ratio[boot_idx])),
                "binary_nu_transfer_entropy_bits": float(np.mean(binary_log_ratio[boot_idx])),
                "focal_nu_log_ratio_given_nu_bits": float(np.mean(focal_log_ratio_all_contexts[boot_idx][nu_mask])),
            }
        )
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(output_dir / "joint_categorical_phenotype_bootstrap.csv", index=False)

    seed_rows = []
    for seed, seed_idx in meta.groupby("seed").groups.items():
        ii = np.asarray(list(seed_idx), dtype=int)
        seed_rows.append(
            {
                "seed": int(seed),
                "complete_phenotype_transfer_entropy_bits": float(np.mean(state_log_ratio[ii])),
                "binary_nu_transfer_entropy_bits": float(np.mean(binary_log_ratio[ii])),
                "focal_nu_log_ratio_given_nu_bits": float(np.mean(focal_log_ratio_all_contexts[ii][event[ii]])),
            }
        )
    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(output_dir / "joint_categorical_phenotype_by_seed.csv", index=False)

    rows = []
    for quantity, observed, col in [
        ("complete_four_state_joint_source_information", complete_te, "complete_phenotype_transfer_entropy_bits"),
        ("binary_nu_joint_source_information", binary_te, "binary_nu_transfer_entropy_bits"),
        ("focal_nu_conditional_log_ratio_given_nu", focal_log_ratio_nu, "focal_nu_log_ratio_given_nu_bits"),
    ]:
        vals = boot[col].to_numpy(dtype=float)
        se = float(np.std(vals, ddof=1))
        rows.append(
            {
                "quantity": quantity,
                "estimate_bits": observed,
                "bootstrap_mean_bits": float(np.mean(vals)),
                "bootstrap_se_bits": se,
                "ci_method": "lineage-cluster bootstrap over lineage-level cross-fitted log-probability ratios",
                "ci_lower_bits": observed - Z_CRIT * se,
                "ci_upper_bits": observed + Z_CRIT * se,
            }
        )
    summary = pd.DataFrame(rows)
    summary["complete_ge_binary_nu_check"] = bool(complete_te + 1e-10 >= binary_te)
    summary["weighted_decomposition_sum_bits"] = float(decomp["weighted_contribution_bits"].sum())
    summary["weighted_decomposition_error_bits"] = abs(float(decomp["weighted_contribution_bits"].sum()) - complete_te)
    summary["target"] = "V_{d,tau+1}, four mutually exclusive phenotype states"
    summary["source"] = "complete retained epigenetic and ecological segments at tau"
    summary["conditioning_set"] = "X^{ELC\\{epigenetic,ecological},(k=2)}_{d,tau} plus generation effects"
    summary["estimator"] = "lineage-level cross-fitted multinomial logistic model"
    summary.to_csv(output_dir / "joint_categorical_phenotype_information_summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "joint_categorical_phenotype_crossfit_folds.csv", index=False)
    return summary, boot, seed_df, decomp


def _phenotype_labels_from_life(life: np.ndarray, config: CorrectedAuditConfig) -> np.ndarray:
    maturation = life[:, :, 0]
    crosses = maturation >= config.theta_maturation
    crossing = np.full(maturation.shape[0], -1, dtype=int)
    for t_l in range(maturation.shape[1]):
        crossing[(crossing < 0) & crosses[:, t_l]] = t_l
    timestamps = level_timestamps(config.pilot_config(config.seeds[0]))["life_history"]
    crossing_u = np.full(crossing.shape, np.nan, dtype=float)
    valid = crossing >= 0
    crossing_u[valid] = timestamps[crossing[valid]]
    early = valid & (crossing_u < config.pilot_config(config.seeds[0]).early_maturation_u)
    high = life[:, -1, 1] >= config.theta_growth
    labels = np.empty(maturation.shape[0], dtype=object)
    labels[early & high] = "nu_early_maturation_high_growth"
    labels[early & ~high] = "early_maturation_low_growth"
    labels[~early & high] = "non_early_maturation_high_growth"
    labels[~early & ~high] = "non_early_maturation_low_growth"
    return labels


def _joint_theta_specification(config: CorrectedAuditConfig) -> pd.DataFrame:
    h = config.fisher_step
    specs: list[tuple[str, tuple[float, float, float, float, float], str]] = [
        ("theta_joint_minus_all", (-0.5, -0.5, -0.5, -0.5, -0.5), "minimal_response"),
        ("theta_joint_zero", (0.0, 0.0, 0.0, 0.0, 0.0), "minimal_response_and_fisher_base"),
        ("theta_joint_plus_all", (0.5, 0.5, 0.5, 0.5, 0.5), "minimal_response"),
    ]
    for j, component in enumerate(JOINT_THETA_COMPONENTS):
        plus = np.zeros(len(JOINT_THETA_COMPONENTS), dtype=float)
        minus = np.zeros(len(JOINT_THETA_COMPONENTS), dtype=float)
        plus[j] = h
        minus[j] = -h
        specs.append((f"theta_joint_plus_{component}", tuple(plus.tolist()), "fisher_finite_difference"))
        specs.append((f"theta_joint_minus_{component}", tuple(minus.tolist()), "fisher_finite_difference"))
    rows = []
    for label, vector, role in specs:
        row = {"theta_label": label, "role": role}
        row.update({component: float(vector[i]) for i, component in enumerate(JOINT_THETA_COMPONENTS)})
        rows.append(row)
    return pd.DataFrame(rows)


def _prepare_contexts(corrected_archive: Path, config: CorrectedAuditConfig, output_dir: Path) -> pd.DataFrame:
    context_path = corrected_archive / "intervention_generation_stratified_contexts.csv"
    if not context_path.exists():
        raise FileNotFoundError(f"missing approved generation-stratified contexts: {context_path}")
    contexts = pd.read_csv(context_path)
    required = {"seed", "lineage_id", "tau", "context_id"}
    missing = required - set(contexts.columns)
    if missing:
        raise ValueError(f"approved context table is missing columns: {sorted(missing)}")
    if len(contexts) != config.n_contexts:
        raise AssertionError(f"expected {config.n_contexts} contexts, found {len(contexts)}")
    per_seed = contexts.groupby("seed").size()
    if not np.all(per_seed.to_numpy(dtype=int) == config.n_contexts_per_seed):
        raise AssertionError("approved context table is not balanced at 250 contexts per seed")
    contexts.to_csv(output_dir / "joint_intervention_contexts.csv", index=False)
    return contexts


def simulate_joint_intervention_probabilities(
    seed_results: list[dict[str, object]],
    config: CorrectedAuditConfig,
    corrected_archive: Path,
    output_dir: Path,
) -> pd.DataFrame:
    contexts = _prepare_contexts(corrected_archive, config, output_dir)
    theta_df = _joint_theta_specification(config)
    theta_df.to_csv(output_dir / "joint_intervention_theta_values.csv", index=False)

    seed_map = {int(sd["seed"]): sd for sd in seed_results}
    schedule_cache: dict[int, list[tuple[float, tuple[str, ...]]]] = {}
    rows = []
    for (seed, tau), group in contexts.groupby(["seed", "tau"], sort=True):
        seed_data = seed_map[int(seed)]
        lineages = group["lineage_id"].to_numpy(dtype=int)
        context_ids = group["context_id"].to_numpy(dtype=int)
        repeated_lineages = np.repeat(lineages, config.n_intervention_draws)
        repeated_context_ids = np.repeat(context_ids, config.n_intervention_draws)
        base_n = repeated_lineages.size
        n_theta = len(theta_df)
        expanded_n = base_n * n_theta
        expanded_config: PilotConfig = config.pilot_config(int(seed), n_lineages=expanded_n)
        if expanded_n not in schedule_cache:
            schedule_cache[expanded_n] = build_event_schedule(expanded_config)
        start = {
            level: np.tile(np.asarray(seed_data["full_time_series"][level])[repeated_lineages, int(tau), 0].copy(), (n_theta, 1))
            for level in ELC_LEVELS + (BACKGROUND_LEVEL,)
        }
        theta_mat = theta_df[list(JOINT_THETA_COMPONENTS)].to_numpy(dtype=float)
        theta_by_row = np.repeat(theta_mat, base_n, axis=0)
        start["epigenetic"][:, :2] += theta_by_row[:, :2]
        start["ecological"][:, :3] += theta_by_row[:, 2:5]
        expanded_lineages = np.tile(repeated_lineages, n_theta)
        rng = _CommonRandomThetaBatch(
            config.intervention_seed + 700_000 + int(seed) * 100_000 + int(tau) * 1009,
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
        source_reproductive = {
            level: reproductive_state(source_retained[level], level, expanded_config)
            for level in ELC_LEVELS + (BACKGROUND_LEVEL,)
        }
        future_start = _next_generation_start_vec(source_reproductive, rng, expanded_config, seed_data["parameters"])
        future_retained = _integrate_generation_vec(
            future_start,
            int(tau) + 1,
            rng,
            expanded_config,
            seed_data["parameters"],
            schedule_cache[expanded_n],
            d_idx=expanded_lineages,
        )
        life_all = future_retained["life_history"].reshape(n_theta, base_n, future_retained["life_history"].shape[1], future_retained["life_history"].shape[2])
        for theta_i, theta_row in theta_df.reset_index(drop=True).iterrows():
            labels = _phenotype_labels_from_life(life_all[int(theta_i)], config)
            tmp = pd.DataFrame({"context_id": repeated_context_ids, "phenotype_state": labels})
            counts = (
                tmp.groupby(["context_id", "phenotype_state"])
                .size()
                .unstack(fill_value=0)
                .reindex(columns=PHENOTYPE_LABELS, fill_value=0)
            )
            counts = counts.reindex(context_ids, fill_value=0)
            probs = counts.div(float(config.n_intervention_draws))
            for context_id, lineage, prob_row in zip(context_ids, lineages, probs.to_dict("records")):
                out = {
                    "seed": int(seed),
                    "lineage_id": int(lineage),
                    "tau": int(tau),
                    "context_id": int(context_id),
                    "theta_label": str(theta_row["theta_label"]),
                    "theta_role": str(theta_row["role"]),
                }
                out.update({component: float(theta_row[component]) for component in JOINT_THETA_COMPONENTS})
                out.update({f"p_{label}": float(prob_row[label]) for label in PHENOTYPE_LABELS})
                rows.append(out)
    prob_by_context = pd.DataFrame(rows)
    prob_by_context.to_csv(output_dir / "joint_intervention_probability_by_context.csv", index=False)
    return prob_by_context


def summarize_joint_intervention(
    prob_by_context: pd.DataFrame,
    config: CorrectedAuditConfig,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prob_cols = [f"p_{label}" for label in PHENOTYPE_LABELS]
    response = prob_by_context[prob_by_context["theta_label"].isin(["theta_joint_minus_all", "theta_joint_zero", "theta_joint_plus_all"])].copy()
    summary = response.groupby(["theta_label", *JOINT_THETA_COMPONENTS])[prob_cols].mean().reset_index()
    summary.to_csv(output_dir / "joint_intervention_response_summary.csv", index=False)

    wide = response.pivot(index=["seed", "lineage_id", "tau", "context_id"], columns="theta_label", values=f"p_{NU_LABEL}").reset_index()
    wide["delta_p_nu_plus"] = wide["theta_joint_plus_all"] - wide["theta_joint_zero"]
    wide["delta_p_nu_minus"] = wide["theta_joint_minus_all"] - wide["theta_joint_zero"]
    wide.to_csv(output_dir / "joint_intervention_delta_by_context.csv", index=False)

    wide["cluster_id"] = wide["seed"].astype(str) + "_" + wide["lineage_id"].astype(str)
    groups = {cid: group.index.to_numpy(dtype=int) for cid, group in wide.groupby("cluster_id")}
    cluster_ids = np.array(list(groups.keys()), dtype=object)
    rng = np.random.default_rng(config.bootstrap_seed + 177000)
    boot_rows = []
    for b in range(config.n_bootstrap):
        sampled = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        idx = np.concatenate([groups[cid] for cid in sampled])
        boot_rows.append(
            {
                "bootstrap": int(b),
                "delta_p_nu_plus": float(wide.loc[idx, "delta_p_nu_plus"].mean()),
                "delta_p_nu_minus": float(wide.loc[idx, "delta_p_nu_minus"].mean()),
            }
        )
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(output_dir / "joint_intervention_delta_bootstrap.csv", index=False)

    rows = []
    for quantity in ["delta_p_nu_plus", "delta_p_nu_minus"]:
        vals = boot[quantity].to_numpy(dtype=float)
        interval = summarize_interval(vals)
        observed = float(wide[quantity].mean())
        rows.append(
            {
                "quantity": quantity,
                "estimate_probability_difference": observed,
                "bootstrap_mean": interval["mean"],
                "ci_method": "lineage-cluster bootstrap over context-level probability differences",
                "ci_lower": interval["lower"],
                "ci_upper": interval["upper"],
            }
        )
    delta_summary = pd.DataFrame(rows)
    delta_summary.to_csv(output_dir / "joint_intervention_delta_summary.csv", index=False)

    by_generation = response.groupby(["tau", "theta_label"])[prob_cols].mean().reset_index()
    by_generation.to_csv(output_dir / "joint_intervention_response_by_generation.csv", index=False)
    delta_generation = wide.groupby("tau")[["delta_p_nu_plus", "delta_p_nu_minus"]].mean().reset_index()
    delta_generation.to_csv(output_dir / "joint_intervention_delta_by_generation.csv", index=False)
    return summary, delta_summary, boot


def _theta_probability_matrix(prob_by_context: pd.DataFrame, theta_label: str, config: CorrectedAuditConfig) -> tuple[np.ndarray, pd.DataFrame]:
    prob_cols = [f"p_{label}" for label in PHENOTYPE_LABELS]
    table = prob_by_context[prob_by_context["theta_label"] == theta_label].copy()
    meta = table[["context_id", "seed", "lineage_id", "tau"]].sort_values("context_id").reset_index(drop=True)
    probs = table.sort_values("context_id")[prob_cols].to_numpy(dtype=float)
    return _smooth_probabilities(probs, config.n_intervention_draws, config.intervention_probability_smoothing_alpha), meta


def compute_joint_fisher_information(
    prob_by_context: pd.DataFrame,
    config: CorrectedAuditConfig,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    h = config.fisher_step
    p0, meta = _theta_probability_matrix(prob_by_context, "theta_joint_zero", config)
    derivs = []
    for component in JOINT_THETA_COMPONENTS:
        p_plus, _ = _theta_probability_matrix(prob_by_context, f"theta_joint_plus_{component}", config)
        p_minus, _ = _theta_probability_matrix(prob_by_context, f"theta_joint_minus_{component}", config)
        derivs.append((p_plus - p_minus) / (2.0 * h))
    grad = np.stack(derivs, axis=1)
    context_mats = np.einsum("cav,cbv,cv->cab", grad, grad, 1.0 / p0)
    context_mats = 0.5 * (context_mats + np.swapaxes(context_mats, 1, 2))

    rows = []
    for cidx, context_id in enumerate(meta["context_id"].to_numpy(dtype=int)):
        for i, row_component in enumerate(JOINT_THETA_COMPONENTS):
            for j, col_component in enumerate(JOINT_THETA_COMPONENTS):
                rows.append(
                    {
                        "context_id": int(context_id),
                        "seed": int(meta.loc[cidx, "seed"]),
                        "lineage_id": int(meta.loc[cidx, "lineage_id"]),
                        "tau": int(meta.loc[cidx, "tau"]),
                        "row_component": row_component,
                        "col_component": col_component,
                        "fisher_information": float(context_mats[cidx, i, j]),
                    }
                )
    context_df = pd.DataFrame(rows)
    context_df.to_csv(output_dir / "joint_fisher_information_by_context.csv", index=False)

    fmat = 0.5 * (context_mats.mean(axis=0) + context_mats.mean(axis=0).T)
    eig = np.linalg.eigvalsh(fmat)
    observed_rows = []
    for i, row_component in enumerate(JOINT_THETA_COMPONENTS):
        for j, col_component in enumerate(JOINT_THETA_COMPONENTS):
            observed_rows.append(
                {
                    "row_component": row_component,
                    "col_component": col_component,
                    "fisher_information": float(fmat[i, j]),
                    "n_contexts": int(context_mats.shape[0]),
                    "finite_difference_step": h,
                    "eigenvalue_min": float(eig[0]),
                    "eigenvalue_max": float(eig[-1]),
                    "symmetric": bool(np.allclose(fmat, fmat.T, atol=1e-12)),
                    "positive_semidefinite": bool(eig[0] >= -1e-10),
                }
            )
    observed = pd.DataFrame(observed_rows)

    meta2 = meta.copy()
    meta2["cluster_id"] = meta2["seed"].astype(str) + "_" + meta2["lineage_id"].astype(str)
    groups = meta2.groupby("cluster_id").indices
    cluster_ids = np.array(list(groups.keys()), dtype=object)
    rng = np.random.default_rng(config.bootstrap_seed + 188000)
    boot_rows = []
    for b in range(config.n_bootstrap):
        sampled = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        idx = np.concatenate([np.asarray(groups[cid], dtype=int) for cid in sampled])
        bmat = 0.5 * (context_mats[idx].mean(axis=0) + context_mats[idx].mean(axis=0).T)
        beigs = np.linalg.eigvalsh(bmat)
        for i, row_component in enumerate(JOINT_THETA_COMPONENTS):
            for j, col_component in enumerate(JOINT_THETA_COMPONENTS):
                boot_rows.append(
                    {
                        "bootstrap": int(b),
                        "row_component": row_component,
                        "col_component": col_component,
                        "fisher_information": float(bmat[i, j]),
                        "eigenvalue_min": float(beigs[0]),
                        "eigenvalue_max": float(beigs[-1]),
                    }
                )
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(output_dir / "joint_fisher_information_bootstrap.csv", index=False)

    interval_rows = []
    for key, group in boot.groupby(["row_component", "col_component"]):
        vals = group["fisher_information"].to_numpy(dtype=float)
        interval = summarize_interval(vals)
        interval_rows.append(
            {
                "row_component": key[0],
                "col_component": key[1],
                "bootstrap_mean": interval["mean"],
                "ci_lower": interval["lower"],
                "ci_upper": interval["upper"],
            }
        )
    observed = observed.merge(pd.DataFrame(interval_rows), on=["row_component", "col_component"], how="left")
    observed.to_csv(output_dir / "joint_fisher_information_matrix.csv", index=False)

    eig_df = pd.DataFrame(
        {
            "eigenvalue_index": np.arange(len(eig), dtype=int),
            "eigenvalue": eig,
            "positive_semidefinite": eig >= -1e-10,
        }
    )
    eig_df.to_csv(output_dir / "joint_fisher_information_eigenvalues.csv", index=False)

    diag_rows = []
    for component in JOINT_THETA_COMPONENTS:
        row = observed[(observed["row_component"] == component) & (observed["col_component"] == component)].iloc[0]
        diag_rows.append(
            {
                "block": "epigenetic" if component in {"theta_reg", "theta_stress"} else "ecological",
                "component": component,
                "fisher_information": float(row["fisher_information"]),
                "ci_lower": float(row["ci_lower"]),
                "ci_upper": float(row["ci_upper"]),
            }
        )
    pd.DataFrame(diag_rows).to_csv(output_dir / "joint_fisher_diagonal_blocks.csv", index=False)
    return observed, eig_df, boot


def _analysis_row(df: pd.DataFrame, analysis: str) -> pd.Series:
    subset = df[df["analysis"] == analysis]
    if subset.empty:
        raise KeyError(analysis)
    return subset.iloc[0]


def _quantity_row(df: pd.DataFrame, quantity: str) -> pd.Series:
    subset = df[df["quantity"] == quantity]
    if subset.empty:
        raise KeyError(quantity)
    return subset.iloc[0]


def build_complex_unit_qualification_table(output_dir: Path, corrected_archive: Path) -> pd.DataFrame:
    continuous = pd.read_csv(corrected_archive / "coherent_continuous_transfer_entropy_summary.csv")
    pid = pd.read_csv(corrected_archive / "corrected_epigenetic_ecological_delta_g_pid_summary.csv").iloc[0]
    pid_intervals = pd.read_csv(corrected_archive / "corrected_epigenetic_ecological_delta_g_pid_centered_intervals.csv")
    joint_pheno = pd.read_csv(output_dir / "joint_categorical_phenotype_information_summary.csv")
    delta = pd.read_csv(output_dir / "joint_intervention_delta_summary.csv")
    fisher = pd.read_csv(output_dir / "joint_fisher_information_matrix.csv")

    joint_pc = _analysis_row(continuous, "joint_epigenetic_ecological_predictive_contribution")
    joint_closure = _analysis_row(continuous, "joint_epigenetic_ecological_predictive_closure")
    synergy_interval = _quantity_row(pid_intervals, "synergy_bits")
    focal = _quantity_row(joint_pheno, "focal_nu_conditional_log_ratio_given_nu")
    binary = _quantity_row(joint_pheno, "binary_nu_joint_source_information")
    plus_delta = _quantity_row(delta, "delta_p_nu_plus")

    fmat = fisher.pivot(index="row_component", columns="col_component", values="fisher_information").loc[list(JOINT_THETA_COMPONENTS), list(JOINT_THETA_COMPONENTS)].to_numpy(dtype=float)
    eig = np.linalg.eigvalsh(0.5 * (fmat + fmat.T))
    diag = np.diag(fmat)
    epi_diag_nonzero = bool(np.any(np.abs(diag[:2]) > 1e-10))
    eco_diag_nonzero = bool(np.any(np.abs(diag[2:]) > 1e-10))
    fisher_pass = bool(eig[0] >= -1e-10 and epi_diag_nonzero and eco_diag_nonzero)

    rows = [
        {
            "requirement": "Joint predictive contribution",
            "result": f"raw={joint_pc['raw_estimate_bits']:.6f} bits; null-excess={joint_pc['null_excess_bits']:.6f} bits; p={joint_pc['generation_preserving_surrogate_p_value']:.6f}",
            "pass_fail": "pass" if bool(joint_pc["null_excess_bits"] > 0.0 and joint_pc["generation_preserving_surrogate_p_value"] <= 0.05) else "fail",
            "source_file": str(corrected_archive / "coherent_continuous_transfer_entropy_summary.csv"),
        },
        {
            "requirement": "Joint predictive closure",
            "result": f"raw={joint_closure['raw_estimate_bits']:.6f} bits; null-excess={joint_closure['null_excess_bits']:.6f} bits; p={joint_closure['generation_preserving_surrogate_p_value']:.6f}",
            "pass_fail": "pass" if bool(joint_closure["null_excess_bits"] > 0.0 and joint_closure["generation_preserving_surrogate_p_value"] <= 0.05) else "fail",
            "source_file": str(corrected_archive / "coherent_continuous_transfer_entropy_summary.csv"),
        },
        {
            "requirement": "Positive PID synergy",
            "result": f"synergy={pid['synergy_bits']:.6f} bits; centered CI=[{synergy_interval['ci_lower']:.6f},{synergy_interval['ci_upper']:.6f}]",
            "pass_fail": "pass" if bool(pid["synergy_bits"] > 0.0 and synergy_interval["ci_lower"] > 0.0 and pid["within_tolerance"]) else "fail",
            "source_file": str(corrected_archive / "corrected_epigenetic_ecological_delta_g_pid_summary.csv"),
        },
        {
            "requirement": "Joint focal-variant predictive information",
            "result": f"focal log-ratio among nu={focal['estimate_bits']:.6f} bits; CI=[{focal['ci_lower_bits']:.6f},{focal['ci_upper_bits']:.6f}]; binary nu={binary['estimate_bits']:.6f} bits",
            "pass_fail": "pass" if bool(focal["estimate_bits"] > 0.0 and focal["ci_lower_bits"] > 0.0 and binary["estimate_bits"] > 0.0) else "fail",
            "source_file": str(output_dir / "joint_categorical_phenotype_information_summary.csv"),
        },
        {
            "requirement": "Joint intervention increases p(nu)",
            "result": f"Delta p_nu^+={plus_delta['estimate_probability_difference']:.6f}; CI=[{plus_delta['ci_lower']:.6f},{plus_delta['ci_upper']:.6f}]",
            "pass_fail": "pass" if bool(plus_delta["ci_lower"] > 0.0) else "fail",
            "source_file": str(output_dir / "joint_intervention_delta_summary.csv"),
        },
        {
            "requirement": "Joint causal specificity",
            "result": f"min eigenvalue={eig[0]:.6e}; epigenetic diagonal nonzero={epi_diag_nonzero}; ecological diagonal nonzero={eco_diag_nonzero}",
            "pass_fail": "pass" if fisher_pass else "fail",
            "source_file": str(output_dir / "joint_fisher_information_matrix.csv"),
        },
    ]
    table = pd.DataFrame(rows)
    table["overall_complex_unit_status"] = "fully_qualified_complex_unit" if bool((table["pass_fail"] == "pass").all()) else "candidate_complex_unit_of_inheritance"
    table.to_csv(output_dir / "complex_unit_qualification_table.csv", index=False)
    return table


def build_draft32_compliance_matrix(output_dir: Path, corrected_archive: Path) -> pd.DataFrame:
    continuous = pd.read_csv(corrected_archive / "coherent_continuous_transfer_entropy_summary.csv")
    categorical = pd.read_csv(corrected_archive / "categorical_phenotype_information_summary.csv")
    intervention = pd.read_csv(corrected_archive / "intervention_generation_stratified_response_surface_summary.csv")
    fisher = pd.read_csv(corrected_archive / "corrected_intervention_fisher_information.csv")
    stability = pd.read_csv(corrected_archive / "corrected_intergenerational_stability_summary.csv")
    location = pd.read_csv(corrected_archive / "corrected_level_time_specific_transfer_entropy_summary.csv")
    pid = pd.read_csv(corrected_archive / "corrected_epigenetic_ecological_delta_g_pid_summary.csv").iloc[0]
    mp = pd.read_csv(corrected_archive / "corrected_multiple_parent_transfer_entropy_summary.csv")
    complex_table = pd.read_csv(output_dir / "complex_unit_qualification_table.csv")

    epi_pc = _analysis_row(continuous, "epigenetic_predictive_contribution")
    epi_closure = _analysis_row(continuous, "epigenetic_predictive_closure")
    bg_pc = _analysis_row(continuous, "background_predictive_contribution")
    bg_closure = _analysis_row(continuous, "background_closure_control")
    cat_binary = _quantity_row(categorical, "binary_nu_transfer_entropy")
    cat_focal = _quantity_row(categorical, "focal_nu_log_ratio_given_nu")
    base_row = intervention[(intervention["theta_reg"] == 0.0) & (intervention["theta_stress"] == 0.0)].iloc[0]
    plus_row = intervention[(intervention["theta_reg"] == 1.5) & (intervention["theta_stress"] == 1.5)].iloc[0]
    delta_plus = float(plus_row[f"p_{NU_LABEL}"] - base_row[f"p_{NU_LABEL}"])
    fisher0 = fisher[(fisher["theta_reg"] == 0.0) & (fisher["theta_stress"] == 0.0)]
    fisher0_min_eig = float(fisher0["eigenvalue_min"].iloc[0])
    stability_min_excess = float(stability["null_excess_bits"].min())
    location_count = int(len(location))
    complex_status = str(complex_table["overall_complex_unit_status"].iloc[0])
    mp_joint = _analysis_row(mp, "multiple_parent_joint_factors")

    rows = [
        {
            "draft32_requirement": "hereditary-factor predictive contribution",
            "exact_estimand": "I(X^{ELC\\{[l_epi]}}_{d,tau+1}; x^{[l_epi]}_{d,s_l(tau)} | X^{ELC\\{[l_epi]},(k)}_{d,tau})",
            "target": "future ELC excluding epigenetic level",
            "source_or_intervention": "epigenetic retained segment",
            "conditioning_set": "order-k history of ELC excluding epigenetic level plus generation effects",
            "estimator": "coherent Gaussian-copula transfer entropy estimator",
            "numerical_result": f"raw={epi_pc['raw_estimate_bits']:.6f}; null-excess={epi_pc['null_excess_bits']:.6f}",
            "uncertainty": f"CI=[{epi_pc['ci_lower_bits']:.6f},{epi_pc['ci_upper_bits']:.6f}], p={epi_pc['generation_preserving_surrogate_p_value']:.6f}",
            "output_filename": str(corrected_archive / "coherent_continuous_transfer_entropy_summary.csv"),
            "pass_fail": "pass" if bool(epi_pc["null_excess_bits"] > 0.0 and epi_pc["generation_preserving_surrogate_p_value"] <= 0.05) else "fail",
        },
        {
            "draft32_requirement": "predictive closure",
            "exact_estimand": "I(x^{[l_epi]}_{d,s_l(tau+1)}; X^{ELC\\{[l_epi]},(k)}_{d,tau} | x^{[l_epi]}_{d,s_l(tau)})",
            "target": "future epigenetic retained segment",
            "source_or_intervention": "ELC history excluding epigenetic level",
            "conditioning_set": "current epigenetic retained segment plus generation effects",
            "estimator": "coherent Gaussian-copula transfer entropy estimator",
            "numerical_result": f"raw={epi_closure['raw_estimate_bits']:.6f}; null-excess={epi_closure['null_excess_bits']:.6f}",
            "uncertainty": f"CI=[{epi_closure['ci_lower_bits']:.6f},{epi_closure['ci_upper_bits']:.6f}], p={epi_closure['generation_preserving_surrogate_p_value']:.6f}",
            "output_filename": str(corrected_archive / "coherent_continuous_transfer_entropy_summary.csv"),
            "pass_fail": "pass" if bool(epi_closure["null_excess_bits"] > 0.0 and epi_closure["generation_preserving_surrogate_p_value"] <= 0.05) else "fail",
        },
        {
            "draft32_requirement": "background predictive-contribution control",
            "exact_estimand": "I(X^{ELC}_{d,tau+1}; B_{d,tau} | X^{ELC,(k)}_{d,tau})",
            "target": "future full ELC",
            "source_or_intervention": "exogenous background control",
            "conditioning_set": "order-k full ELC history plus generation effects",
            "estimator": "coherent Gaussian-copula transfer entropy estimator",
            "numerical_result": f"raw={bg_pc['raw_estimate_bits']:.6f}; null-excess={bg_pc['null_excess_bits']:.6f}",
            "uncertainty": f"CI=[{bg_pc['ci_lower_bits']:.6f},{bg_pc['ci_upper_bits']:.6f}], p={bg_pc['generation_preserving_surrogate_p_value']:.6f}",
            "output_filename": str(corrected_archive / "coherent_continuous_transfer_entropy_summary.csv"),
            "pass_fail": "pass" if bool(bg_pc["null_excess_bits"] > 0.0 and bg_pc["generation_preserving_surrogate_p_value"] <= 0.05) else "fail",
        },
        {
            "draft32_requirement": "background closure control",
            "exact_estimand": "I(B_{d,tau+1}; X^{ELC,(k)}_{d,tau} | B_{d,tau})",
            "target": "future background control",
            "source_or_intervention": "full ELC history",
            "conditioning_set": "current background control plus generation effects",
            "estimator": "coherent Gaussian-copula transfer entropy estimator",
            "numerical_result": f"raw={bg_closure['raw_estimate_bits']:.6f}; null-excess={bg_closure['null_excess_bits']:.6f}",
            "uncertainty": f"CI=[{bg_closure['ci_lower_bits']:.6f},{bg_closure['ci_upper_bits']:.6f}], p={bg_closure['generation_preserving_surrogate_p_value']:.6f}",
            "output_filename": str(corrected_archive / "coherent_continuous_transfer_entropy_summary.csv"),
            "pass_fail": "pass" if bool(bg_closure["null_excess_bits"] <= 0.0 or bg_closure["generation_preserving_surrogate_p_value"] > 0.05) else "fail",
        },
        {
            "draft32_requirement": "focal-variant predictive information",
            "exact_estimand": "log_2 p(V=nu | x^{[l_epi]}_{d,s_l(tau)}, Z) / p(V=nu | Z), with Z=X^{ELC\\{[l_epi]},(k)}_{d,tau}",
            "target": "four-state phenotype and focal state nu",
            "source_or_intervention": "epigenetic retained segment",
            "conditioning_set": "order-k ELC history excluding epigenetic level plus generation effects",
            "estimator": "lineage-level cross-fitted multinomial logistic model",
            "numerical_result": f"binary nu={cat_binary['estimate_bits']:.6f}; focal log-ratio={cat_focal['estimate_bits']:.6f}",
            "uncertainty": f"focal CI=[{cat_focal['ci_lower_bits']:.6f},{cat_focal['ci_upper_bits']:.6f}]",
            "output_filename": str(corrected_archive / "categorical_phenotype_information_summary.csv"),
            "pass_fail": "pass" if bool(cat_focal["ci_lower_bits"] > 0.0 and cat_binary["ci_lower_bits"] > 0.0) else "fail",
        },
        {
            "draft32_requirement": "intervention-induced probability increase",
            "exact_estimand": "p_{theta_x}(nu | X^{ELC\\{[l_epi]},(k)}_{d,tau}) under the approved epigenetic intervention grid",
            "target": "focal phenotype nu",
            "source_or_intervention": "epigenetic intervention theta_x",
            "conditioning_set": "1,000 generation-stratified ELC contexts",
            "estimator": "Monte Carlo intervention response with common random numbers",
            "numerical_result": f"p_nu(0,0)={base_row[f'p_{NU_LABEL}']:.6f}; p_nu(1.5,1.5)={plus_row[f'p_{NU_LABEL}']:.6f}; delta={delta_plus:.6f}",
            "uncertainty": f"p_nu(1.5,1.5) CI=[{plus_row[f'p_{NU_LABEL}_ci_lower']:.6f},{plus_row[f'p_{NU_LABEL}_ci_upper']:.6f}]",
            "output_filename": str(corrected_archive / "intervention_generation_stratified_response_surface_summary.csv"),
            "pass_fail": "pass" if delta_plus > 0.0 else "fail",
        },
        {
            "draft32_requirement": "Fisher causal specificity",
            "exact_estimand": "context-averaged Fisher information matrix over all four phenotype states",
            "target": "four-state phenotype distribution",
            "source_or_intervention": "local epigenetic intervention parameters theta_reg, theta_stress",
            "conditioning_set": "1,000 generation-stratified ELC contexts",
            "estimator": "probability-scale centered finite differences with h=0.10",
            "numerical_result": f"min eigenvalue at (0,0)={fisher0_min_eig:.6f}",
            "uncertainty": "entry-wise lineage-cluster bootstrap intervals",
            "output_filename": str(corrected_archive / "corrected_intervention_fisher_information.csv"),
            "pass_fail": "pass" if fisher0_min_eig >= -1e-10 else "fail",
        },
        {
            "draft32_requirement": "intergenerational stability",
            "exact_estimand": "R^{[l_epi]}_{d,tau}(rho) for rho=1,...,5",
            "target": "future ELC excluding epigenetic level at tau+rho",
            "source_or_intervention": "epigenetic retained segment at source generation tau",
            "conditioning_set": "order-k history excluding epigenetic level plus generation effects",
            "estimator": "coherent Gaussian-copula transfer entropy estimator",
            "numerical_result": f"minimum null-excess across horizons={stability_min_excess:.6f}",
            "uncertainty": "lineage-cluster bootstrap intervals by horizon",
            "output_filename": str(corrected_archive / "corrected_intergenerational_stability_summary.csv"),
            "pass_fail": "pass" if stability_min_excess > 0.0 else "fail",
        },
        {
            "draft32_requirement": "level- and time-specific location",
            "exact_estimand": "T^{[l_epi -> l]}_{d,tau} by target level and retained t_l",
            "target": "development, microbiome, life history, and ecology at each retained t_l",
            "source_or_intervention": "epigenetic retained segment",
            "conditioning_set": "order-k history excluding epigenetic level plus generation effects; target level history remains included",
            "estimator": "coherent Gaussian-copula transfer entropy estimator",
            "numerical_result": f"{location_count} target-level/time estimates generated",
            "uncertainty": "lineage-cluster bootstrap intervals and generation-preserving surrogate nulls",
            "output_filename": str(corrected_archive / "corrected_level_time_specific_transfer_entropy_summary.csv"),
            "pass_fail": "pass" if location_count > 0 else "fail",
        },
        {
            "draft32_requirement": "multiple-system PID",
            "exact_estimand": "delta_G PID of epigenetic and ecological sources for source-excluded future ELC, after approved conditional preprocessing",
            "target": "future ELC excluding epigenetic and ecological levels",
            "source_or_intervention": "epigenetic hereditary factor and ecological candidate hereditary factor",
            "conditioning_set": "order-k ELC history excluding both source levels plus generation effects",
            "estimator": "Gaussian-copula transformation, linear residualization, Venkatesh-Schamberg delta_G PID",
            "numerical_result": f"matched joint={pid['matched_joint_information_bits']:.6f}; synergy={pid['synergy_bits']:.6f}; reconstruction error={pid['absolute_reconstruction_error_bits']:.3e}",
            "uncertainty": "centered lineage-cluster PID intervals",
            "output_filename": str(corrected_archive / "corrected_epigenetic_ecological_delta_g_pid_summary.csv"),
            "pass_fail": "pass" if bool(pid["within_tolerance"] and pid["synergy_bits"] > 0.0) else "fail",
        },
        {
            "draft32_requirement": "complex-unit qualification",
            "exact_estimand": "joint predictive contribution, joint closure, synergy, joint phenotype information, joint intervention response, and joint Fisher specificity",
            "target": "source-excluded ELC and phenotype nu",
            "source_or_intervention": "joint epigenetic-ecological configuration",
            "conditioning_set": "ELC history excluding both source levels",
            "estimator": "approved corrected estimators plus the final joint phenotype/intervention/Fisher tests",
            "numerical_result": complex_status,
            "uncertainty": "reported in component rows of complex_unit_qualification_table.csv",
            "output_filename": str(output_dir / "complex_unit_qualification_table.csv"),
            "pass_fail": "pass" if complex_status == "fully_qualified_complex_unit" else "fail",
        },
        {
            "draft32_requirement": "multiple-parent contribution",
            "exact_estimand": "I(Y; S_d,S_{d'} | Z) with receiving history Z and one-to-one deranged contributor d' != d",
            "target": "future receiving-individual ELC excluding epigenetic level",
            "source_or_intervention": "epigenetic source from d and epigenetic source from d'",
            "conditioning_set": "receiving individual's order-k ELC history excluding epigenetic level plus generation effects",
            "estimator": "coherent Gaussian-copula transfer entropy estimator",
            "numerical_result": f"joint raw={mp_joint['raw_estimate_bits']:.6f}; null-excess={mp_joint['null_excess_bits']:.6f}",
            "uncertainty": f"CI=[{mp_joint['ci_lower_bits']:.6f},{mp_joint['ci_upper_bits']:.6f}], p={mp_joint['generation_preserving_surrogate_p_value']:.6f}",
            "output_filename": str(corrected_archive / "corrected_multiple_parent_transfer_entropy_summary.csv"),
            "pass_fail": "pass" if bool(mp_joint["null_excess_bits"] > 0.0 and mp_joint["generation_preserving_surrogate_p_value"] <= 0.05) else "fail",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "draft32_compliance_matrix.csv", index=False)
    return out


def copy_analysis_code(output_dir: Path) -> None:
    code_dir = output_dir.parent / "analysis_code"
    code_dir.mkdir(parents=True, exist_ok=True)
    files = [
        "run_final_draft32_completeness_audit.py",
        "src/final_draft32_completeness_audit.py",
        "src/corrected_numerical_audit.py",
        "src/final_numerical_audit.py",
        "src/corrected_pilot_model.py",
        "src/pid_analysis.py",
        "reference_code/delta_g_pid.py",
        "src/information_measures.py",
    ]
    root = Path.cwd()
    for rel in files:
        src = root / rel
        if src.exists():
            shutil.copy2(src, code_dir / Path(rel).name)


def run_final_draft32_completeness_audit(
    output_root: Path,
    corrected_archive: Path,
    frozen_source_archive: Path,
    config: CorrectedAuditConfig | None = None,
) -> dict[str, object]:
    t0 = perf_counter()
    config = config or CorrectedAuditConfig()
    output_root.mkdir(parents=True, exist_ok=True)
    data_dir = output_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "final_draft32_completeness_configuration.json").write_text(json.dumps(asdict(config), indent=2))

    seed_results: list[dict[str, object]] = []
    for seed in config.seeds:
        data = load_saved_seed(seed, config, frozen_source_archive, multiparent=False)
        if data is None:
            raise FileNotFoundError(f"missing frozen seed archive for seed {seed} in {frozen_source_archive}")
        seed_results.append(data)

    table = build_analysis_table(seed_results, config)
    joint_summary, joint_boot, joint_seed, joint_decomp = run_joint_categorical_phenotype_information(table, config, data_dir)
    prob_by_context = simulate_joint_intervention_probabilities(seed_results, config, corrected_archive, data_dir)
    intervention_summary, intervention_delta, intervention_boot = summarize_joint_intervention(prob_by_context, config, data_dir)
    fisher, fisher_eigs, fisher_boot = compute_joint_fisher_information(prob_by_context, config, data_dir)
    complex_table = build_complex_unit_qualification_table(data_dir, corrected_archive)
    compliance = build_draft32_compliance_matrix(data_dir, corrected_archive)

    runtime = {
        "runtime_seconds": perf_counter() - t0,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": __import__("sklearn").__version__,
        "frozen_biological_simulation_changed": False,
        "previously_approved_information_analyses_rerun": False,
        "new_analyses": [
            "joint epigenetic-ecological categorical phenotype criterion",
            "minimal joint epigenetic-ecological intervention criterion",
            "joint epigenetic-ecological Fisher causal-specificity criterion",
        ],
        "random_seeds": {
            "simulation_seeds": list(config.seeds),
            "bootstrap_seed": config.bootstrap_seed,
            "intervention_seed": config.intervention_seed,
        },
    }
    (data_dir / "software_versions_and_runtime.json").write_text(json.dumps(runtime, indent=2))
    copy_analysis_code(data_dir)
    return {
        "data_dir": data_dir,
        "joint_categorical_summary": joint_summary,
        "joint_intervention_delta": intervention_delta,
        "joint_fisher": fisher,
        "complex_unit_qualification": complex_table,
        "draft32_compliance_matrix": compliance,
        "runtime": runtime,
    }
