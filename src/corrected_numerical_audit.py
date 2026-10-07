from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence
import json
import platform

import numpy as np
import pandas as pd
from scipy.special import ndtri
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .corrected_pilot_model import (
    PilotConfig,
    _params,
    acceptance_criteria,
    build_event_schedule,
    convergence_check,
    effective_nonzero_cross_level_edges,
    extract_analysis_segment,
    level_segment_indices,
    level_timestamps,
    nonzero_cross_level_edges,
    reproductive_indices,
    reproductive_state,
)
from .final_numerical_audit import (
    BACKGROUND_LEVEL,
    ELC_LEVELS,
    FinalAuditConfig,
    PHENOTYPE_LABELS,
    _empty_arrays_vec,
    _empty_full_arrays_vec,
    _empty_reproductive_arrays_vec,
    _founder_states_vec,
    _integrate_generation_vec,
    _next_generation_start_multiparent_vec,
    _next_generation_start_vec,
    _params_for_config,
    _segment,
    build_analysis_table,
    build_horizon_table,
    load_saved_seed,
    phenotype_support,
    save_seed_archive,
)
from .information_measures import summarize_interval, unit_index_groups
from .pid_analysis import _load_delta_g_pid


JITTER_GRID = (0.0, 1e-10, 1e-8, 1e-6, 1e-4)


@dataclass(frozen=True)
class CorrectedAuditConfig(FinalAuditConfig):
    covariance_jitter_grid: tuple[float, ...] = JITTER_GRID
    categorical_folds: int = 5
    categorical_logistic_c: float = 0.35
    intervention_probability_smoothing_alpha: float = 0.5
    n_response_grid_draws: int = 100
    n_intervention_draws: int = 1000


@dataclass(frozen=True)
class SourceBlock:
    label: str
    key: str


@dataclass(frozen=True)
class CmiFamily:
    family: str
    target_key: str
    condition_key: str
    sources: tuple[SourceBlock, ...]
    subsets: tuple[tuple[str, tuple[str, ...]], ...]
    interpretation: str


def _as2d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise AssertionError(f"expected 2-D array, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise FloatingPointError("non-finite values in information array")
    return arr


def gc_rank(x: np.ndarray) -> np.ndarray:
    x = _as2d(x)
    n = x.shape[0]
    ranks = pd.DataFrame(x).rank(axis=0, method="average").to_numpy(dtype=float)
    u = np.clip((ranks - 0.5) / float(n), 1e-6, 1.0 - 1e-6)
    return ndtri(u)


def _sym(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + a.T)


def _cov(x: np.ndarray, jitter: float = 0.0) -> np.ndarray:
    x = _as2d(x)
    centered = x - x.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
    if jitter > 0.0:
        cov = cov + float(jitter) * np.eye(cov.shape[0])
    return _sym(cov)


def _logdet_signed_spd(a: np.ndarray) -> float:
    a = _sym(np.asarray(a, dtype=float))
    sign, value = np.linalg.slogdet(a)
    if sign <= 0 or not np.isfinite(value):
        raise np.linalg.LinAlgError("matrix is not positive definite for signed log determinant")
    return float(value)


def _schur(cov: np.ndarray, keep: np.ndarray, cond: np.ndarray) -> np.ndarray:
    keep = np.asarray(keep, dtype=int)
    cond = np.asarray(cond, dtype=int)
    a = cov[np.ix_(keep, keep)]
    if cond.size == 0:
        return _sym(a)
    b = cov[np.ix_(keep, cond)]
    c = cov[np.ix_(cond, cond)]
    solved = np.linalg.solve(c, b.T)
    return _sym(a - b @ solved)


def coherent_cmi_from_cov(cov: np.ndarray, y: np.ndarray, z: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    z = np.asarray(z, dtype=int)
    s = np.asarray(s, dtype=int)
    yz = _schur(cov, y, z)
    sz = _schur(cov, s, z)
    ysz = _schur(cov, np.r_[y, s], z)
    return 0.5 * (_logdet_signed_spd(yz) + _logdet_signed_spd(sz) - _logdet_signed_spd(ysz)) / np.log(2.0)


def generation_design(meta: pd.DataFrame) -> np.ndarray:
    tau = meta["tau"].to_numpy(dtype=int)
    values = np.array(sorted(np.unique(tau)), dtype=int)
    if values.size <= 1:
        return np.zeros((len(meta), 0), dtype=float)
    out = np.zeros((len(meta), values.size - 1), dtype=float)
    for j, value in enumerate(values[1:]):
        out[:, j] = tau == value
    return out


def _ranked_joint_arrays(
    table: Mapping[str, object],
    family: CmiFamily,
    *,
    include_generation: bool = True,
) -> tuple[np.ndarray, dict[str, slice], dict[str, np.ndarray]]:
    y = gc_rank(_as2d(table[family.target_key]))
    z_raw = _as2d(table[family.condition_key])
    if include_generation:
        gen = generation_design(table["meta"])
        z_raw = np.hstack([z_raw, gen]) if gen.size else z_raw
    z = gc_rank(z_raw)
    parts = [y, z]
    slices: dict[str, slice] = {}
    start = 0
    slices["target"] = slice(start, start + y.shape[1])
    start += y.shape[1]
    slices["condition"] = slice(start, start + z.shape[1])
    start += z.shape[1]
    source_arrays = {}
    for source in family.sources:
        arr = gc_rank(_as2d(table[source.key]))
        source_arrays[source.label] = arr
        parts.append(arr)
        slices[source.label] = slice(start, start + arr.shape[1])
        start += arr.shape[1]
    return np.hstack(parts), slices, source_arrays


def _slice_indices(s: slice) -> np.ndarray:
    return np.arange(int(s.start), int(s.stop), dtype=int)


def _family_estimates_from_cov(cov: np.ndarray, slices: Mapping[str, slice], family: CmiFamily) -> dict[str, float]:
    y_idx = _slice_indices(slices["target"])
    z_idx = _slice_indices(slices["condition"])
    out = {}
    for analysis, labels in family.subsets:
        s_idx = np.concatenate([_slice_indices(slices[label]) for label in labels])
        out[analysis] = coherent_cmi_from_cov(cov, y_idx, z_idx, s_idx)
    return out


def _family_residual_joint(joint: np.ndarray, slices: Mapping[str, slice], family: CmiFamily) -> tuple[np.ndarray, dict[str, slice]]:
    y = joint[:, slices["target"]]
    z = joint[:, slices["condition"]]
    parts = [_residualize_ols(y, z)]
    residual_slices = {"target": slice(0, parts[0].shape[1])}
    start = parts[0].shape[1]
    for source in family.sources:
        resid = _residualize_ols(joint[:, slices[source.label]], z)
        parts.append(resid)
        residual_slices[source.label] = slice(start, start + resid.shape[1])
        start += resid.shape[1]
    return np.hstack(parts), residual_slices


def _family_estimates_from_residual_cov(cov: np.ndarray, slices: Mapping[str, slice], family: CmiFamily) -> dict[str, float]:
    y_idx = _slice_indices(slices["target"])
    out = {}
    for analysis, labels in family.subsets:
        s_idx = np.concatenate([_slice_indices(slices[label]) for label in labels])
        ys_idx = np.r_[y_idx, s_idx]
        out[analysis] = 0.5 * (
            _logdet_signed_spd(cov[np.ix_(y_idx, y_idx)])
            + _logdet_signed_spd(cov[np.ix_(s_idx, s_idx)])
            - _logdet_signed_spd(cov[np.ix_(ys_idx, ys_idx)])
        ) / np.log(2.0)
    return out


def _cluster_summaries_from_joint(joint: np.ndarray, meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    groups = unit_index_groups(meta)
    unit_ids = np.array(list(groups.keys()), dtype=int)
    centered = joint - joint.mean(axis=0, keepdims=True)
    counts, sums, crosses = [], [], []
    for uid in unit_ids:
        block = centered[groups[int(uid)]]
        counts.append(block.shape[0])
        sums.append(block.sum(axis=0))
        crosses.append(block.T @ block)
    return unit_ids, np.asarray(counts, dtype=float), np.asarray(sums, dtype=float), np.asarray(crosses, dtype=float)


def _weighted_cov_from_cluster_summaries(
    counts: np.ndarray,
    sums: np.ndarray,
    crosses: np.ndarray,
    weights: np.ndarray,
    jitter: float,
) -> np.ndarray:
    n = float(weights @ counts)
    s = weights @ sums
    c = np.tensordot(weights, crosses, axes=(0, 0))
    cov = (c - np.outer(s, s) / max(n, 1.0)) / max(n - 1.0, 1.0)
    if jitter > 0.0:
        cov += jitter * np.eye(cov.shape[0])
    return _sym(cov)


def _bootstrap_family(
    joint: np.ndarray,
    slices: Mapping[str, slice],
    family: CmiFamily,
    meta: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
    jitter: float,
) -> pd.DataFrame:
    residual_joint, residual_slices = _family_residual_joint(joint, slices, family)
    unit_ids, counts, sums, crosses = _cluster_summaries_from_joint(residual_joint, meta)
    n_units = len(unit_ids)
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_boot):
        weights = rng.multinomial(n_units, np.full(n_units, 1.0 / n_units)).astype(float)
        cov = _weighted_cov_from_cluster_summaries(counts, sums, crosses, weights, jitter)
        vals = _family_estimates_from_residual_cov(cov, residual_slices, family)
        for analysis, estimate in vals.items():
            rows.append({"family": family.family, "analysis": analysis, "bootstrap": b, "estimate_bits": float(estimate)})
    return pd.DataFrame(rows)


def _generation_preserving_permutation_indices(meta: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    perm = np.arange(len(meta), dtype=int)
    for _, group in meta.groupby(["seed", "tau"], sort=False):
        idx = group.index.to_numpy(dtype=int)
        perm[idx] = rng.permutation(idx)
    return perm


def _null_family(
    joint: np.ndarray,
    slices: Mapping[str, slice],
    family: CmiFamily,
    meta: pd.DataFrame,
    *,
    n_null: int,
    seed: int,
    jitter: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    residual_joint, residual_slices = _family_residual_joint(joint, slices, family)
    y_slice = residual_slices["target"]
    source_slices = [residual_slices[source.label] for source in family.sources]
    a = residual_joint[:, y_slice]
    s = np.hstack([residual_joint[:, source_slice] for source_slice in source_slices])
    a_center = a - a.mean(axis=0, keepdims=True)
    s_center = s - s.mean(axis=0, keepdims=True)
    n = joint.shape[0]
    cov_a = a_center.T @ a_center / max(n - 1, 1)
    cov_s = s_center.T @ s_center / max(n - 1, 1)
    y_dim = y_slice.stop - y_slice.start
    start = y_dim
    local_start = 0
    null_slices = {"target": slice(0, y_dim)}
    for source, source_slice in zip(family.sources, source_slices):
        dim = source_slice.stop - source_slice.start
        null_slices[source.label] = slice(start + local_start, start + local_start + dim)
        local_start += dim
    rows = []
    for r in range(n_null):
        perm = _generation_preserving_permutation_indices(meta, rng)
        cross_as = a_center.T @ s_center[perm] / max(n - 1, 1)
        cov = np.zeros((a.shape[1] + s.shape[1], a.shape[1] + s.shape[1]), dtype=float)
        cov[: a.shape[1], : a.shape[1]] = cov_a
        cov[a.shape[1] :, a.shape[1] :] = cov_s
        cov[: a.shape[1], a.shape[1] :] = cross_as
        cov[a.shape[1] :, : a.shape[1]] = cross_as.T
        if jitter > 0.0:
            cov += jitter * np.eye(cov.shape[0])
        vals = _family_estimates_from_residual_cov(_sym(cov), null_slices, family)
        for analysis, estimate in vals.items():
            rows.append({"family": family.family, "analysis": analysis, "null_iteration": r, "estimate_bits_raw_signed": float(estimate)})
    return pd.DataFrame(rows)


def _summarize_family(
    family: CmiFamily,
    observed: Mapping[str, float],
    boot: pd.DataFrame,
    null: pd.DataFrame,
    *,
    target_dim: int,
    source_dims: Mapping[str, int],
    condition_dim: int,
    jitter: float,
) -> pd.DataFrame:
    rows = []
    for analysis, labels in family.subsets:
        obs = float(observed[analysis])
        boot_vals = boot.loc[boot["analysis"] == analysis, "estimate_bits"].to_numpy(dtype=float)
        null_vals = null.loc[null["analysis"] == analysis, "estimate_bits_raw_signed"].to_numpy(dtype=float)
        se = float(np.std(boot_vals, ddof=1)) if boot_vals.size > 1 else np.nan
        z = 1.959963984540054
        rows.append(
            {
                "family": family.family,
                "analysis": analysis,
                "target_dimension": int(target_dim),
                "source_dimension": int(sum(source_dims[label] for label in labels)),
                "conditioning_dimension": int(condition_dim),
                "source_labels": "+".join(labels),
                "estimator": "generation-conditioned coherent Gaussian-copula CMI from one joint covariance",
                "selected_common_jitter": jitter,
                "raw_estimate_bits": obs,
                "bootstrap_mean_bits": float(np.mean(boot_vals)),
                "bootstrap_se_bits": se,
                "ci_method": "lineage-cluster bootstrap, centered standard-error interval",
                "ci_lower_bits": obs - z * se,
                "ci_upper_bits": obs + z * se,
                "surrogate_null_mean_bits": float(np.mean(null_vals)),
                "null_excess_bits": obs - float(np.mean(null_vals)),
                "generation_preserving_surrogate_p_value": float((1.0 + np.sum(null_vals >= obs)) / (null_vals.size + 1.0)),
                "null_min_bits_raw_signed": float(np.min(null_vals)),
                "null_max_bits_raw_signed": float(np.max(null_vals)),
                "null_negative_proportion": float(np.mean(null_vals < 0.0)),
                "n_null": int(null_vals.size),
                "biological_interpretation": family.interpretation,
            }
        )
    return pd.DataFrame(rows)


def _seed_level_family(
    seed_tables: Sequence[Mapping[str, object]],
    family: CmiFamily,
    *,
    jitter: float,
    include_generation: bool = True,
) -> pd.DataFrame:
    rows = []
    for table in seed_tables:
        seed = int(table["meta"]["seed"].iloc[0])
        joint, slices, _ = _ranked_joint_arrays(table, family, include_generation=include_generation)
        vals = _family_estimates_from_cov(_cov(joint, jitter), slices, family)
        for analysis, estimate in vals.items():
            rows.append({"seed": seed, "family": family.family, "analysis": analysis, "raw_estimate_bits": float(estimate)})
    return pd.DataFrame(rows)


def continuous_families() -> tuple[CmiFamily, ...]:
    return (
        CmiFamily(
            "epigenetic_predictive_contribution",
            "target_remainder_without_epigenetic",
            "history_remainder_without_epigenetic",
            (SourceBlock("epigenetic", "source_epigenetic"),),
            (("epigenetic_predictive_contribution", ("epigenetic",)),),
            "predictive contribution of the epigenetic hereditary factor to the future source-excluded ELC",
        ),
        CmiFamily(
            "background_predictive_contribution",
            "target_full_elc",
            "history_full_elc",
            (SourceBlock("background", "source_background"),),
            (("background_predictive_contribution", ("background",)),),
            "predictive contribution of the exogenous background control to the future ELC",
        ),
        CmiFamily(
            "epigenetic_predictive_closure",
            "target_epigenetic_future",
            "source_epigenetic",
            (SourceBlock("remainder_history", "history_remainder_without_epigenetic"),),
            (("epigenetic_predictive_closure", ("remainder_history",)),),
            "dependence of the future epigenetic state on the current source-excluded ELC history beyond the current epigenetic state",
        ),
        CmiFamily(
            "background_closure_control",
            "target_background_future",
            "source_background",
            (SourceBlock("full_elc_history", "history_full_elc"),),
            (("background_closure_control", ("full_elc_history",)),),
            "negative-control closure quantity for an exogenous background condition outside the ELC",
        ),
        CmiFamily(
            "epigenetic_ecological_predictive_contribution",
            "target_remainder_without_epigenetic_ecological",
            "history_remainder_without_epigenetic_ecological",
            (
                SourceBlock("epigenetic", "source_epigenetic"),
                SourceBlock("ecological", "source_ecological"),
            ),
            (
                ("epigenetic_only_predictive_contribution_to_joint_target", ("epigenetic",)),
                ("ecological_only_predictive_contribution_to_joint_target", ("ecological",)),
                ("joint_epigenetic_ecological_predictive_contribution", ("epigenetic", "ecological")),
            ),
            "joint predictive contribution of epigenetic and ecological factors to the future ELC excluding both source levels",
        ),
        CmiFamily(
            "epigenetic_ecological_predictive_closure",
            "target_epigenetic_ecological_future",
            "complex_source_epigenetic_ecological",
            (SourceBlock("remainder_history", "history_remainder_without_epigenetic_ecological"),),
            (("joint_epigenetic_ecological_predictive_closure", ("remainder_history",)),),
            "predictive closure of the joint epigenetic-ecological state through the source-excluded ELC history",
        ),
    )


def choose_common_jitter(
    table: Mapping[str, object],
    families: Sequence[CmiFamily],
    config: CorrectedAuditConfig,
    output_dir: Path,
) -> float:
    rows = []
    selected = None
    for jitter in config.covariance_jitter_grid:
        all_ok = True
        for family in families:
            joint, slices, _ = _ranked_joint_arrays(table, family, include_generation=True)
            try:
                vals = _family_estimates_from_cov(_cov(joint, jitter), slices, family)
                nesting_ok = True
                if len(family.sources) > 1:
                    joint_analysis = [name for name, labels in family.subsets if len(labels) == len(family.sources)]
                    if joint_analysis:
                        joint_value = vals[joint_analysis[0]]
                        for name, labels in family.subsets:
                            if len(labels) == 1 and joint_value + 1e-10 < vals[name]:
                                nesting_ok = False
                status = "ok" if nesting_ok else "nesting_failed"
                if not nesting_ok:
                    all_ok = False
            except Exception as exc:
                vals = {}
                nesting_ok = False
                status = f"failed:{type(exc).__name__}"
                all_ok = False
            row = {"jitter": jitter, "family": family.family, "status": status, "nesting_ok": nesting_ok}
            for key, value in vals.items():
                row[key] = float(value)
            rows.append(row)
        if selected is None and all_ok:
            selected = jitter
    pd.DataFrame(rows).to_csv(output_dir / "coherent_cmi_jitter_sensitivity.csv", index=False)
    if selected is None:
        raise FloatingPointError("no common covariance jitter produced stable coherent CMI estimates")
    return float(selected)


def run_continuous_transfer_entropy(
    table: Mapping[str, object],
    seed_tables: Sequence[Mapping[str, object]],
    config: CorrectedAuditConfig,
    output_dir: Path,
    *,
    jitter: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries, boots, nulls, seed_rows = [], [], [], []
    for i, family in enumerate(continuous_families()):
        print(f"continuous coherent CMI {i + 1}/{len(continuous_families())}: {family.family}", flush=True)
        joint, slices, source_arrays = _ranked_joint_arrays(table, family, include_generation=True)
        cov = _cov(joint, jitter)
        observed = _family_estimates_from_cov(cov, slices, family)
        boot = _bootstrap_family(
            joint,
            slices,
            family,
            table["meta"],
            n_boot=config.n_bootstrap,
            seed=config.bootstrap_seed + 22000 + 991 * i,
            jitter=jitter,
        )
        null = _null_family(
            joint,
            slices,
            family,
            table["meta"],
            n_null=config.n_null,
            seed=config.null_seed + 22000 + 991 * i,
            jitter=jitter,
        )
        source_dims = {source.label: source_arrays[source.label].shape[1] for source in family.sources}
        summaries.append(
            _summarize_family(
                family,
                observed,
                boot,
                null,
                target_dim=slices["target"].stop - slices["target"].start,
                source_dims=source_dims,
                condition_dim=slices["condition"].stop - slices["condition"].start,
                jitter=jitter,
            )
        )
        boots.append(boot)
        nulls.append(null)
        seed_rows.append(_seed_level_family(seed_tables, family, jitter=jitter, include_generation=True))
    summary = pd.concat(summaries, ignore_index=True)
    boot_df = pd.concat(boots, ignore_index=True)
    null_df = pd.concat(nulls, ignore_index=True)
    seed_df = pd.concat(seed_rows, ignore_index=True)
    summary.to_csv(output_dir / "coherent_continuous_transfer_entropy_summary.csv", index=False)
    boot_df.to_csv(output_dir / "coherent_continuous_transfer_entropy_bootstrap.csv", index=False)
    null_df.to_csv(output_dir / "coherent_continuous_transfer_entropy_generation_preserving_null.csv", index=False)
    seed_df.to_csv(output_dir / "coherent_continuous_transfer_entropy_by_seed.csv", index=False)
    return summary, boot_df, null_df, seed_df


def _feature_matrix_for_phenotype(table: Mapping[str, object], *, include_source: bool) -> np.ndarray:
    z = gc_rank(_as2d(table["history_remainder_without_epigenetic"]))
    gen = generation_design(table["meta"])
    parts = [z]
    if include_source:
        parts.append(gc_rank(_as2d(table["source_epigenetic"])))
    if gen.size:
        parts.append(gen)
    return np.hstack(parts)


def _label_codes(labels: np.ndarray) -> np.ndarray:
    lookup = {label: i for i, label in enumerate(PHENOTYPE_LABELS)}
    return np.asarray([lookup[str(label)] for label in labels], dtype=int)


def run_categorical_phenotype_information(
    table: Mapping[str, object],
    config: CorrectedAuditConfig,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("categorical phenotype transfer entropy: cross-fitted multinomial models", flush=True)
    labels = np.asarray(table["variant_label"], dtype=object)
    y = _label_codes(labels)
    x_null = _feature_matrix_for_phenotype(table, include_source=False)
    x_full = _feature_matrix_for_phenotype(table, include_source=True)
    groups = table["meta"]["unit_id"].to_numpy(dtype=int)
    gkf = GroupKFold(n_splits=config.categorical_folds)
    p_null = np.zeros((len(y), len(PHENOTYPE_LABELS)), dtype=float)
    p_full = np.zeros_like(p_null)
    fold_rows = []
    for fold, (train, test) in enumerate(gkf.split(x_full, y, groups=groups)):
        null_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=config.categorical_logistic_c,
                penalty="l2",
                solver="lbfgs",
                max_iter=900,
                class_weight=None,
            ),
        )
        full_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=config.categorical_logistic_c,
                penalty="l2",
                solver="lbfgs",
                max_iter=900,
                class_weight=None,
            ),
        )
        null_model.fit(x_null[train], y[train])
        full_model.fit(x_full[train], y[train])
        p_null[test] = null_model.predict_proba(x_null[test])
        p_full[test] = full_model.predict_proba(x_full[test])
        fold_rows.append({"fold": fold, "n_train": int(train.size), "n_test": int(test.size)})
    eps = 1e-12
    p_null = np.clip(p_null, eps, 1.0)
    p_full = np.clip(p_full, eps, 1.0)
    p_null = p_null / p_null.sum(axis=1, keepdims=True)
    p_full = p_full / p_full.sum(axis=1, keepdims=True)
    idx = np.arange(len(y))
    state_log_ratio = np.log2(p_full[idx, y] / p_null[idx, y])
    complete_te = float(np.mean(state_log_ratio))
    nu_index = PHENOTYPE_LABELS.index("nu_early_maturation_high_growth")
    event = y == nu_index
    p_full_event = np.where(event, p_full[:, nu_index], 1.0 - p_full[:, nu_index])
    p_null_event = np.where(event, p_null[:, nu_index], 1.0 - p_null[:, nu_index])
    binary_log_ratio = np.log2(np.clip(p_full_event, eps, 1.0) / np.clip(p_null_event, eps, 1.0))
    binary_te = float(np.mean(binary_log_ratio))
    focal_log_ratio_all_contexts = np.log2(p_full[:, nu_index] / p_null[:, nu_index])
    focal_log_ratio_nu_observed = float(np.mean(focal_log_ratio_all_contexts[event]))
    decomp_rows = []
    for label_index, label in enumerate(PHENOTYPE_LABELS):
        mask = y == label_index
        contribution = float(np.mean(state_log_ratio[mask]) * np.mean(mask)) if np.any(mask) else 0.0
        decomp_rows.append(
            {
                "phenotype_state": label,
                "proportion": float(np.mean(mask)),
                "mean_state_specific_log_ratio_bits": float(np.mean(state_log_ratio[mask])) if np.any(mask) else np.nan,
                "weighted_contribution_bits": contribution,
            }
        )
    decomp = pd.DataFrame(decomp_rows)
    decomp.to_csv(output_dir / "categorical_phenotype_state_specific_decomposition.csv", index=False)
    meta = table["meta"].reset_index(drop=True)
    score_df = meta[["seed", "lineage_id", "unit_id", "tau"]].copy()
    score_df["phenotype_state"] = labels
    score_df["complete_state_log_ratio_bits"] = state_log_ratio
    score_df["binary_nu_log_ratio_bits"] = binary_log_ratio
    score_df["focal_nu_probability_log_ratio_bits"] = focal_log_ratio_all_contexts
    for label_index, label in enumerate(PHENOTYPE_LABELS):
        score_df[f"p_full_{label}"] = p_full[:, label_index]
        score_df[f"p_null_{label}"] = p_null[:, label_index]
    score_df.to_csv(output_dir / "categorical_phenotype_crossfit_scores.csv", index=False)
    groups_index = unit_index_groups(meta)
    unit_ids = np.array(list(groups_index.keys()), dtype=int)
    rng = np.random.default_rng(config.bootstrap_seed + 66000)
    boot_rows = []
    for b in range(config.n_bootstrap):
        sampled = rng.choice(unit_ids, size=len(unit_ids), replace=True)
        boot_idx = np.concatenate([groups_index[int(uid)] for uid in sampled])
        boot_rows.append(
            {
                "bootstrap": b,
                "complete_phenotype_transfer_entropy_bits": float(np.mean(state_log_ratio[boot_idx])),
                "binary_nu_transfer_entropy_bits": float(np.mean(binary_log_ratio[boot_idx])),
                "focal_nu_log_ratio_given_nu_bits": float(np.mean(focal_log_ratio_all_contexts[boot_idx][event[boot_idx]])),
            }
        )
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(output_dir / "categorical_phenotype_bootstrap.csv", index=False)
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
    seed_df.to_csv(output_dir / "categorical_phenotype_by_seed.csv", index=False)
    z = 1.959963984540054
    rows = []
    for quantity, observed, col in [
        ("complete_phenotype_transfer_entropy", complete_te, "complete_phenotype_transfer_entropy_bits"),
        ("binary_nu_transfer_entropy", binary_te, "binary_nu_transfer_entropy_bits"),
        ("focal_nu_log_ratio_given_nu", focal_log_ratio_nu_observed, "focal_nu_log_ratio_given_nu_bits"),
    ]:
        vals = boot[col].to_numpy(dtype=float)
        se = float(np.std(vals, ddof=1))
        rows.append(
            {
                "quantity": quantity,
                "estimate_bits": observed,
                "bootstrap_mean_bits": float(np.mean(vals)),
                "bootstrap_se_bits": se,
                "ci_method": "lineage-cluster bootstrap over cross-fitted log-probability ratios",
                "ci_lower_bits": observed - z * se,
                "ci_upper_bits": observed + z * se,
            }
        )
    summary = pd.DataFrame(rows)
    summary["complete_ge_binary_nu_check"] = bool(complete_te + 1e-10 >= binary_te)
    summary["weighted_decomposition_sum_bits"] = float(decomp["weighted_contribution_bits"].sum())
    summary["weighted_decomposition_error_bits"] = abs(float(decomp["weighted_contribution_bits"].sum()) - complete_te)
    summary.to_csv(output_dir / "categorical_phenotype_information_summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "categorical_phenotype_crossfit_folds.csv", index=False)
    generation = score_df.groupby(["tau", "phenotype_state"]).size().unstack(fill_value=0).reindex(columns=PHENOTYPE_LABELS, fill_value=0)
    generation = generation.div(generation.sum(axis=1), axis=0).reset_index()
    generation.to_csv(output_dir / "generation_specific_phenotype_distribution.csv", index=False)
    return summary, boot, seed_df, decomp


def _select_generation_stratified_contexts(seed_results: Sequence[Mapping[str, object]], config: CorrectedAuditConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.intervention_seed)
    rows = []
    tau_values = np.arange(config.source_tau_start, config.source_tau_stop + 1, dtype=int)
    for seed_data in seed_results:
        seed = int(seed_data["seed"])
        n = seed_data["config"].n_lineages
        base = config.n_contexts_per_seed // len(tau_values)
        remainder = config.n_contexts_per_seed % len(tau_values)
        for pos, tau in enumerate(tau_values):
            count = base + (1 if pos < remainder else 0)
            chosen = rng.choice(np.arange(n), size=count, replace=False)
            for lineage in chosen:
                rows.append({"seed": seed, "lineage_id": int(lineage), "tau": int(tau), "context_id": len(rows)})
    return pd.DataFrame(rows)


def _smooth_probabilities(prob: np.ndarray, n_draws: int, alpha: float) -> np.ndarray:
    counts = np.asarray(prob, dtype=float) * float(n_draws)
    smoothed = counts + float(alpha)
    smoothed = smoothed / smoothed.sum(axis=-1, keepdims=True)
    return smoothed


class _CommonRandomThetaBatch:
    def __init__(self, seed: int, n_theta: int, base_n: int):
        self.rng = np.random.default_rng(seed)
        self.n_theta = int(n_theta)
        self.base_n = int(base_n)

    def normal(self, loc=0.0, scale=1.0, size=None):
        if size is not None:
            size_tuple = tuple(size) if isinstance(size, tuple) else (int(size),)
            if len(size_tuple) >= 1 and size_tuple[0] == self.n_theta * self.base_n:
                base_size = (self.base_n,) + size_tuple[1:]
                z = self.rng.normal(0.0, 1.0, size=base_size)
                z = np.tile(z, (self.n_theta,) + (1,) * (len(size_tuple) - 1))
                return np.asarray(loc) + np.asarray(scale) * z
        return self.rng.normal(loc, scale, size=size)


def run_generation_stratified_intervention(
    seed_results: Sequence[Mapping[str, object]],
    config: CorrectedAuditConfig,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("generation-stratified intervention simulation", flush=True)
    contexts = _select_generation_stratified_contexts(seed_results, config)
    contexts.to_csv(output_dir / "intervention_generation_stratified_contexts.csv", index=False)
    seed_map = {int(sd["seed"]): sd for sd in seed_results}
    theta_values = np.linspace(config.intervention_grid_min, config.intervention_grid_max, config.intervention_grid_size)
    response_grid = {(float(a), float(b)) for a in theta_values for b in theta_values}
    fisher_points = [(0.0, 0.0), (-0.5, 0.0), (0.5, 0.0), (0.0, -0.5), (0.0, 0.5)]
    h = config.fisher_step
    fisher_theta = set()
    for a, b in fisher_points:
        fisher_theta.update([(a, b), (a + h, b), (a - h, b), (a, b + h), (a, b - h)])
    theta_set = sorted(response_grid | fisher_theta)
    schedule_cache: dict[int, Sequence[tuple[float, tuple[str, ...]]]] = {}
    rows = []
    for (seed, tau), group in contexts.groupby(["seed", "tau"], sort=True):
        seed_data = seed_map[int(seed)]
        lineages = group["lineage_id"].to_numpy(dtype=int)
        context_ids = group["context_id"].to_numpy(dtype=int)
        designs = [
            ("response_grid", np.asarray(sorted(response_grid), dtype=float), int(config.n_response_grid_draws)),
            ("local_fisher", np.asarray(sorted(fisher_theta), dtype=float), int(config.n_intervention_draws)),
        ]
        for design_index, (analysis_design, theta_array_all, n_draws) in enumerate(designs):
            repeated_lineages = np.repeat(lineages, n_draws)
            repeated_context_ids = np.repeat(context_ids, n_draws)
            batch_n = repeated_lineages.size
            batch_config = config.pilot_config(int(seed), n_lineages=batch_n)
            if batch_n not in schedule_cache:
                schedule_cache[batch_n] = build_event_schedule(batch_config)
            source_start_template = {
                level: np.asarray(seed_data["full_time_series"][level])[repeated_lineages, int(tau), 0].copy()
                for level in ELC_LEVELS + (BACKGROUND_LEVEL,)
            }
            theta_chunk_size = 8
            for theta_start in range(0, len(theta_array_all), theta_chunk_size):
                theta_chunk = theta_array_all[theta_start : theta_start + theta_chunk_size]
                n_theta = theta_chunk.shape[0]
                theta_epi = np.repeat(theta_chunk, batch_n, axis=0)
                expanded_config = config.pilot_config(int(seed), n_lineages=n_theta * batch_n)
                if n_theta * batch_n not in schedule_cache:
                    schedule_cache[n_theta * batch_n] = build_event_schedule(expanded_config)
                expanded_start = {level: np.tile(values, (n_theta, 1)) for level, values in source_start_template.items()}
                expanded_start["epigenetic"] = expanded_start["epigenetic"] + theta_epi
                expanded_lineages = np.tile(repeated_lineages, n_theta)
                rng = _CommonRandomThetaBatch(
                    config.intervention_seed
                    + int(seed) * 100_000
                    + int(tau) * 1009
                    + design_index * 10_000_019,
                    n_theta=n_theta,
                    base_n=batch_n,
                )
                source_retained = _integrate_generation_vec(
                    expanded_start,
                    int(tau),
                    rng,
                    expanded_config,
                    seed_data["parameters"],
                    schedule_cache[n_theta * batch_n],
                    d_idx=expanded_lineages,
                )
                reproductive_for_future = {
                    level: reproductive_state(source_retained[level], level, expanded_config)
                    for level in ELC_LEVELS + (BACKGROUND_LEVEL,)
                }
                start = _next_generation_start_vec(
                    reproductive_for_future,
                    rng,
                    expanded_config,
                    seed_data["parameters"],
                )
                retained = _integrate_generation_vec(
                    start,
                    int(tau) + 1,
                    rng,
                    expanded_config,
                    seed_data["parameters"],
                    schedule_cache[n_theta * batch_n],
                    d_idx=expanded_lineages,
                )
                life_all = retained["life_history"].reshape(n_theta, batch_n, retained["life_history"].shape[1], retained["life_history"].shape[2])
                for theta_i, (theta_reg, theta_stress) in enumerate(theta_chunk):
                    life = life_all[theta_i]
                    maturation = life[:, :, 0]
                    crosses = maturation >= config.theta_maturation
                    crossing = np.full(batch_n, -1, dtype=int)
                    for t_l in range(maturation.shape[1]):
                        crossing[(crossing < 0) & crosses[:, t_l]] = t_l
                    life_times = level_timestamps(expanded_config)["life_history"]
                    crossing_u = np.full(crossing.shape, np.nan, dtype=float)
                    valid = crossing >= 0
                    crossing_u[valid] = life_times[crossing[valid]]
                    early = valid & (crossing_u < expanded_config.early_maturation_u)
                    high = life[:, -1, 1] >= config.theta_growth
                    labels = np.empty(batch_n, dtype=object)
                    labels[early & high] = "nu_early_maturation_high_growth"
                    labels[early & ~high] = "early_maturation_low_growth"
                    labels[~early & high] = "non_early_maturation_high_growth"
                    labels[~early & ~high] = "non_early_maturation_low_growth"
                    tmp = pd.DataFrame({"context_id": repeated_context_ids, "phenotype_state": labels})
                    counts = tmp.groupby(["context_id", "phenotype_state"]).size().unstack(fill_value=0).reindex(columns=PHENOTYPE_LABELS, fill_value=0)
                    counts = counts.reindex(context_ids, fill_value=0)
                    probs = counts.div(float(n_draws))
                    for context_id, lineage, prob_row in zip(context_ids, lineages, probs.to_dict("records")):
                        rows.append(
                            {
                                "seed": int(seed),
                                "lineage_id": int(lineage),
                                "tau": int(tau),
                                "context_id": int(context_id),
                                "analysis_design": analysis_design,
                                "monte_carlo_draws": int(n_draws),
                                "theta_reg": float(theta_reg),
                                "theta_stress": float(theta_stress),
                                "on_approved_7x7_response_grid": bool(analysis_design == "response_grid"),
                                "used_for_fisher_finite_difference": bool(analysis_design == "local_fisher"),
                                **{f"p_{label}": float(prob_row[label]) for label in PHENOTYPE_LABELS},
                            }
                        )
    prob_by_context = pd.DataFrame(rows)
    prob_by_context.to_csv(output_dir / "intervention_generation_stratified_probability_by_context.csv", index=False)
    response = prob_by_context[prob_by_context["analysis_design"] == "response_grid"].copy()
    prob_cols = [f"p_{label}" for label in PHENOTYPE_LABELS]
    summary = response.groupby(["theta_reg", "theta_stress"])[prob_cols].mean().reset_index()
    summary = _bootstrap_intervention_response(summary, response, config)
    summary.to_csv(output_dir / "intervention_generation_stratified_response_surface_summary.csv", index=False)
    by_generation = response.groupby(["tau", "theta_reg", "theta_stress"])[prob_cols].mean().reset_index()
    by_generation.to_csv(output_dir / "intervention_response_by_generation.csv", index=False)
    fisher, fisher_boot = corrected_fisher_information(prob_by_context, config)
    fisher.to_csv(output_dir / "corrected_intervention_fisher_information.csv", index=False)
    fisher_boot.to_csv(output_dir / "corrected_intervention_fisher_information_bootstrap.csv", index=False)
    variation_rows = []
    for theta, group in by_generation.groupby(["theta_reg", "theta_stress"]):
        values = group["p_nu_early_maturation_high_growth"].to_numpy(dtype=float)
        variation_rows.append(
            {
                "theta_reg": theta[0],
                "theta_stress": theta[1],
                "min_generation_mean_p_nu": float(values.min()),
                "max_generation_mean_p_nu": float(values.max()),
                "range_generation_mean_p_nu": float(values.max() - values.min()),
            }
        )
    variation = pd.DataFrame(variation_rows)
    variation.to_csv(output_dir / "intervention_generation_variation_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "intervention": "epigenetic",
                "timing": "beginning_of_source_generation_tau",
                "mode": "additive_shift_to_context_specific_initial_latent_logits",
                "zero_point": "contextual_no_modification",
                "source_segment_regenerated": True,
                "future_generation_regenerated": True,
                "theta_components": "theta_reg;theta_stress",
                "finite_difference_step": float(config.fisher_step),
                "probability_smoothing_alpha": float(config.intervention_probability_smoothing_alpha),
                "response_grid_monte_carlo_draws_per_context": int(config.n_response_grid_draws),
                "local_gradient_fisher_monte_carlo_draws_per_context": int(config.n_intervention_draws),
            }
        ]
    ).to_csv(output_dir / "epigenetic_intervention_design_metadata.csv", index=False)
    return summary, prob_by_context, fisher, variation


def _bootstrap_intervention_response(summary: pd.DataFrame, response: pd.DataFrame, config: CorrectedAuditConfig) -> pd.DataFrame:
    prob_cols = [f"p_{label}" for label in PHENOTYPE_LABELS]
    response = response.copy()
    response["cluster_id"] = response["seed"].astype(str) + "_" + response["lineage_id"].astype(str)
    groups = {cid: group.index.to_numpy(dtype=int) for cid, group in response.groupby("cluster_id")}
    cluster_ids = np.array(list(groups.keys()), dtype=object)
    rng = np.random.default_rng(config.bootstrap_seed + 93000)
    boot_rows = []
    for b in range(config.n_bootstrap):
        sampled = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        idx = np.concatenate([groups[cid] for cid in sampled])
        boot_mean = response.loc[idx].groupby(["theta_reg", "theta_stress"])[prob_cols].mean().reset_index()
        boot_mean["bootstrap"] = b
        boot_rows.append(boot_mean)
    boot = pd.concat(boot_rows, ignore_index=True)
    intervals = []
    for theta, group in boot.groupby(["theta_reg", "theta_stress"]):
        row = {"theta_reg": theta[0], "theta_stress": theta[1]}
        for col in prob_cols:
            vals = group[col].to_numpy(dtype=float)
            interval = summarize_interval(vals)
            row[f"{col}_bootstrap_mean"] = interval["mean"]
            row[f"{col}_ci_lower"] = interval["lower"]
            row[f"{col}_ci_upper"] = interval["upper"]
        intervals.append(row)
    return summary.merge(pd.DataFrame(intervals), on=["theta_reg", "theta_stress"], how="left")


def _prob_row(prob_by_context: pd.DataFrame, theta: tuple[float, float]) -> pd.DataFrame:
    rows = prob_by_context[
        np.isclose(prob_by_context["theta_reg"], theta[0])
        & np.isclose(prob_by_context["theta_stress"], theta[1])
    ].copy()
    if "analysis_design" in rows.columns:
        rows = rows[rows["analysis_design"] == "local_fisher"].copy()
    return rows


def _fisher_for_context_table(prob_by_context: pd.DataFrame, config: CorrectedAuditConfig, context_index: np.ndarray | None = None) -> pd.DataFrame:
    points = [(0.0, 0.0), (-0.5, 0.0), (0.5, 0.0), (0.0, -0.5), (0.0, 0.5)]
    h = config.fisher_step
    labels = PHENOTYPE_LABELS
    prob_cols = [f"p_{label}" for label in labels]
    if context_index is not None:
        keep = set(map(int, context_index))
        prob_by_context = prob_by_context[prob_by_context["context_id"].isin(keep)].copy()
    rows = []
    for theta in points:
        base = _prob_row(prob_by_context, theta).set_index("context_id")
        plus_r = _prob_row(prob_by_context, (theta[0] + h, theta[1])).set_index("context_id")
        minus_r = _prob_row(prob_by_context, (theta[0] - h, theta[1])).set_index("context_id")
        plus_s = _prob_row(prob_by_context, (theta[0], theta[1] + h)).set_index("context_id")
        minus_s = _prob_row(prob_by_context, (theta[0], theta[1] - h)).set_index("context_id")
        common = base.index.intersection(plus_r.index).intersection(minus_r.index).intersection(plus_s.index).intersection(minus_s.index)
        matrices = []
        for context_id in common:
            p0 = _smooth_probabilities(base.loc[context_id, prob_cols].to_numpy(dtype=float), config.n_intervention_draws, config.intervention_probability_smoothing_alpha)
            pr_plus = _smooth_probabilities(plus_r.loc[context_id, prob_cols].to_numpy(dtype=float), config.n_intervention_draws, config.intervention_probability_smoothing_alpha)
            pr_minus = _smooth_probabilities(minus_r.loc[context_id, prob_cols].to_numpy(dtype=float), config.n_intervention_draws, config.intervention_probability_smoothing_alpha)
            ps_plus = _smooth_probabilities(plus_s.loc[context_id, prob_cols].to_numpy(dtype=float), config.n_intervention_draws, config.intervention_probability_smoothing_alpha)
            ps_minus = _smooth_probabilities(minus_s.loc[context_id, prob_cols].to_numpy(dtype=float), config.n_intervention_draws, config.intervention_probability_smoothing_alpha)
            dp_reg = (pr_plus - pr_minus) / (2.0 * h)
            dp_stress = (ps_plus - ps_minus) / (2.0 * h)
            grad = np.vstack([dp_reg, dp_stress])
            matrices.append(grad @ np.diag(1.0 / p0) @ grad.T)
        fmat = _sym(np.mean(matrices, axis=0))
        eig = np.linalg.eigvalsh(fmat)
        rows.extend(
            [
                {"theta_reg": theta[0], "theta_stress": theta[1], "row_component": "theta_reg", "col_component": "theta_reg", "fisher_information": float(fmat[0, 0]), "eigenvalue_min": float(eig[0]), "eigenvalue_max": float(eig[-1]), "n_contexts": len(common), "positive_semidefinite": bool(eig[0] >= -1e-10)},
                {"theta_reg": theta[0], "theta_stress": theta[1], "row_component": "theta_reg", "col_component": "theta_stress", "fisher_information": float(fmat[0, 1]), "eigenvalue_min": float(eig[0]), "eigenvalue_max": float(eig[-1]), "n_contexts": len(common), "positive_semidefinite": bool(eig[0] >= -1e-10)},
                {"theta_reg": theta[0], "theta_stress": theta[1], "row_component": "theta_stress", "col_component": "theta_reg", "fisher_information": float(fmat[1, 0]), "eigenvalue_min": float(eig[0]), "eigenvalue_max": float(eig[-1]), "n_contexts": len(common), "positive_semidefinite": bool(eig[0] >= -1e-10)},
                {"theta_reg": theta[0], "theta_stress": theta[1], "row_component": "theta_stress", "col_component": "theta_stress", "fisher_information": float(fmat[1, 1]), "eigenvalue_min": float(eig[0]), "eigenvalue_max": float(eig[-1]), "n_contexts": len(common), "positive_semidefinite": bool(eig[0] >= -1e-10)},
            ]
        )
    return pd.DataFrame(rows)


def corrected_fisher_information(prob_by_context: pd.DataFrame, config: CorrectedAuditConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "analysis_design" in prob_by_context.columns:
        prob_by_context = prob_by_context[prob_by_context["analysis_design"] == "local_fisher"].copy()
    points = [(0.0, 0.0), (-0.5, 0.0), (0.5, 0.0), (0.0, -0.5), (0.0, 0.5)]
    h = config.fisher_step
    prob_cols = [f"p_{label}" for label in PHENOTYPE_LABELS]
    context_meta = prob_by_context[["context_id", "seed", "lineage_id", "tau"]].drop_duplicates().sort_values("context_id")
    context_ids = context_meta["context_id"].to_numpy(dtype=int)
    context_pos = {int(cid): i for i, cid in enumerate(context_ids)}
    theta_prob: dict[tuple[float, float], np.ndarray] = {}
    for theta, group in prob_by_context.groupby(["theta_reg", "theta_stress"]):
        arr = np.zeros((len(context_ids), len(PHENOTYPE_LABELS)), dtype=float)
        for _, row in group.iterrows():
            arr[context_pos[int(row["context_id"])]] = row[prob_cols].to_numpy(dtype=float)
        theta_prob[(float(theta[0]), float(theta[1]))] = _smooth_probabilities(
            arr,
            config.n_intervention_draws,
            config.intervention_probability_smoothing_alpha,
        )

    def context_fisher_matrices(theta: tuple[float, float]) -> np.ndarray:
        p0 = theta_prob[(theta[0], theta[1])]
        pr_plus = theta_prob[(theta[0] + h, theta[1])]
        pr_minus = theta_prob[(theta[0] - h, theta[1])]
        ps_plus = theta_prob[(theta[0], theta[1] + h)]
        ps_minus = theta_prob[(theta[0], theta[1] - h)]
        dp_reg = (pr_plus - pr_minus) / (2.0 * h)
        dp_stress = (ps_plus - ps_minus) / (2.0 * h)
        grad = np.stack([dp_reg, dp_stress], axis=1)
        return np.einsum("cav,cbv,cv->cab", grad, grad, 1.0 / p0)

    fisher_by_theta = {theta: context_fisher_matrices(theta) for theta in points}
    observed_rows = []
    for theta, mats in fisher_by_theta.items():
        fmat = _sym(mats.mean(axis=0))
        eig = np.linalg.eigvalsh(fmat)
        observed_rows.extend(
            [
                {"theta_reg": theta[0], "theta_stress": theta[1], "row_component": "theta_reg", "col_component": "theta_reg", "fisher_information": float(fmat[0, 0]), "eigenvalue_min": float(eig[0]), "eigenvalue_max": float(eig[-1]), "n_contexts": mats.shape[0], "positive_semidefinite": bool(eig[0] >= -1e-10)},
                {"theta_reg": theta[0], "theta_stress": theta[1], "row_component": "theta_reg", "col_component": "theta_stress", "fisher_information": float(fmat[0, 1]), "eigenvalue_min": float(eig[0]), "eigenvalue_max": float(eig[-1]), "n_contexts": mats.shape[0], "positive_semidefinite": bool(eig[0] >= -1e-10)},
                {"theta_reg": theta[0], "theta_stress": theta[1], "row_component": "theta_stress", "col_component": "theta_reg", "fisher_information": float(fmat[1, 0]), "eigenvalue_min": float(eig[0]), "eigenvalue_max": float(eig[-1]), "n_contexts": mats.shape[0], "positive_semidefinite": bool(eig[0] >= -1e-10)},
                {"theta_reg": theta[0], "theta_stress": theta[1], "row_component": "theta_stress", "col_component": "theta_stress", "fisher_information": float(fmat[1, 1]), "eigenvalue_min": float(eig[0]), "eigenvalue_max": float(eig[-1]), "n_contexts": mats.shape[0], "positive_semidefinite": bool(eig[0] >= -1e-10)},
            ]
        )
    observed = pd.DataFrame(observed_rows)
    context_meta = context_meta.copy()
    context_meta["cluster_id"] = context_meta["seed"].astype(str) + "_" + context_meta["lineage_id"].astype(str)
    groups = context_meta.groupby("cluster_id").indices
    cluster_ids = np.array(list(groups.keys()), dtype=object)
    rng = np.random.default_rng(config.bootstrap_seed + 99000)
    boot_rows = []
    for b in range(config.n_bootstrap):
        sampled = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        idx = np.concatenate([np.asarray(groups[cid], dtype=int) for cid in sampled])
        for theta, mats in fisher_by_theta.items():
            fmat = _sym(mats[idx].mean(axis=0))
            eig = np.linalg.eigvalsh(fmat)
            boot_rows.extend(
                [
                    {"bootstrap": b, "theta_reg": theta[0], "theta_stress": theta[1], "row_component": "theta_reg", "col_component": "theta_reg", "fisher_information": float(fmat[0, 0]), "eigenvalue_min": float(eig[0]), "eigenvalue_max": float(eig[-1])},
                    {"bootstrap": b, "theta_reg": theta[0], "theta_stress": theta[1], "row_component": "theta_reg", "col_component": "theta_stress", "fisher_information": float(fmat[0, 1]), "eigenvalue_min": float(eig[0]), "eigenvalue_max": float(eig[-1])},
                    {"bootstrap": b, "theta_reg": theta[0], "theta_stress": theta[1], "row_component": "theta_stress", "col_component": "theta_reg", "fisher_information": float(fmat[1, 0]), "eigenvalue_min": float(eig[0]), "eigenvalue_max": float(eig[-1])},
                    {"bootstrap": b, "theta_reg": theta[0], "theta_stress": theta[1], "row_component": "theta_stress", "col_component": "theta_stress", "fisher_information": float(fmat[1, 1]), "eigenvalue_min": float(eig[0]), "eigenvalue_max": float(eig[-1])},
                ]
            )
    boot = pd.DataFrame(boot_rows)
    interval_rows = []
    keys = ["theta_reg", "theta_stress", "row_component", "col_component"]
    for key, group in boot.groupby(keys):
        vals = group["fisher_information"].to_numpy(dtype=float)
        interval = summarize_interval(vals)
        interval_rows.append(
            {
                "theta_reg": key[0],
                "theta_stress": key[1],
                "row_component": key[2],
                "col_component": key[3],
                "bootstrap_mean": interval["mean"],
                "ci_lower": interval["lower"],
                "ci_upper": interval["upper"],
            }
        )
    observed = observed.merge(pd.DataFrame(interval_rows), on=keys, how="left")
    return observed, boot


def run_stability_corrected(
    seed_results: Sequence[Mapping[str, object]],
    config: CorrectedAuditConfig,
    output_dir: Path,
    *,
    jitter: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, boot_parts, null_parts, seed_parts = [], [], [], []
    for rho in range(1, 6):
        print(f"corrected stability rho={rho}", flush=True)
        table = build_horizon_table(seed_results, config, rho)
        table["meta"] = table["meta"].reset_index(drop=True)
        family = CmiFamily(
            f"intergenerational_stability_rho_{rho}",
            "target_remainder_without_epigenetic",
            "history_remainder_without_epigenetic",
            (SourceBlock("epigenetic", "source_epigenetic"),),
            ((f"R_epigenetic_rho_{rho}", ("epigenetic",)),),
            "propagated predictive contribution under the stated source-excluded conditioning structure",
        )
        joint, slices, source_arrays = _ranked_joint_arrays(table, family, include_generation=True)
        observed = _family_estimates_from_cov(_cov(joint, jitter), slices, family)
        boot = _bootstrap_family(joint, slices, family, table["meta"], n_boot=config.n_bootstrap, seed=config.bootstrap_seed + 52000 + 997 * rho, jitter=jitter)
        null = _null_family(joint, slices, family, table["meta"], n_null=config.n_null, seed=config.null_seed + 52000 + 997 * rho, jitter=jitter)
        summary = _summarize_family(
            family,
            observed,
            boot,
            null,
            target_dim=slices["target"].stop - slices["target"].start,
            source_dims={"epigenetic": source_arrays["epigenetic"].shape[1]},
            condition_dim=slices["condition"].stop - slices["condition"].start,
            jitter=jitter,
        )
        summary.insert(0, "rho", rho)
        summary["n_valid_transitions"] = len(table["meta"])
        summary["source_generation_min"] = int(table["meta"]["tau"].min())
        summary["source_generation_max"] = int(table["meta"]["tau"].max())
        rows.append(summary)
        boot["rho"] = rho
        null["rho"] = rho
        boot_parts.append(boot)
        null_parts.append(null)
        seed_rows = []
        for seed in config.seeds:
            seed_table = {k: v for k, v in table.items()}
            mask = table["meta"]["seed"].to_numpy(dtype=int) == seed
            seed_table["meta"] = table["meta"].loc[mask].reset_index(drop=True)
            for key in ("source_epigenetic", "target_remainder_without_epigenetic", "history_remainder_without_epigenetic"):
                seed_table[key] = table[key][mask]
            seed_joint, seed_slices, _ = _ranked_joint_arrays(seed_table, family, include_generation=True)
            vals = _family_estimates_from_cov(_cov(seed_joint, jitter), seed_slices, family)
            seed_rows.append({"seed": seed, "rho": rho, "analysis": f"R_epigenetic_rho_{rho}", "raw_estimate_bits": vals[f"R_epigenetic_rho_{rho}"]})
        seed_parts.append(pd.DataFrame(seed_rows))
    summary_df = pd.concat(rows, ignore_index=True)
    boot_df = pd.concat(boot_parts, ignore_index=True)
    null_df = pd.concat(null_parts, ignore_index=True)
    seed_df = pd.concat(seed_parts, ignore_index=True)
    summary_df.to_csv(output_dir / "corrected_intergenerational_stability_summary.csv", index=False)
    boot_df.to_csv(output_dir / "corrected_intergenerational_stability_bootstrap.csv", index=False)
    null_df.to_csv(output_dir / "corrected_intergenerational_stability_generation_preserving_null.csv", index=False)
    seed_df.to_csv(output_dir / "corrected_intergenerational_stability_by_seed.csv", index=False)
    return summary_df, boot_df, null_df, seed_df


def run_location_corrected(
    table: Mapping[str, object],
    config: CorrectedAuditConfig,
    output_dir: Path,
    *,
    jitter: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, boot_parts, null_parts = [], [], []
    condition = np.asarray(table["history_remainder_without_epigenetic"])
    source = np.asarray(table["source_epigenetic"])
    meta = table["meta"].reset_index(drop=True)
    source_results = table["seed_results"]
    target_levels = ("development", "microbiome", "life_history", "ecological")
    for target_level in target_levels:
        print(f"corrected location target={target_level}", flush=True)
        arrs = []
        for seed_data in source_results:
            for tau in range(config.source_tau_start, config.source_tau_stop + 1):
                arrs.append(np.asarray(seed_data["full_time_series"][target_level])[:, tau + 1])
        target_all = np.vstack(arrs)
        timestamps = np.asarray(source_results[0]["timestamps"][target_level], dtype=float)
        selected_indices = set(np.asarray(source_results[0]["segment_indices"][target_level], dtype=int).tolist())
        for t_l in range(target_all.shape[1]):
            analysis = f"epigenetic_to_{target_level}_t_{t_l}"
            local_table = {
                "meta": meta,
                "target": target_all[:, t_l, :],
                "source": source,
                "condition": condition,
            }
            family = CmiFamily(
                analysis,
                "target",
                "condition",
                (SourceBlock("epigenetic", "source"),),
                ((analysis, ("epigenetic",)),),
                f"level- and time-specific transfer entropy from the epigenetic segment to {target_level} full-series time {t_l}",
            )
            joint, slices, source_arrays = _ranked_joint_arrays(local_table, family, include_generation=True)
            observed = _family_estimates_from_cov(_cov(joint, jitter), slices, family)
            boot = _bootstrap_family(joint, slices, family, meta, n_boot=config.n_bootstrap, seed=config.bootstrap_seed + 72000 + len(rows), jitter=jitter)
            null = _null_family(joint, slices, family, meta, n_null=config.n_null, seed=config.null_seed + 72000 + len(rows), jitter=jitter)
            summary = _summarize_family(
                family,
                observed,
                boot,
                null,
                target_dim=slices["target"].stop - slices["target"].start,
                source_dims={"epigenetic": source_arrays["epigenetic"].shape[1]},
                condition_dim=slices["condition"].stop - slices["condition"].start,
                jitter=jitter,
            )
            summary["source_level"] = "epigenetic"
            summary["target_level"] = target_level
            summary["target_t_l"] = t_l
            summary["target_u"] = float(timestamps[t_l])
            summary["target_full_length_T_l"] = int(target_all.shape[1])
            summary["target_time_is_in_selected_segment"] = bool(t_l in selected_indices)
            rows.append(summary)
            boot["target_level"] = target_level
            boot["target_t_l"] = t_l
            boot["target_u"] = float(timestamps[t_l])
            null["target_level"] = target_level
            null["target_t_l"] = t_l
            null["target_u"] = float(timestamps[t_l])
            boot_parts.append(boot)
            null_parts.append(null)
    summary_df = pd.concat(rows, ignore_index=True)
    boot_df = pd.concat(boot_parts, ignore_index=True)
    null_df = pd.concat(null_parts, ignore_index=True)
    summary_df.to_csv(output_dir / "corrected_level_time_specific_transfer_entropy_summary.csv", index=False)
    boot_df.to_csv(output_dir / "corrected_level_time_specific_transfer_entropy_bootstrap.csv", index=False)
    null_df.to_csv(output_dir / "corrected_level_time_specific_transfer_entropy_generation_preserving_null.csv", index=False)
    return summary_df, boot_df, null_df


def _residualize_ols(values: np.ndarray, condition: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    values = _as2d(values)
    condition = _as2d(condition)
    design = np.column_stack([np.ones(condition.shape[0]), condition])
    gram = design.T @ design + ridge * np.eye(design.shape[1])
    coef = np.linalg.solve(gram, design.T @ values)
    return values - design @ coef


def conditional_residuals_for_pid(
    target: np.ndarray,
    source1: np.ndarray,
    source2: np.ndarray,
    condition: np.ndarray,
    meta: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gen = generation_design(meta)
    condition_aug = np.hstack([_as2d(condition), gen]) if gen.size else _as2d(condition)
    cr = gc_rank(condition_aug)
    return (
        _residualize_ols(gc_rank(_as2d(target)), cr),
        _residualize_ols(gc_rank(_as2d(source1)), cr),
        _residualize_ols(gc_rank(_as2d(source2)), cr),
    )


def _gaussian_mi_from_residuals(target: np.ndarray, source: np.ndarray, jitter: float) -> float:
    joint = np.hstack([_as2d(target), _as2d(source)])
    cov = _cov(joint, jitter)
    y = np.arange(target.shape[1], dtype=int)
    s = np.arange(target.shape[1], target.shape[1] + source.shape[1], dtype=int)
    return 0.5 * (
        _logdet_signed_spd(cov[np.ix_(y, y)])
        + _logdet_signed_spd(cov[np.ix_(s, s)])
        - _logdet_signed_spd(cov)
    ) / np.log(2.0)


def _run_delta_g_pid(target_r: np.ndarray, source1_r: np.ndarray, source2_r: np.ndarray, config: CorrectedAuditConfig) -> dict[str, object]:
    delta_g_pid = _load_delta_g_pid()
    pid = delta_g_pid(target_r, source1_r, source2_r, rank_transform=False, bias_correct=False, max_iter=config.pid_max_iter)
    atom_sum = pid["RI"] + pid["UI_X"] + pid["UI_Y"] + pid["SI"]
    return {
        "matched_joint_information_bits": float(pid["I_MXY"]),
        "source_1_information_bits": float(pid["I_MX"]),
        "source_2_information_bits": float(pid["I_MY"]),
        "redundancy_bits": float(pid["RI"]),
        "unique_source_1_bits": float(pid["UI_X"]),
        "unique_source_2_bits": float(pid["UI_Y"]),
        "synergy_bits": float(pid["SI"]),
        "atom_sum_bits": float(atom_sum),
        "absolute_reconstruction_error_bits": float(abs(atom_sum - pid["I_MXY"])),
        "within_tolerance": bool(abs(atom_sum - pid["I_MXY"]) <= 1e-6),
        "target_dimension": int(target_r.shape[1]),
        "source_1_dimension": int(source1_r.shape[1]),
        "source_2_dimension": int(source2_r.shape[1]),
        "n": int(target_r.shape[0]),
        "solver_successful_starts": 1,
        "solver_starts": 1,
        "solver_convergence_status": "reference projected-gradient solver returned finite deficiencies",
        "delta_source_1_given_source_2_bits": float(pid["delta_X_given_Y"]),
        "delta_source_2_given_source_1_bits": float(pid["delta_Y_given_X"]),
    }


def _pid_bootstrap(
    target_r: np.ndarray,
    source1_r: np.ndarray,
    source2_r: np.ndarray,
    meta: pd.DataFrame,
    config: CorrectedAuditConfig,
    *,
    seed: int,
    label: str,
) -> pd.DataFrame:
    groups = unit_index_groups(meta)
    unit_ids = np.array(list(groups.keys()), dtype=int)
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(config.n_pid_bootstrap):
        sampled = rng.choice(unit_ids, size=len(unit_ids), replace=True)
        idx = np.concatenate([groups[int(uid)] for uid in sampled])
        pid = _run_delta_g_pid(target_r[idx], source1_r[idx], source2_r[idx], config)
        pid["bootstrap"] = b
        pid["pid"] = label
        rows.append(pid)
    return pd.DataFrame(rows)


def _centered_pid_interval(observed: pd.DataFrame, boot: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "matched_joint_information_bits",
        "source_1_information_bits",
        "source_2_information_bits",
        "redundancy_bits",
        "unique_source_1_bits",
        "unique_source_2_bits",
        "synergy_bits",
    ]
    rows = []
    for col in cols:
        vals = boot[col].to_numpy(dtype=float)
        obs = float(observed[col].iloc[0])
        se = float(np.std(vals, ddof=1))
        z = 1.959963984540054
        rows.append(
            {
                "quantity": col,
                "observed": obs,
                "bootstrap_mean": float(np.mean(vals)),
                "bootstrap_bias": float(np.mean(vals) - obs),
                "bootstrap_se": se,
                "ci_method": "centered standard-error interval because percentile bootstrap was biased upward",
                "ci_lower": obs - z * se,
                "ci_upper": obs + z * se,
            }
        )
    return pd.DataFrame(rows)


def run_pid_corrected(
    table: Mapping[str, object],
    seed_tables: Sequence[Mapping[str, object]],
    config: CorrectedAuditConfig,
    output_dir: Path,
    *,
    prefix: str,
    jitter: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if prefix == "epigenetic_ecological":
        target_key = "target_remainder_without_epigenetic_ecological"
        source1_key = "source_epigenetic"
        source2_key = "source_ecological"
        condition_key = "history_remainder_without_epigenetic_ecological"
        source1_name = "epigenetic_hereditary_factor"
        source2_name = "ecological_candidate_hereditary_factor"
    elif prefix == "multiple_parent":
        target_key = "target_remainder_without_epigenetic"
        source1_key = "source_epigenetic"
        source2_key = "source_epigenetic_second_parent"
        condition_key = "history_remainder_without_epigenetic"
        source1_name = "epigenetic_factor_from_d"
        source2_name = "epigenetic_factor_from_d_prime"
    else:
        raise ValueError(prefix)
    print(f"corrected PID {prefix}", flush=True)
    tr, s1r, s2r = conditional_residuals_for_pid(
        table[target_key],
        table[source1_key],
        table[source2_key],
        table[condition_key],
        table["meta"],
    )
    observed = _run_delta_g_pid(tr, s1r, s2r, config)
    observed["pid"] = prefix
    observed["source_1"] = source1_name
    observed["source_2"] = source2_name
    observed["conditional_preprocessing"] = "Gaussian-copula transform, generation-aware conditioning, and linear residualization before delta_G PID"
    observed["coherent_gaussian_joint_information_bits"] = _gaussian_mi_from_residuals(tr, np.hstack([s1r, s2r]), jitter)
    observed["matched_joint_difference_from_coherent_cmi_bits"] = float(observed["matched_joint_information_bits"] - observed["coherent_gaussian_joint_information_bits"])
    summary = pd.DataFrame([observed])
    boot = _pid_bootstrap(tr, s1r, s2r, table["meta"], config, seed=config.bootstrap_seed + (81000 if prefix == "epigenetic_ecological" else 88000), label=prefix)
    intervals = _centered_pid_interval(summary, boot)
    seed_rows = []
    for seed_table in seed_tables:
        seed = int(seed_table["meta"]["seed"].iloc[0])
        seed_tr, seed_s1, seed_s2 = conditional_residuals_for_pid(
            seed_table[target_key],
            seed_table[source1_key],
            seed_table[source2_key],
            seed_table[condition_key],
            seed_table["meta"],
        )
        row = _run_delta_g_pid(seed_tr, seed_s1, seed_s2, config)
        row["seed"] = seed
        row["pid"] = prefix
        seed_rows.append(row)
    seed_df = pd.DataFrame(seed_rows)
    sens_rows = []
    for lam in config.covariance_jitter_grid:
        sens_rows.append(
            {
                "pid": prefix,
                "common_covariance_jitter": lam,
                "coherent_gaussian_joint_information_bits": _gaussian_mi_from_residuals(tr, np.hstack([s1r, s2r]), lam),
                "coherent_gaussian_source_1_information_bits": _gaussian_mi_from_residuals(tr, s1r, lam),
                "coherent_gaussian_source_2_information_bits": _gaussian_mi_from_residuals(tr, s2r, lam),
            }
        )
    sensitivity = pd.DataFrame(sens_rows)
    summary.to_csv(output_dir / f"corrected_{prefix}_delta_g_pid_summary.csv", index=False)
    boot.to_csv(output_dir / f"corrected_{prefix}_delta_g_pid_bootstrap.csv", index=False)
    intervals.to_csv(output_dir / f"corrected_{prefix}_delta_g_pid_centered_intervals.csv", index=False)
    seed_df.to_csv(output_dir / f"corrected_{prefix}_delta_g_pid_by_seed.csv", index=False)
    sensitivity.to_csv(output_dir / f"corrected_{prefix}_pid_jitter_sensitivity.csv", index=False)
    return summary, boot, seed_df, sensitivity


def derangement_second_parent_pairs(n_lineages: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(900_000 + int(seed))
    for _ in range(10_000):
        perm = rng.permutation(n_lineages)
        if np.all(perm != np.arange(n_lineages)):
            return perm
    raise RuntimeError("failed to generate one-to-one derangement")


def simulate_deranged_multiple_parent_seed(seed: int, config: CorrectedAuditConfig) -> dict[str, object]:
    pilot = config.pilot_config(seed)
    rng = np.random.default_rng(pilot.seed)
    par = _params_for_config(config)
    schedule = build_event_schedule(pilot)
    arrays = _empty_arrays_vec(pilot)
    full_arrays = _empty_full_arrays_vec(pilot)
    reproductive_arrays = _empty_reproductive_arrays_vec(pilot)
    starts = _founder_states_vec(rng, pilot, par)
    retained = _integrate_generation_vec(starts, 0, rng, pilot, par, schedule)
    for key in arrays:
        full_arrays[key][:, 0] = retained[key]
        arrays[key][:, 0] = extract_analysis_segment(retained[key], key, pilot)
        reproductive_arrays[key][:, 0] = reproductive_state(retained[key], key, pilot)
    pairs = derangement_second_parent_pairs(pilot.n_lineages, seed)
    for tau in range(pilot.n_generations):
        prev_reproductive = {key: reproductive_arrays[key][:, tau] for key in arrays}
        start = _next_generation_start_multiparent_vec(prev_reproductive, pairs, rng, pilot, par)
        retained = _integrate_generation_vec(start, tau + 1, rng, pilot, par, schedule)
        for key in arrays:
            full_arrays[key][:, tau + 1] = retained[key]
            arrays[key][:, tau + 1] = extract_analysis_segment(retained[key], key, pilot)
            reproductive_arrays[key][:, tau + 1] = reproductive_state(retained[key], key, pilot)
    return {
        "config": pilot,
        "seed": seed,
        "parameters": par,
        "second_parent_pairs": pairs,
        "full_time_series": full_arrays,
        "reproductive_states": reproductive_arrays,
        "timestamps": level_timestamps(pilot),
        "segment_indices": level_segment_indices(pilot),
        "reproductive_indices": reproductive_indices(pilot),
        **arrays,
    }


def load_deranged_multiple_parent_seed(seed: int, config: CorrectedAuditConfig, output_dir: Path) -> dict[str, object] | None:
    return load_saved_seed(seed, config, output_dir, multiparent=True)


def run_multiple_parent_corrected(
    seed_results_mp: Sequence[Mapping[str, object]],
    config: CorrectedAuditConfig,
    output_dir: Path,
    *,
    jitter: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    table = build_analysis_table(seed_results_mp, config, multiparent=True)
    family = CmiFamily(
        "multiple_parent_epigenetic_contribution",
        "target_remainder_without_epigenetic",
        "history_remainder_without_epigenetic",
        (
            SourceBlock("epigenetic_from_d", "source_epigenetic"),
            SourceBlock("epigenetic_from_d_prime", "source_epigenetic_second_parent"),
        ),
        (
            ("multiple_parent_factor_from_d", ("epigenetic_from_d",)),
            ("multiple_parent_factor_from_d_prime", ("epigenetic_from_d_prime",)),
            ("multiple_parent_joint_factors", ("epigenetic_from_d", "epigenetic_from_d_prime")),
        ),
        "multiple-parent contribution to the receiving individual's future source-excluded ELC",
    )
    joint, slices, source_arrays = _ranked_joint_arrays(table, family, include_generation=True)
    observed = _family_estimates_from_cov(_cov(joint, jitter), slices, family)
    boot = _bootstrap_family(joint, slices, family, table["meta"], n_boot=config.n_bootstrap, seed=config.bootstrap_seed + 101000, jitter=jitter)
    null = _null_family(joint, slices, family, table["meta"], n_null=config.n_null, seed=config.null_seed + 101000, jitter=jitter)
    summary = _summarize_family(
        family,
        observed,
        boot,
        null,
        target_dim=slices["target"].stop - slices["target"].start,
        source_dims={label: source_arrays[label].shape[1] for label in source_arrays},
        condition_dim=slices["condition"].stop - slices["condition"].start,
        jitter=jitter,
    )
    joint_val = float(summary.set_index("analysis").loc["multiple_parent_joint_factors", "raw_estimate_bits"])
    s1 = float(summary.set_index("analysis").loc["multiple_parent_factor_from_d", "raw_estimate_bits"])
    s2 = float(summary.set_index("analysis").loc["multiple_parent_factor_from_d_prime", "raw_estimate_bits"])
    nesting = pd.DataFrame(
        [
            {"check": "joint_ge_factor_from_d", "passes": bool(joint_val + 1e-10 >= s1), "joint_bits": joint_val, "single_bits": s1},
            {"check": "joint_ge_factor_from_d_prime", "passes": bool(joint_val + 1e-10 >= s2), "joint_bits": joint_val, "single_bits": s2},
        ]
    )
    seed_tables = [build_analysis_table([sd], config, multiparent=True) for sd in seed_results_mp]
    seed_df = _seed_level_family(seed_tables, family, jitter=jitter, include_generation=True)
    pairs = table.get("second_parent_pairs")
    if isinstance(pairs, pd.DataFrame):
        pairs.to_csv(output_dir / "multiple_parent_derangement_pairs.csv", index=False)
        pair_summary = pairs.groupby("seed").agg(
            n_receiving=("lineage_id", "count"),
            n_unique_second_parents=("second_parent_lineage_id", "nunique"),
            any_self_pair=("same_individual", "any"),
        ).reset_index()
        pair_summary.to_csv(output_dir / "multiple_parent_derangement_pair_summary.csv", index=False)
    summary.to_csv(output_dir / "corrected_multiple_parent_transfer_entropy_summary.csv", index=False)
    boot.to_csv(output_dir / "corrected_multiple_parent_transfer_entropy_bootstrap.csv", index=False)
    null.to_csv(output_dir / "corrected_multiple_parent_transfer_entropy_generation_preserving_null.csv", index=False)
    seed_df.to_csv(output_dir / "corrected_multiple_parent_transfer_entropy_by_seed.csv", index=False)
    nesting.to_csv(output_dir / "corrected_multiple_parent_nesting_checks.csv", index=False)
    pid_summary, pid_boot, pid_seed, pid_sens = run_pid_corrected(
        table,
        seed_tables,
        config,
        output_dir,
        prefix="multiple_parent",
        jitter=jitter,
    )
    return summary, boot, null, seed_df, pid_summary


def write_corrected_sanity_outputs(
    source_archive: Path,
    output_dir: Path,
    config: CorrectedAuditConfig,
) -> None:
    edges = effective_nonzero_cross_level_edges(_params_for_config(config))
    edges.to_csv(output_dir / "corrected_final_nonzero_cross_level_edges.csv", index=False)
    old_verification = source_archive / "cross_level_coupling_verification.csv"
    if old_verification.exists():
        pd.read_csv(old_verification).to_csv(output_dir / "cross_level_coupling_verification.csv", index=False)


def covariance_diagnostics(table: Mapping[str, object], output_dir: Path) -> pd.DataFrame:
    rows = []
    for key in [
        "source_epigenetic",
        "source_ecological",
        "source_background",
        "target_remainder_without_epigenetic",
        "history_remainder_without_epigenetic",
        "target_full_elc",
        "history_full_elc",
        "target_remainder_without_epigenetic_ecological",
        "history_remainder_without_epigenetic_ecological",
    ]:
        arr = _as2d(table[key])
        cov = _cov(gc_rank(arr), 0.0)
        rows.append(
            {
                "array": key,
                "n": arr.shape[0],
                "dimension": arr.shape[1],
                "matrix_rank": int(np.linalg.matrix_rank(cov)),
                "condition_number": float(np.linalg.cond(cov)),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "corrected_covariance_rank_condition_diagnostics.csv", index=False)
    return df


def write_corrected_figure_specs(output_dir: Path) -> pd.DataFrame:
    rows = [
        ("Figure 1", "Frozen simulated ELC state-space architecture", "ELC levels, source-excluded targets, negative control, and generational ordering", "model specification and corrected coupling table", "schematic axes only", "none", "shows which levels are inside the ELC and which control remains outside"),
        ("Figure 2", "Multivariate simulated dynamics by level", "retained complete state segments for all ELC levels and the background control", "frozen simulated retained state arrays", "generation tau and retained within-generation time t_l", "predetermined illustrative lineages; no median trajectory", "documents the approved frozen biological dynamics"),
        ("Figure 3", "Predictive contribution with generation-preserving null", "raw, null mean, and null-excess transfer entropy estimates", "coherent_continuous_transfer_entropy_summary.csv", "analysis category", "lineage-cluster bootstrap intervals and generation-preserving null distributions", "contrasts the epigenetic hereditary factor with the predictive background control"),
        ("Figure 4", "Predictive closure with generation-preserving null", "transfer entropy from ELC remainder to future source state conditioned on current source state", "coherent_continuous_transfer_entropy_summary.csv", "source state under evaluation", "lineage-cluster bootstrap intervals and generation-preserving null distributions", "shows reconstruction/maintenance contrast without using circular shifts"),
        ("Figure 5", "Four-state phenotype information", "cross-fitted multinomial transfer entropy and state-specific log-probability ratios", "categorical_phenotype_information_summary.csv", "phenotype state or quantity", "lineage-cluster bootstrap intervals", "uses categorical probabilities rather than one-hot Gaussian covariance"),
        ("Figure 6", "Generation-stratified intervention response and Fisher information", "p_theta(V) over four phenotype states and context-averaged Fisher matrices", "intervention_generation_stratified_response_surface_summary.csv and corrected_intervention_fisher_information.csv", "theta_reg and theta_stress", "context/lineage-cluster bootstrap intervals", "separates probability increase from local sensitivity"),
        ("Figure 7", "Intergenerational stability profile", "R^{[l]}_{d,tau}(rho), raw and null-excess", "corrected_intergenerational_stability_summary.csv", "horizon rho", "lineage-cluster bootstrap intervals", "measures propagated predictive contribution under the stated conditioning structure"),
        ("Figure 8", "Level- and time-specific transfer entropy", "T^{[l'->l]}_{d,tau} by target level and retained t_l", "corrected_level_time_specific_transfer_entropy_summary.csv", "actual target retained time t_l in separate target-level panels", "lineage-cluster bootstrap intervals and generation-preserving nulls", "covers development, microbiome, life history, and ecology"),
        ("Figure 9", "Partial Information Decomposition of Epigenetic and Ecological Contributions", "Gaussian-deficiency PID atoms and matched coherent Gaussian joint information", "corrected_epigenetic_ecological_delta_g_pid_summary.csv", "PID atom", "centered lineage-cluster PID intervals", "identifies unique, redundant, and synergistic source contributions"),
        ("Figure 10", "Candidate complex unit from the epigenetic hereditary factor and ecological candidate hereditary factor", "joint predictive contribution, joint closure, and positive PID synergy", "coherent continuous TE and corrected PID files", "criterion or quantity", "bootstrap intervals and null-excess estimates", "does not claim a fully qualified complex unit unless all required criteria are evaluated"),
        ("Figure 11", "Multiple-parent epigenetic contribution", "coherent nested transfer entropy and matched PID for source factors from d and d-prime", "corrected_multiple_parent files", "source comparison", "bootstrap/null intervals", "uses one-to-one deranged second-parent pairing"),
    ]
    df = pd.DataFrame(rows, columns=["figure", "formal_title", "mathematical_quantity", "source_data", "horizontal_axis", "uncertainty_display", "biological_interpretation"])
    df.to_csv(output_dir / "updated_proposed_final_figures_and_captions.csv", index=False)
    return df


def central_contrast_from_summary(summary: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    idx = summary.set_index("analysis")
    epi_pc = idx.loc["epigenetic_predictive_contribution"]
    bg_pc = idx.loc["background_predictive_contribution"]
    epi_closure = idx.loc["epigenetic_predictive_closure"]
    bg_closure = idx.loc["background_closure_control"]
    rows = [
        {
            "criterion": "epigenetic_predictive_contribution_positive_null_excess",
            "passes": bool(epi_pc["null_excess_bits"] > 0.0),
            "raw_bits": float(epi_pc["raw_estimate_bits"]),
            "null_excess_bits": float(epi_pc["null_excess_bits"]),
            "p_value": float(epi_pc["generation_preserving_surrogate_p_value"]),
        },
        {
            "criterion": "background_predictive_contribution_positive_null_excess",
            "passes": bool(bg_pc["null_excess_bits"] > 0.0),
            "raw_bits": float(bg_pc["raw_estimate_bits"]),
            "null_excess_bits": float(bg_pc["null_excess_bits"]),
            "p_value": float(bg_pc["generation_preserving_surrogate_p_value"]),
        },
        {
            "criterion": "epigenetic_predictive_closure_above_generation_preserving_null",
            "passes": bool(epi_closure["generation_preserving_surrogate_p_value"] <= 0.05 and epi_closure["null_excess_bits"] > 0.0),
            "raw_bits": float(epi_closure["raw_estimate_bits"]),
            "null_excess_bits": float(epi_closure["null_excess_bits"]),
            "p_value": float(epi_closure["generation_preserving_surrogate_p_value"]),
        },
        {
            "criterion": "background_closure_control_not_above_generation_preserving_null",
            "passes": bool(bg_closure["generation_preserving_surrogate_p_value"] > 0.05 or bg_closure["null_excess_bits"] <= 0.0),
            "raw_bits": float(bg_closure["raw_estimate_bits"]),
            "null_excess_bits": float(bg_closure["null_excess_bits"]),
            "p_value": float(bg_closure["generation_preserving_surrogate_p_value"]),
        },
        {
            "criterion": "epigenetic_closure_null_excess_exceeds_background_closure_control",
            "passes": bool(epi_closure["null_excess_bits"] > bg_closure["null_excess_bits"]),
            "raw_bits": float(epi_closure["raw_estimate_bits"] - bg_closure["raw_estimate_bits"]),
            "null_excess_bits": float(epi_closure["null_excess_bits"] - bg_closure["null_excess_bits"]),
            "p_value": np.nan,
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "corrected_central_contrast_check.csv", index=False)
    return out


def run_corrected_numerical_audit(
    output_dir: Path,
    source_archive: Path,
    config: CorrectedAuditConfig | None = None,
) -> dict[str, object]:
    t0 = perf_counter()
    config = config or CorrectedAuditConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    write_corrected_sanity_outputs(source_archive, data_dir, config)
    (data_dir / "corrected_final_simulation_configuration.json").write_text(json.dumps(asdict(config), indent=2))
    seed_results = []
    acceptance_rows = []
    for seed in config.seeds:
        data = load_saved_seed(seed, config, source_archive, multiparent=False)
        if data is None:
            raise FileNotFoundError(f"missing frozen seed archive for seed {seed} in {source_archive}")
        seed_results.append(data)
        acc = acceptance_criteria(data, data["config"])
        acc.insert(0, "seed", seed)
        acceptance_rows.append(acc)
    pd.concat(acceptance_rows, ignore_index=True).to_csv(data_dir / "qualitative_acceptance_criteria_by_seed_frozen_model.csv", index=False)
    convergence_check(config.pilot_config(config.seeds[0])).to_csv(data_dir / "integration_step_convergence_confirmation.csv", index=False)
    table = build_analysis_table(seed_results, config)
    table["seed_results"] = seed_results
    table["complex_source_epigenetic_ecological"] = np.hstack([table["source_epigenetic"], table["source_ecological"]])
    seed_tables = [build_analysis_table([sd], config) for sd in seed_results]
    for seed_table in seed_tables:
        seed_table["complex_source_epigenetic_ecological"] = np.hstack([seed_table["source_epigenetic"], seed_table["source_ecological"]])
    phenotype_support(table).to_csv(data_dir / "phenotype_support_by_seed.csv", index=False)
    covariance_diagnostics(table, data_dir)
    families = continuous_families()
    selected_path = data_dir / "selected_common_covariance_jitter.txt"
    if selected_path.exists() and (data_dir / "coherent_continuous_transfer_entropy_summary.csv").exists():
        selected_jitter = float(selected_path.read_text().strip())
        continuous_summary = pd.read_csv(data_dir / "coherent_continuous_transfer_entropy_summary.csv")
        print("loaded existing corrected coherent continuous outputs", flush=True)
    else:
        selected_jitter = choose_common_jitter(table, families, config, data_dir)
        selected_path.write_text(f"{selected_jitter:.12g}\n")
        continuous_summary, continuous_boot, continuous_null, continuous_seed = run_continuous_transfer_entropy(
            table,
            seed_tables,
            config,
            data_dir,
            jitter=selected_jitter,
        )
    if (data_dir / "corrected_central_contrast_check.csv").exists():
        central = pd.read_csv(data_dir / "corrected_central_contrast_check.csv")
    else:
        central = central_contrast_from_summary(continuous_summary, data_dir)
    if not bool(central["passes"].all()):
        print("central contrast failed after the temporal-architecture correction; continuing the required downstream audit without retuning", flush=True)
    if (data_dir / "categorical_phenotype_information_summary.csv").exists():
        print("loaded existing corrected categorical phenotype outputs", flush=True)
    else:
        run_categorical_phenotype_information(table, config, data_dir)
    intervention_metadata = data_dir / "epigenetic_intervention_design_metadata.csv"
    intervention_outputs_current = False
    if intervention_metadata.exists():
        meta_design = pd.read_csv(intervention_metadata)
        if len(meta_design):
            intervention_outputs_current = bool(
                meta_design["mode"].iloc[0] == "additive_shift_to_context_specific_initial_latent_logits"
                and meta_design["zero_point"].iloc[0] == "contextual_no_modification"
            )
    if (
        (data_dir / "intervention_generation_stratified_response_surface_summary.csv").exists()
        and (data_dir / "corrected_intervention_fisher_information.csv").exists()
        and intervention_outputs_current
    ):
        print("loaded existing corrected intervention outputs", flush=True)
    elif (
        (data_dir / "intervention_generation_stratified_probability_by_context.csv").exists()
        and (data_dir / "intervention_generation_stratified_response_surface_summary.csv").exists()
        and intervention_outputs_current
    ):
        print("loaded corrected intervention probabilities; computing corrected Fisher outputs", flush=True)
        prob_by_context = pd.read_csv(data_dir / "intervention_generation_stratified_probability_by_context.csv")
        fisher, fisher_boot = corrected_fisher_information(prob_by_context, config)
        fisher.to_csv(data_dir / "corrected_intervention_fisher_information.csv", index=False)
        fisher_boot.to_csv(data_dir / "corrected_intervention_fisher_information_bootstrap.csv", index=False)
        by_generation_path = data_dir / "intervention_response_by_generation.csv"
        variation_path = data_dir / "intervention_generation_variation_summary.csv"
        if by_generation_path.exists() and not variation_path.exists():
            by_generation = pd.read_csv(by_generation_path)
            variation_rows = []
            for theta, group in by_generation.groupby(["theta_reg", "theta_stress"]):
                values = group["p_nu_early_maturation_high_growth"].to_numpy(dtype=float)
                variation_rows.append(
                    {
                        "theta_reg": theta[0],
                        "theta_stress": theta[1],
                        "min_generation_mean_p_nu": float(values.min()),
                        "max_generation_mean_p_nu": float(values.max()),
                        "range_generation_mean_p_nu": float(values.max() - values.min()),
                    }
                )
            pd.DataFrame(variation_rows).to_csv(variation_path, index=False)
    else:
        run_generation_stratified_intervention(seed_results, config, data_dir)
    if (data_dir / "corrected_intergenerational_stability_summary.csv").exists():
        print("loaded existing corrected stability outputs", flush=True)
    else:
        run_stability_corrected(seed_results, config, data_dir, jitter=selected_jitter)
    if (data_dir / "corrected_level_time_specific_transfer_entropy_summary.csv").exists():
        print("loaded existing corrected level/time outputs", flush=True)
    else:
        run_location_corrected(table, config, data_dir, jitter=selected_jitter)
    if (data_dir / "corrected_epigenetic_ecological_delta_g_pid_summary.csv").exists():
        print("loaded existing corrected epigenetic/ecological PID outputs", flush=True)
    else:
        run_pid_corrected(table, seed_tables, config, data_dir, prefix="epigenetic_ecological", jitter=selected_jitter)
    seed_results_mp = []
    for seed in config.seeds:
        mp = load_deranged_multiple_parent_seed(seed, config, data_dir)
        if mp is None:
            print(f"simulating deranged multiple-parent seed {seed}", flush=True)
            mp = simulate_deranged_multiple_parent_seed(seed, config)
            save_seed_archive(
                data_dir / f"multiple_parent_temporal_architecture_seed_{seed}.npz",
                mp,
                pair_key="second_parent_pairs",
            )
        else:
            print(f"loaded deranged multiple-parent seed {seed}", flush=True)
        seed_results_mp.append(mp)
    if (data_dir / "corrected_multiple_parent_transfer_entropy_summary.csv").exists():
        print("loaded existing corrected multiple-parent outputs", flush=True)
    else:
        run_multiple_parent_corrected(seed_results_mp, config, data_dir, jitter=selected_jitter)
    write_corrected_figure_specs(data_dir)
    runtime = _runtime_record(t0, config)
    (data_dir / "software_versions_and_runtime.json").write_text(json.dumps(runtime, indent=2))
    return {"data_dir": data_dir, "continuous_summary": continuous_summary, "central_contrast": central, "runtime": runtime}


def _runtime_record(t0: float, config: CorrectedAuditConfig) -> dict[str, object]:
    return {
        "runtime_seconds": perf_counter() - t0,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": __import__("sklearn").__version__,
        "simulation_frozen_after_pilot": True,
        "biological_dynamics_retuned": False,
        "continuous_estimator": "generation-conditioned coherent Gaussian-copula CMI with one common covariance jitter",
        "n_bootstrap": config.n_bootstrap,
        "n_null": config.n_null,
        "n_pid_bootstrap": config.n_pid_bootstrap,
    }
