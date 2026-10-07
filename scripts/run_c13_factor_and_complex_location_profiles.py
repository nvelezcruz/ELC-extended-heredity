from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.corrected_numerical_audit import (  # noqa: E402
    CorrectedAuditConfig,
    _family_estimates_from_residual_cov,
    _generation_preserving_permutation_indices,
    _logdet_signed_spd,
    _sym,
    gc_rank,
    generation_design,
)
from src.final_numerical_audit import (  # noqa: E402
    ELC_LEVELS,
    build_analysis_table,
    load_saved_seed,
)


PRIMARY = ROOT / "outputs" / "c13_final_untouched"
SOURCE = PRIMARY / "source_data"
CORRECTED = PRIMARY / "corrected_numerical_audit" / "data"
OUTPUT = PRIMARY / "factor_and_complex_location_profiles"


def _load_config() -> CorrectedAuditConfig:
    raw = json.loads((PRIMARY / "final_configuration.json").read_text())
    allowed = {field.name for field in fields(CorrectedAuditConfig)}
    kwargs = {key: value for key, value in raw.items() if key in allowed}
    for key in ("seeds", "parameter_overrides", "covariance_jitter_grid"):
        if key in kwargs:
            kwargs[key] = tuple(tuple(v) if isinstance(v, list) else v for v in kwargs[key])
    return CorrectedAuditConfig(**kwargs)


def _residualizer(
    condition: np.ndarray, meta: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_raw = np.hstack([condition, generation_design(meta)])
    z = gc_rank(z_raw)
    design = np.column_stack([np.ones(z.shape[0]), z])
    gram_inv = np.linalg.inv(design.T @ design + 1e-8 * np.eye(design.shape[1]))
    return design, gram_inv @ design.T, z


def _residualize(values: np.ndarray, design: np.ndarray, projection_left: np.ndarray) -> np.ndarray:
    ranked = gc_rank(values)
    return ranked - design @ (projection_left @ ranked)


def _exact_cache(
    source_ranked: np.ndarray, condition_ranked: np.ndarray, jitter: float
) -> dict[str, np.ndarray | float]:
    n_minus_one = float(max(len(source_ranked) - 1, 1))
    source_c = source_ranked - source_ranked.mean(axis=0, keepdims=True)
    condition_c = condition_ranked - condition_ranked.mean(axis=0, keepdims=True)
    cov_z = condition_c.T @ condition_c / n_minus_one
    cov_s = source_c.T @ source_c / n_minus_one
    if jitter:
        cov_z = cov_z + jitter * np.eye(cov_z.shape[0])
        cov_s = cov_s + jitter * np.eye(cov_s.shape[0])
    inv_z = np.linalg.inv(cov_z)
    cov_sz = source_c.T @ condition_c / n_minus_one
    source_conditional = _sym(cov_s - cov_sz @ inv_z @ cov_sz.T)
    return {
        "source_centered": source_c,
        "condition_centered": condition_c,
        "inv_condition_cov": inv_z,
        "source_condition_cross": cov_sz,
        "source_conditional_cov": source_conditional,
        "n_minus_one": n_minus_one,
        "jitter": float(jitter),
    }


def _estimate_exact(target_ranked: np.ndarray, cache: dict[str, np.ndarray | float]) -> float:
    target_c = target_ranked - target_ranked.mean(axis=0, keepdims=True)
    source_c = np.asarray(cache["source_centered"])
    condition_c = np.asarray(cache["condition_centered"])
    inv_z = np.asarray(cache["inv_condition_cov"])
    cov_sz = np.asarray(cache["source_condition_cross"])
    s_z = np.asarray(cache["source_conditional_cov"])
    n_minus_one = float(cache["n_minus_one"])
    jitter = float(cache["jitter"])
    cov_y = target_c.T @ target_c / n_minus_one
    if jitter:
        cov_y = cov_y + jitter * np.eye(cov_y.shape[0])
    cov_yz = target_c.T @ condition_c / n_minus_one
    cov_ys = target_c.T @ source_c / n_minus_one
    y_z = _sym(cov_y - cov_yz @ inv_z @ cov_yz.T)
    ys_cross = cov_ys - cov_yz @ inv_z @ cov_sz.T
    ys_z = np.block([[y_z, ys_cross], [ys_cross.T, s_z]])
    return float(
        0.5
        * (
            _logdet_signed_spd(y_z)
            + _logdet_signed_spd(s_z)
            - _logdet_signed_spd(ys_z)
        )
        / np.log(2.0)
    )


def _cluster_indices(meta: pd.DataFrame) -> list[np.ndarray]:
    return [group.index.to_numpy(dtype=int) for _, group in meta.groupby("unit_id", sort=False)]


def _cluster_summaries(joint: np.ndarray, groups: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = joint - joint.mean(axis=0, keepdims=True)
    counts = np.asarray([len(idx) for idx in groups], dtype=float)
    sums = np.asarray([centered[idx].sum(axis=0) for idx in groups], dtype=float)
    crosses = np.asarray([centered[idx].T @ centered[idx] for idx in groups], dtype=float)
    return counts, sums, crosses


def _weighted_cov(
    counts: np.ndarray,
    sums: np.ndarray,
    crosses: np.ndarray,
    weights: np.ndarray,
    jitter: float,
) -> np.ndarray:
    n = float(weights @ counts)
    total = weights @ sums
    cross = np.tensordot(weights, crosses, axes=(0, 0))
    cov = (cross - np.outer(total, total) / max(n, 1.0)) / max(n - 1.0, 1.0)
    if jitter:
        cov = cov + jitter * np.eye(cov.shape[0])
    return _sym(cov)


def _mi_from_small_cov(cov: np.ndarray, target_dim: int) -> float:
    y = np.arange(target_dim)
    s = np.arange(target_dim, cov.shape[0])
    ys = np.r_[y, s]
    return float(
        0.5
        * (
            _logdet_signed_spd(cov[np.ix_(y, y)])
            + _logdet_signed_spd(cov[np.ix_(s, s)])
            - _logdet_signed_spd(cov[np.ix_(ys, ys)])
        )
        / np.log(2.0)
    )


def _bootstrap(
    target: np.ndarray,
    source: np.ndarray,
    groups: list[np.ndarray],
    *,
    n_boot: int,
    seed: int,
    jitter: float,
) -> np.ndarray:
    counts, sums, crosses = _cluster_summaries(np.hstack([target, source]), groups)
    rng = np.random.default_rng(seed)
    probs = np.full(len(groups), 1.0 / len(groups))
    weights = rng.multinomial(len(groups), probs, size=n_boot).astype(float)
    n = weights @ counts
    total = weights @ sums
    cross = np.tensordot(weights, crosses, axes=(1, 0))
    covariances = (
        cross - np.einsum("bi,bj->bij", total, total) / np.maximum(n[:, None, None], 1.0)
    ) / np.maximum(n[:, None, None] - 1.0, 1.0)
    if jitter:
        covariances = covariances + jitter * np.eye(covariances.shape[1])[None, :, :]
    values = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        values[b] = _mi_from_small_cov(_sym(covariances[b]), target.shape[1])
    return values


def _null(
    target: np.ndarray,
    source: np.ndarray,
    meta: pd.DataFrame,
    *,
    n_null: int,
    seed: int,
    jitter: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    target_c = target - target.mean(axis=0, keepdims=True)
    source_c = source - source.mean(axis=0, keepdims=True)
    n = len(meta)
    cov_y = target_c.T @ target_c / (n - 1)
    cov_s = source_c.T @ source_c / (n - 1)
    values = np.empty(n_null, dtype=float)
    for i in range(n_null):
        perm = _generation_preserving_permutation_indices(meta, rng)
        cross = target_c.T @ source_c[perm] / (n - 1)
        cov = np.block([[cov_y, cross], [cross.T, cov_s]])
        if jitter:
            cov = cov + jitter * np.eye(cov.shape[0])
        values[i] = _mi_from_small_cov(_sym(cov), target.shape[1])
    return values


def _target_by_time(seed_results: list[dict[str, object]], config: CorrectedAuditConfig, level: str) -> np.ndarray:
    parts = []
    for seed_data in seed_results:
        full = np.asarray(seed_data["full_time_series"][level])
        for tau in range(config.source_tau_start, config.source_tau_stop + 1):
            parts.append(full[:, tau + 1])
    return np.vstack(parts)


def _run_profile(
    *,
    source_name: str,
    source: np.ndarray,
    condition: np.ndarray,
    seed_results: list[dict[str, object]],
    table: dict[str, object],
    config: CorrectedAuditConfig,
    jitter: float,
    start_seed_offset: int,
    target_levels: tuple[str, ...] = ELC_LEVELS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta = table["meta"].reset_index(drop=True)
    design, projection_left, condition_ranked = _residualizer(condition, meta)
    source_ranked = gc_rank(source)
    source_resid = source_ranked - design @ (projection_left @ source_ranked)
    exact_cache = _exact_cache(source_ranked, condition_ranked, jitter)
    groups = _cluster_indices(meta)
    summaries: list[dict[str, object]] = []
    boot_rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []
    analysis_index = 0
    for level in target_levels:
        target_all = _target_by_time(seed_results, config, level)
        times = np.asarray(seed_results[0]["timestamps"][level], dtype=float)
        for t_l in range(target_all.shape[1]):
            print(f"{source_name}: target={level} t_l={t_l}/{target_all.shape[1]-1}", flush=True)
            target_ranked = gc_rank(target_all[:, t_l, :])
            target_resid = target_ranked - design @ (projection_left @ target_ranked)
            estimate = _estimate_exact(target_ranked, exact_cache)
            boot = _bootstrap(
                target_resid,
                source_resid,
                groups,
                n_boot=config.n_bootstrap,
                seed=config.bootstrap_seed + start_seed_offset + analysis_index,
                jitter=jitter,
            )
            null = _null(
                target_resid,
                source_resid,
                meta,
                n_null=config.n_null,
                seed=config.null_seed + start_seed_offset + analysis_index,
                jitter=jitter,
            )
            se = float(np.std(boot, ddof=1))
            null_mean = float(np.mean(null))
            summaries.append(
                {
                    "source_profile": source_name,
                    "source_levels": "epigenetic" if source_name == "epigenetic_factor" else "epigenetic+ecological",
                    "target_level": level,
                    "target_t_l": t_l,
                    "target_u": float(times[t_l]),
                    "target_dimension": int(target_resid.shape[1]),
                    "source_dimension": int(source_resid.shape[1]),
                    "conditioning_dimension": int(condition_ranked.shape[1]),
                    "raw_estimate_bits": estimate,
                    "bootstrap_se_bits": se,
                    "ci_lower_bits": estimate - 1.959963984540054 * se,
                    "ci_upper_bits": estimate + 1.959963984540054 * se,
                    "randomized_source_mean_bits": null_mean,
                    "information_beyond_randomized_mean_bits": estimate - null_mean,
                    "randomized_source_p_value": float((1 + np.sum(null >= estimate)) / (len(null) + 1)),
                    "n_bootstrap": config.n_bootstrap,
                    "n_randomized": config.n_null,
                }
            )
            boot_rows.extend(
                {
                    "source_profile": source_name,
                    "target_level": level,
                    "target_t_l": t_l,
                    "bootstrap": b,
                    "estimate_bits": value,
                }
                for b, value in enumerate(boot)
            )
            null_rows.extend(
                {
                    "source_profile": source_name,
                    "target_level": level,
                    "target_t_l": t_l,
                    "randomized_iteration": i,
                    "estimate_bits": value,
                }
                for i, value in enumerate(null)
            )
            analysis_index += 1
            pd.DataFrame(summaries).to_csv(OUTPUT / f"{source_name}_location_summary.partial.csv", index=False)
    return pd.DataFrame(summaries), pd.DataFrame(boot_rows), pd.DataFrame(null_rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config = _load_config()
    seed_results = []
    for seed in config.seeds:
        loaded = load_saved_seed(seed, config, SOURCE, multiparent=False)
        if loaded is None:
            raise FileNotFoundError(f"missing saved C13 trajectory array for seed {seed}")
        seed_results.append(loaded)
    table = build_analysis_table(seed_results, config)
    jitter = float((CORRECTED / "selected_common_covariance_jitter.txt").read_text().strip())

    existing_summary = pd.read_csv(
        CORRECTED / "corrected_level_time_specific_transfer_entropy_summary.csv"
    ).rename(
        columns={
            "surrogate_null_mean_bits": "randomized_source_mean_bits",
            "null_excess_bits": "information_beyond_randomized_mean_bits",
            "generation_preserving_surrogate_p_value": "randomized_source_p_value",
        }
    )
    existing_summary["source_profile"] = "epigenetic_factor"
    existing_summary["source_levels"] = "epigenetic"
    epi_extra, epi_extra_boot, epi_extra_null = _run_profile(
        source_name="epigenetic_factor",
        source=np.asarray(table["source_epigenetic"]),
        condition=np.asarray(table["history_remainder_without_epigenetic"]),
        seed_results=seed_results,
        table=table,
        config=config,
        jitter=jitter,
        start_seed_offset=120_000,
        target_levels=("epigenetic",),
    )
    epi_summary = pd.concat([existing_summary, epi_extra], ignore_index=True, sort=False)

    existing_boot = pd.read_csv(
        CORRECTED / "corrected_level_time_specific_transfer_entropy_bootstrap.csv"
    )
    existing_boot["source_profile"] = "epigenetic_factor"
    epi_boot = pd.concat([existing_boot, epi_extra_boot], ignore_index=True, sort=False)
    existing_null = pd.read_csv(
        CORRECTED
        / "corrected_level_time_specific_transfer_entropy_generation_preserving_null.csv"
    ).rename(
        columns={
            "null_iteration": "randomized_iteration",
            "estimate_bits_raw_signed": "estimate_bits",
        }
    )
    existing_null["source_profile"] = "epigenetic_factor"
    epi_null = pd.concat([existing_null, epi_extra_null], ignore_index=True, sort=False)
    epi_summary.to_csv(OUTPUT / "epigenetic_factor_location_summary.csv", index=False)
    epi_boot.to_csv(OUTPUT / "epigenetic_factor_location_bootstrap.csv", index=False)
    epi_null.to_csv(OUTPUT / "epigenetic_factor_location_randomized_source.csv", index=False)

    joint_summary, joint_boot, joint_null = _run_profile(
        source_name="joint_epigenetic_ecological",
        source=np.hstack([table["source_epigenetic"], table["source_ecological"]]),
        condition=np.asarray(table["history_remainder_without_epigenetic_ecological"]),
        seed_results=seed_results,
        table=table,
        config=config,
        jitter=jitter,
        start_seed_offset=220_000,
    )
    joint_summary.to_csv(
        OUTPUT / "joint_epigenetic_ecological_location_summary.csv", index=False
    )
    joint_boot.to_csv(
        OUTPUT / "joint_epigenetic_ecological_location_bootstrap.csv", index=False
    )
    joint_null.to_csv(
        OUTPUT / "joint_epigenetic_ecological_location_randomized_source.csv", index=False
    )

    combined = pd.concat([epi_summary, joint_summary], ignore_index=True, sort=False)
    combined.to_csv(OUTPUT / "factor_and_complex_location_summary.csv", index=False)
    metadata = {
        "primary_trajectory_directory": str(SOURCE.relative_to(ROOT)),
        "seeds": list(config.seeds),
        "history_order": config.history_order,
        "bootstrap_replicates": config.n_bootstrap,
        "randomized_source_replicates": config.n_null,
        "selected_common_covariance_jitter": jitter,
        "target_levels": list(ELC_LEVELS),
        "epigenetic_condition_excludes": ["epigenetic"],
        "joint_condition_excludes": ["epigenetic", "ecological"],
        "note": "Target-level profiles are statistically dependent and are not additive partitions.",
    }
    (OUTPUT / "location_profile_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(combined.groupby(["source_profile", "target_level"]).size(), flush=True)


if __name__ == "__main__":
    main()
