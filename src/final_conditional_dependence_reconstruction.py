from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .corrected_numerical_audit import (
    CorrectedAuditConfig,
    _as2d,
    _cov,
    _logdet_signed_spd,
    _residualize_ols,
    _sym,
    gc_rank,
    generation_design,
)
from .corrected_pilot_model import nonzero_cross_level_edges
from .final_numerical_audit import ELC_LEVELS, BACKGROUND_LEVEL, load_saved_seed


SOURCE_LEVELS = ("development", "microbiome", "life_history", "epigenetic", "ecological", "background")
TARGET_LEVELS = ("development", "microbiome", "life_history", "epigenetic", "ecological", "background")


@dataclass(frozen=True)
class GraphReconstructionConfig(CorrectedAuditConfig):
    graph_null_replicates: int = 500
    graph_alpha: float = 0.01
    graph_null_seed: int = 2026072391
    graph_jitter_grid: tuple[float, ...] = (0.0, 1e-10, 1e-8, 1e-6, 1e-4)


def _seed_meta(seed: int, seed_position: int, n_lineages: int, taus: Sequence[int]) -> pd.DataFrame:
    rows = []
    for tau in taus:
        for d in range(n_lineages):
            rows.append(
                {
                    "seed": int(seed),
                    "lineage_id": int(d),
                    "unit_id": int(seed_position * n_lineages + d),
                    "tau": int(tau),
                }
            )
    return pd.DataFrame(rows)


def _segment(seed_data: Mapping[str, object], level: str, tau: int) -> np.ndarray:
    arr = np.asarray(seed_data[level], dtype=float)
    return arr[:, tau].reshape(arr.shape[0], -1)


def build_graph_reconstruction_table(
    seed_results: Sequence[Mapping[str, object]],
    config: GraphReconstructionConfig,
) -> dict[str, object]:
    taus = list(range(config.source_tau_start, config.source_tau_stop + 1))
    metas: list[pd.DataFrame] = []
    source_parts: dict[str, list[np.ndarray]] = {level: [] for level in SOURCE_LEVELS}
    target_parts: dict[str, list[np.ndarray]] = {level: [] for level in TARGET_LEVELS}
    for seed_position, seed_data in enumerate(seed_results):
        seed = int(seed_data["seed"])
        n = int(seed_data["config"].n_lineages)
        metas.append(_seed_meta(seed, seed_position, n, taus))
        for tau in taus:
            for level in SOURCE_LEVELS:
                source_parts[level].append(_segment(seed_data, level, tau))
            for level in TARGET_LEVELS:
                target_parts[level].append(_segment(seed_data, level, tau + 1))
    return {
        "meta": pd.concat(metas, ignore_index=True),
        "sources": {level: np.vstack(parts) for level, parts in source_parts.items()},
        "targets": {level: np.vstack(parts) for level, parts in target_parts.items()},
    }


def known_dependency_table() -> pd.DataFrame:
    edges = nonzero_cross_level_edges()
    known = {(row.source_level, row.target_level) for row in edges.itertuples(index=False)}
    for level in TARGET_LEVELS:
        known.add((level, level))
    rows = []
    for source in SOURCE_LEVELS:
        for target in TARGET_LEVELS:
            rows.append(
                {
                    "source_level": source,
                    "target_level": target,
                    "known_dependency": int((source, target) in known),
                    "basis": "self-dynamics" if source == target else ("nonzero cross-level coefficient" if (source, target) in known else "not specified"),
                }
            )
    return pd.DataFrame(rows)


def reclassify_graph_dependencies(matrix: pd.DataFrame, config: GraphReconstructionConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute expected/recovered classes from the audited edge inventory.

    This leaves the estimated information values unchanged.  It updates only the
    known-dependency labels after the component-level coupling inventory changes.
    """

    config = GraphReconstructionConfig() if config is None else config
    value_cols = [c for c in matrix.columns if c not in {"known_dependency", "basis", "edge_status"}]
    updated = matrix[value_cols].merge(known_dependency_table(), on=["source_level", "target_level"], how="left")
    updated["known_dependency"] = updated["known_dependency"].fillna(0).astype(int)
    if "recovered_dependency" not in updated:
        updated["recovered_dependency"] = (
            (updated["generation_preserving_surrogate_p_value"] <= config.graph_alpha)
            & (updated["null_excess_bits"] > 0.0)
        ).astype(int)
    else:
        updated["recovered_dependency"] = updated["recovered_dependency"].astype(int)
    updated["edge_status"] = np.select(
        [
            (updated["known_dependency"] == 1) & (updated["recovered_dependency"] == 1),
            (updated["known_dependency"] == 1) & (updated["recovered_dependency"] == 0),
            (updated["known_dependency"] == 0) & (updated["recovered_dependency"] == 1),
        ],
        ["expected_recovered", "expected_missed", "additional_inferred"],
        default="expected_absent",
    )
    tp = int(((updated["known_dependency"] == 1) & (updated["recovered_dependency"] == 1)).sum())
    fp = int(((updated["known_dependency"] == 0) & (updated["recovered_dependency"] == 1)).sum())
    fn = int(((updated["known_dependency"] == 1) & (updated["recovered_dependency"] == 0)).sum())
    tn = int(((updated["known_dependency"] == 0) & (updated["recovered_dependency"] == 0)).sum())
    summary = pd.DataFrame(
        [
            {
                "alpha": float(config.graph_alpha),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "true_negatives": tn,
                "precision": float(tp / max(tp + fp, 1)),
                "recall": float(tp / max(tp + fn, 1)),
                "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
                "exact_recovery": bool(fp == 0 and fn == 0),
                "n_null_per_edge": int(updated["n_null"].iloc[0]) if "n_null" in updated and len(updated) else int(config.graph_null_replicates),
                "estimator": "coherent Gaussian-copula transfer entropy with generation-preserving source permutations",
            }
        ]
    )
    return updated, summary


def _mi_from_residuals(Yr: np.ndarray, Sr: np.ndarray, jitter: float) -> float:
    joint = np.hstack([Yr, Sr])
    cov = _cov(joint, jitter)
    y = np.arange(Yr.shape[1], dtype=int)
    s = np.arange(Yr.shape[1], joint.shape[1], dtype=int)
    ys = np.r_[y, s]
    return 0.5 * (
        _logdet_signed_spd(cov[np.ix_(y, y)])
        + _logdet_signed_spd(cov[np.ix_(s, s)])
        - _logdet_signed_spd(cov[np.ix_(ys, ys)])
    ) / np.log(2.0)


def _mi_from_residual_cov_blocks(cov_y: np.ndarray, cov_s: np.ndarray, cov_ys: np.ndarray, jitter: float) -> float:
    cov = np.zeros((cov_y.shape[0] + cov_s.shape[0], cov_y.shape[0] + cov_s.shape[0]), dtype=float)
    cov[: cov_y.shape[0], : cov_y.shape[0]] = cov_y
    cov[cov_y.shape[0] :, cov_y.shape[0] :] = cov_s
    cov[: cov_y.shape[0], cov_y.shape[0] :] = cov_ys
    cov[cov_y.shape[0] :, : cov_y.shape[0]] = cov_ys.T
    if jitter > 0.0:
        cov = cov + jitter * np.eye(cov.shape[0])
    cov = _sym(cov)
    y = np.arange(cov_y.shape[0], dtype=int)
    s = np.arange(cov_y.shape[0], cov.shape[0], dtype=int)
    ys = np.r_[y, s]
    return 0.5 * (
        _logdet_signed_spd(cov[np.ix_(y, y)])
        + _logdet_signed_spd(cov[np.ix_(s, s)])
        - _logdet_signed_spd(cov[np.ix_(ys, ys)])
    ) / np.log(2.0)


def _select_jitter(Yr: np.ndarray, Sr: np.ndarray, grid: Sequence[float]) -> float:
    for jitter in grid:
        try:
            _ = _mi_from_residuals(Yr, Sr, float(jitter))
            return float(jitter)
        except Exception:
            continue
    raise FloatingPointError("no jitter in graph grid produced a stable covariance calculation")


def estimate_graph_dependencies(
    table: Mapping[str, object],
    config: GraphReconstructionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta = table["meta"]
    assert isinstance(meta, pd.DataFrame)
    sources: Mapping[str, np.ndarray] = table["sources"]
    targets: Mapping[str, np.ndarray] = table["targets"]
    strata = [group.index.to_numpy(dtype=int) for _, group in meta.groupby(["seed", "tau"], sort=False)]

    def generation_preserving_perm(rng: np.random.Generator) -> np.ndarray:
        perm = np.arange(len(meta), dtype=int)
        for idx in strata:
            perm[idx] = rng.permutation(idx)
        return perm

    ranked_sources = {level: gc_rank(_as2d(value)) for level, value in sources.items()}
    ranked_targets = {level: gc_rank(_as2d(value)) for level, value in targets.items()}
    gen = generation_design(meta)
    ranked_gen = gc_rank(gen) if gen.size else np.zeros((len(meta), 0))
    known = known_dependency_table()
    rows = []
    null_rows = []
    pair_counter = 0
    n_pairs = len(TARGET_LEVELS) * len(SOURCE_LEVELS)
    for target in TARGET_LEVELS:
        Y = ranked_targets[target]
        for source in SOURCE_LEVELS:
            pair_counter += 1
            print(f"conditional graph {pair_counter}/{n_pairs}: {source} -> {target}", flush=True)
            S = ranked_sources[source]
            other = [ranked_sources[level] for level in SOURCE_LEVELS if level != source]
            Z = np.hstack(other + ([ranked_gen] if ranked_gen.size else []))
            Yr = _residualize_ols(Y, Z)
            Sr = _residualize_ols(S, Z)
            jitter = _select_jitter(Yr, Sr, config.graph_jitter_grid)
            observed = _mi_from_residuals(Yr, Sr, jitter)
            Yc = Yr - Yr.mean(axis=0, keepdims=True)
            Sc = Sr - Sr.mean(axis=0, keepdims=True)
            n = Yc.shape[0]
            cov_y = Yc.T @ Yc / max(n - 1, 1)
            cov_s = Sc.T @ Sc / max(n - 1, 1)
            rng = np.random.default_rng(config.graph_null_seed + 7919 * TARGET_LEVELS.index(target) + 101 * SOURCE_LEVELS.index(source))
            null_values = []
            for r in range(config.graph_null_replicates):
                perm = generation_preserving_perm(rng)
                cov_ys = Yc.T @ Sc[perm] / max(n - 1, 1)
                value = _mi_from_residual_cov_blocks(cov_y, cov_s, cov_ys, jitter)
                null_values.append(float(value))
                null_rows.append(
                    {
                        "source_level": source,
                        "target_level": target,
                        "null_iteration": int(r),
                        "estimate_bits_raw_signed": float(value),
                    }
                )
            null_values_arr = np.asarray(null_values, dtype=float)
            p_value = float((1.0 + np.sum(null_values_arr >= observed)) / (len(null_values_arr) + 1.0))
            rows.append(
                {
                    "source_level": source,
                    "target_level": target,
                    "target_dimension": int(Y.shape[1]),
                    "source_dimension": int(S.shape[1]),
                    "conditioning_dimension": int(Z.shape[1]),
                    "selected_common_jitter": float(jitter),
                    "raw_transfer_entropy_bits": float(observed),
                    "surrogate_null_mean_bits": float(np.mean(null_values_arr)),
                    "null_excess_bits": float(observed - np.mean(null_values_arr)),
                    "generation_preserving_surrogate_p_value": p_value,
                    "null_min_bits_raw_signed": float(np.min(null_values_arr)),
                    "null_max_bits_raw_signed": float(np.max(null_values_arr)),
                    "null_negative_proportion": float(np.mean(null_values_arr < 0.0)),
                    "n_null": int(len(null_values_arr)),
                }
            )
    matrix = pd.DataFrame(rows).merge(known, on=["source_level", "target_level"], how="left")
    matrix["recovered_dependency"] = (
        (matrix["generation_preserving_surrogate_p_value"] <= config.graph_alpha)
        & (matrix["null_excess_bits"] > 0.0)
    ).astype(int)
    matrix, summary = reclassify_graph_dependencies(matrix, config)
    return matrix, summary, pd.DataFrame(null_rows)


def run_final_conditional_dependence_reconstruction(
    project_root: Path,
    output_dir: Path,
    config: GraphReconstructionConfig | None = None,
) -> dict[str, pd.DataFrame]:
    config = GraphReconstructionConfig() if config is None else config
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = (
        project_root
        / "outputs"
        / "corrected_pre_reproductive_final_untouched"
        / "source_data"
    )
    required = [
        source_dir / f"corrected_temporal_architecture_seed_{int(seed)}.npz"
        for seed in config.seeds
    ]
    if not all(path.exists() for path in required):
        missing = ", ".join(str(path) for path in required if not path.exists())
        raise FileNotFoundError(f"missing corrected temporal-architecture arrays: {missing}")
    seed_results = []
    for seed in config.seeds:
        data = load_saved_seed(int(seed), config, source_dir, multiparent=False)
        if data is None:
            raise FileNotFoundError(f"missing corrected retained arrays for seed {seed}")
        seed_results.append(data)
    table = build_graph_reconstruction_table(seed_results, config)
    source_table_rows = []
    for level, arr in table["sources"].items():
        source_table_rows.append({"kind": "source", "level": level, "n_observations": int(arr.shape[0]), "dimension": int(arr.shape[1])})
    for level, arr in table["targets"].items():
        source_table_rows.append({"kind": "target", "level": level, "n_observations": int(arr.shape[0]), "dimension": int(arr.shape[1])})
    source_table = pd.DataFrame(source_table_rows)
    matrix, summary, null = estimate_graph_dependencies(table, config)
    source_table.to_csv(output_dir / "graph_reconstruction_source_table.csv", index=False)
    known_dependency_table().to_csv(output_dir / "graph_reconstruction_known_dependencies.csv", index=False)
    matrix.to_csv(output_dir / "graph_reconstruction_information_matrix.csv", index=False)
    summary.to_csv(output_dir / "graph_reconstruction_recovery_summary.csv", index=False)
    null.to_csv(output_dir / "graph_reconstruction_generation_preserving_null.csv", index=False)
    return {"source_table": source_table, "matrix": matrix, "summary": summary, "null": null}
