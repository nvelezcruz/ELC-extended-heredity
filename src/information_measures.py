from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.special import ndtri


def as_2d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise AssertionError("information arrays must be two-dimensional")
    if not np.all(np.isfinite(arr)):
        raise FloatingPointError("information array contains non-finite values")
    return arr


def rank_gauss(x: np.ndarray) -> np.ndarray:
    arr = as_2d(x)
    n = arr.shape[0]
    z = np.zeros_like(arr, dtype=float)
    for j in range(arr.shape[1]):
        v = arr[:, j]
        order = np.argsort(v, kind="mergesort")
        sorted_v = v[order]
        bounds = np.r_[0, np.flatnonzero(sorted_v[1:] != sorted_v[:-1]) + 1, n]
        ranks = np.empty(n, dtype=float)
        for a, b in zip(bounds[:-1], bounds[1:]):
            ranks[order[a:b]] = 0.5 * (a + b - 1) + 1.0
        u = np.clip((ranks - 0.5) / n, 1e-6, 1.0 - 1e-6)
        z[:, j] = ndtri(u)
    return z


def covariance(x: np.ndarray, ridge: float = 1e-4, shrink: float = 0.06) -> np.ndarray:
    arr = as_2d(x)
    arr = arr - arr.mean(axis=0, keepdims=True)
    cov = np.cov(arr, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    p = cov.shape[0]
    diag_mean = np.trace(cov) / max(p, 1)
    cov = (1.0 - shrink) * cov + shrink * diag_mean * np.eye(p)
    return cov + ridge * np.eye(p)


def logdet_spd(x: np.ndarray) -> float:
    cov = np.asarray(x, dtype=float)
    sign, value = np.linalg.slogdet(cov)
    if sign <= 0:
        cov = cov + 1e-3 * np.eye(cov.shape[0])
        sign, value = np.linalg.slogdet(cov)
    if sign <= 0:
        raise FloatingPointError("covariance matrix is not positive definite after ridge")
    return float(value)


def gaussian_mi_bits(a: np.ndarray, b: np.ndarray, *, transform: bool = True) -> float:
    A = rank_gauss(a) if transform else as_2d(a)
    B = rank_gauss(b) if transform else as_2d(b)
    AB = np.hstack([A, B])
    value = 0.5 * (
        logdet_spd(covariance(A))
        + logdet_spd(covariance(B))
        - logdet_spd(covariance(AB))
    ) / np.log(2)
    return max(0.0, float(value))


def gaussian_cmi_bits(
    y: np.ndarray,
    source: np.ndarray,
    condition: np.ndarray,
    *,
    transform: bool = True,
) -> float:
    Y = rank_gauss(y) if transform else as_2d(y)
    S = rank_gauss(source) if transform else as_2d(source)
    C = rank_gauss(condition) if transform else as_2d(condition)
    YC = np.hstack([Y, C])
    SC = np.hstack([S, C])
    YSC = np.hstack([Y, S, C])
    value = 0.5 * (
        logdet_spd(covariance(YC))
        + logdet_spd(covariance(SC))
        - logdet_spd(covariance(C))
        - logdet_spd(covariance(YSC))
    ) / np.log(2)
    return max(0.0, float(value))


def gaussian_cmi_bits_with_cached_target_condition(
    y_ranked: np.ndarray,
    source_ranked: np.ndarray,
    condition_ranked: np.ndarray,
    *,
    logdet_y_condition: Optional[float] = None,
    logdet_condition: Optional[float] = None,
) -> float:
    """Gaussian CMI for rank-normalized arrays with reusable target-condition terms."""

    Y = as_2d(y_ranked)
    S = as_2d(source_ranked)
    C = as_2d(condition_ranked)
    logdet_y_condition = (
        logdet_spd(covariance(np.hstack([Y, C])))
        if logdet_y_condition is None
        else float(logdet_y_condition)
    )
    logdet_condition = logdet_spd(covariance(C)) if logdet_condition is None else float(logdet_condition)
    SC = np.hstack([S, C])
    YSC = np.hstack([Y, S, C])
    value = 0.5 * (
        logdet_y_condition
        + logdet_spd(covariance(SC))
        - logdet_condition
        - logdet_spd(covariance(YSC))
    ) / np.log(2)
    return max(0.0, float(value))


def conditional_entropy_bits(y: np.ndarray, condition: np.ndarray, *, transform: bool = True) -> float:
    Y = rank_gauss(y) if transform else as_2d(y)
    C = rank_gauss(condition) if transform else as_2d(condition)
    YC = np.hstack([Y, C])
    d = Y.shape[1]
    value = 0.5 * (
        d * np.log(2 * np.pi * np.e)
        + logdet_spd(covariance(YC))
        - logdet_spd(covariance(C))
    ) / np.log(2)
    return float(value)


def unit_index_groups(meta: pd.DataFrame) -> Dict[int, np.ndarray]:
    groups: Dict[int, np.ndarray] = {}
    for unit_id, idx in meta.groupby("unit_id").groups.items():
        groups[int(unit_id)] = np.asarray(list(idx), dtype=int)
    return groups


def bootstrap_indices_by_unit(
    meta: pd.DataFrame,
    rng: np.random.Generator,
) -> np.ndarray:
    groups = unit_index_groups(meta)
    return bootstrap_indices_from_groups(groups, rng)


def bootstrap_indices_from_groups(
    groups: Mapping[int, np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    units = np.array(list(groups.keys()), dtype=int)
    sampled = rng.choice(units, size=len(units), replace=True)
    return np.concatenate([groups[int(unit)] for unit in sampled])


def bootstrap_cmi(
    table: Mapping[str, object],
    *,
    target_key: str,
    source_key: str,
    condition_key: str,
    n_boot: int = 200,
    seed: int = 12345,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    meta = table["meta"]
    assert isinstance(meta, pd.DataFrame)
    groups = unit_index_groups(meta)
    target = rank_gauss(np.asarray(table[target_key]))
    source = rank_gauss(np.asarray(table[source_key]))
    condition = rank_gauss(np.asarray(table[condition_key]))
    rows = []
    for b in range(n_boot):
        idx = bootstrap_indices_from_groups(groups, rng)
        value = gaussian_cmi_bits(
            target[idx],
            source[idx],
            condition[idx],
            transform=False,
        )
        rows.append({"bootstrap": b, "cmi_bits": value})
    return pd.DataFrame(rows)


def summarize_interval(values: Sequence[float], alpha: float = 0.05) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "lower": float(np.quantile(arr, alpha / 2)),
        "upper": float(np.quantile(arr, 1.0 - alpha / 2)),
    }


def circular_shift_source(
    table: Mapping[str, object],
    source_key: str,
    *,
    seed: int,
    groups: Optional[Mapping[int, np.ndarray]] = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    meta = table["meta"]
    assert isinstance(meta, pd.DataFrame)
    source = np.asarray(table[source_key])
    shifted = source.copy()
    groups = unit_index_groups(meta) if groups is None else groups
    tau_values = meta["tau"].to_numpy()
    for idx in groups.values():
        idx = np.asarray(idx, dtype=int)
        idx = idx[np.argsort(tau_values[idx])]
        if len(idx) < 2:
            continue
        shift = int(rng.integers(1, len(idx)))
        shifted[idx] = source[np.roll(idx, shift)]
    return shifted


def circular_shift_null(
    table: Mapping[str, object],
    *,
    target_key: str,
    source_key: str,
    condition_key: str,
    n_null: int = 200,
    seed: int = 54321,
) -> pd.DataFrame:
    rows = []
    meta = table["meta"]
    assert isinstance(meta, pd.DataFrame)
    groups = unit_index_groups(meta)
    for i in range(n_null):
        shifted = circular_shift_source(table, source_key, seed=seed + i, groups=groups)
        value = gaussian_cmi_bits(
            np.asarray(table[target_key]),
            shifted,
            np.asarray(table[condition_key]),
        )
        rows.append({"null_iteration": i, "cmi_bits": value})
    return pd.DataFrame(rows)


def permutation_p_value(observed: float, null_values: Sequence[float]) -> float:
    null = np.asarray(null_values, dtype=float)
    return float((np.sum(null >= observed) + 1.0) / (len(null) + 1.0))


def probability_table(labels: Sequence[str], categories: Sequence[str], alpha: float = 0.5) -> pd.Series:
    counts = pd.Series(labels).value_counts().reindex(categories, fill_value=0).astype(float)
    probs = (counts + alpha) / (counts.sum() + alpha * len(categories))
    if not np.isclose(float(probs.sum()), 1.0):
        raise AssertionError("probabilities do not normalize")
    return probs
