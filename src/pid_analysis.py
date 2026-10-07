from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import pandas as pd

from .information_measures import rank_gauss


def _load_delta_g_pid():
    module_path = Path(__file__).resolve().parents[1] / "reference_code" / "delta_g_pid.py"
    if not module_path.exists():
        raise FileNotFoundError(
            f"Expected copied Gaussian-deficiency delta_G PID implementation at {module_path}"
        )
    spec = importlib.util.spec_from_file_location("reference_delta_g_pid", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("could not load delta_g_pid module spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.delta_g_pid


def _rank_reduce_block(values: np.ndarray, max_dim: int) -> np.ndarray:
    z = rank_gauss(np.asarray(values, dtype=float))
    z = z - z.mean(axis=0, keepdims=True)
    _, singular, vt = np.linalg.svd(z, full_matrices=False)
    rank = int(np.sum(singular > 1e-10))
    keep = min(int(max_dim), rank)
    if keep <= 0:
        return z[:, :1] * 0.0
    return z @ vt[:keep].T


def _prepare_pid_inputs(M: np.ndarray, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    n = M.shape[0]
    total_dim = M.shape[1] + X.shape[1] + Y.shape[1]
    if total_dim < n - 2:
        return M, X, Y, True
    # Reduce low-sample test inputs without breaking Gaussian nesting identities.
    budget = max(3, (n - 4) // 4)
    Mr = _rank_reduce_block(M, budget)
    Xr = _rank_reduce_block(X, budget)
    Yr = _rank_reduce_block(Y, min(Y.shape[1], max(2, budget // 2)))
    return Mr, Xr, Yr, False


def run_pid_analysis(
    table: Mapping[str, object],
    *,
    target: str = "next_generation_ELC_excluding_epigenetic",
    max_iter: int = 450,
) -> Dict[str, object]:
    delta_g_pid = _load_delta_g_pid()
    if target == "next_generation_ELC_excluding_epigenetic":
        M = np.asarray(table["target_epigenetic_remainder"], dtype=float)
    elif target == "phenotypic_variant_score":
        M = table["meta"]["variant_score"].to_numpy(dtype=float)[:, None]
    else:
        raise ValueError(f"unknown PID target {target!r}")
    X = np.asarray(table["X_history_epigenetic_remainder"], dtype=float)
    Y = np.asarray(table["factor"], dtype=float)
    M_pid, X_pid, Y_pid, use_rank_transform = _prepare_pid_inputs(M, X, Y)
    pid = delta_g_pid(M_pid, X_pid, Y_pid, rank_transform=use_rank_transform, bias_correct=False, max_iter=max_iter)
    atoms_sum = pid["RI"] + pid["UI_X"] + pid["UI_Y"] + pid["SI"]
    reconstruction_error = abs(atoms_sum - pid["I_MXY"])
    summary = pd.DataFrame(
        [
            {
                "target": target,
                "atom": "redundant_shared_information",
                "bits": pid["RI"],
            },
            {
                "target": target,
                "atom": "unique_current_ELC_history",
                "bits": pid["UI_X"],
            },
            {
                "target": target,
                "atom": "unique_focal_hereditary_factor",
                "bits": pid["UI_Y"],
            },
            {
                "target": target,
                "atom": "synergistic_complementary_information",
                "bits": pid["SI"],
            },
        ]
    )
    audit = pd.DataFrame(
        [
            {
                "target": target,
                "I_joint_sources_bits": pid["I_MXY"],
                "sum_pid_atoms_bits": atoms_sum,
                "absolute_reconstruction_error_bits": reconstruction_error,
                "within_tolerance": bool(reconstruction_error <= 1e-6),
                "I_target_current_ELC_bits": pid["I_MX"],
                "I_target_factor_bits": pid["I_MY"],
                "delta_current_given_factor_bits": pid["delta_X_given_Y"],
                "delta_factor_given_current_bits": pid["delta_Y_given_X"],
                "n": pid["n"],
                "dims": repr(pid["dims"]),
                "rank_transform": bool(pid["rank_transform"]),
                "rank_reduced_for_numerical_stability": bool(not use_rank_transform),
                "original_dims": repr((M.shape[1], X.shape[1], Y.shape[1])),
                "bias_corrected": bool(pid["bias_corrected"]),
            }
        ]
    )
    if not bool(audit.loc[0, "within_tolerance"]):
        raise AssertionError("delta_G PID atoms do not reconstruct total joint information")
    return {"summary": summary, "audit": audit, "raw": pid}
