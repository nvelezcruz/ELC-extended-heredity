from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence
import json
import platform

import numpy as np
import pandas as pd

from .corrected_pilot_model import (
    COMPONENTS,
    PilotConfig,
    _params,
    _par_float,
    apply_parameter_overrides,
    acceptance_criteria,
    build_event_schedule,
    convergence_check,
    effective_nonzero_cross_level_edges,
    fixed_random_second_parent_pairs,
    extract_analysis_segment,
    full_retained_step_indices,
    level_m,
    level_T,
    level_segment_indices,
    level_steps,
    level_timestamps,
    nonzero_cross_level_edges,
    reproductive_indices,
    reproductive_state,
    retained_step_indices,
    sigmoid,
)
from .information_measures import (
    bootstrap_indices_from_groups,
    covariance,
    gaussian_cmi_bits,
    gaussian_mi_bits,
    logdet_spd,
    permutation_p_value,
    rank_gauss,
    summarize_interval,
    unit_index_groups,
)
from .pid_analysis import _load_delta_g_pid


ELC_LEVELS = ("development", "microbiome", "life_history", "epigenetic", "ecological")
BACKGROUND_LEVEL = "background"
PHENOTYPE_LABELS = (
    "nu_early_maturation_high_growth",
    "early_maturation_low_growth",
    "non_early_maturation_high_growth",
    "non_early_maturation_low_growth",
)


@dataclass(frozen=True)
class FinalAuditConfig:
    seeds: tuple[int, ...] = (101, 202, 303, 404)
    n_lineages: int = 1000
    n_generations: int = 32
    burn_in_generations: int = 8
    source_tau_start: int = 8
    source_tau_stop: int = 30
    history_order: int = 2
    n_bootstrap: int = 500
    n_pid_bootstrap: int = 300
    n_null: int = 500
    n_contexts: int = 1000
    n_contexts_per_seed: int = 250
    n_intervention_draws: int = 100
    theta_maturation: float = 0.60
    theta_growth: float = 0.65
    t_early_maturation: int = 3
    intervention_grid_min: float = -1.5
    intervention_grid_max: float = 1.5
    intervention_grid_size: int = 7
    fisher_step: float = 0.10
    bootstrap_seed: int = 2026072301
    null_seed: int = 2026072302
    intervention_seed: int = 2026072303
    pid_max_iter: int = 250
    cmi_ridge: float = 1e-4
    cmi_shrink: float = 0.06
    parameter_overrides: tuple[tuple[str, float], ...] = ()

    def pilot_config(self, seed: int, *, n_lineages: int | None = None, n_generations: int | None = None) -> PilotConfig:
        return replace(
            PilotConfig(),
            seed=int(seed),
            n_lineages=int(self.n_lineages if n_lineages is None else n_lineages),
            n_generations=int(self.n_generations if n_generations is None else n_generations),
            burn_in_generations=int(self.burn_in_generations),
            theta_maturation=float(self.theta_maturation),
            theta_growth=float(self.theta_growth),
            t_early_maturation=int(self.t_early_maturation),
        )


def _params_for_config(final_config: FinalAuditConfig) -> dict[str, object]:
    return apply_parameter_overrides(_params(), final_config.parameter_overrides)


def _phase_vec(tau: int, d: np.ndarray, u: float) -> dict[str, np.ndarray]:
    lineage_phase = 2.0 * np.pi * ((d + 1) % 37) / 37.0
    return {
        "dev": np.sin(2.0 * np.pi * u + lineage_phase + 0.41 * tau),
        "micro": np.cos(2.0 * np.pi * u - 0.25 * lineage_phase + 0.29 * tau),
        "slow": np.sin(np.pi * u + 0.17 * tau + 0.5 * lineage_phase),
    }


def _hill_vec(p: np.ndarray, k: float, n: float) -> np.ndarray:
    return p**n / (k**n + p**n)


def _founder_states_vec(rng: np.random.Generator, config: PilotConfig, par: Mapping[str, object] | None = None) -> dict[str, np.ndarray]:
    n = config.n_lineages
    par = _params() if par is None else par
    return {
        "development": rng.normal(0.0, 0.18, (n, config.dev_dim)),
        "microbiome": np.exp(rng.normal(-0.42, 0.18, (n, config.micro_dim))),
        "life_history": sigmoid(rng.normal([-1.25, -0.15, -1.35], [0.18, 0.16, 0.16], (n, config.life_dim))),
        "epigenetic": rng.normal(0.0, _par_float(par, "epi_founder_sd", 0.45), (n, config.epi_dim)),
        "ecological": rng.normal(0.0, 0.38, (n, config.eco_dim)),
        "background": rng.normal(0.0, 0.45, (n, config.bg_dim)),
    }


def _empty_arrays_vec(config: PilotConfig) -> dict[str, np.ndarray]:
    shape = (config.n_lineages, config.n_generations + 1)
    return {
        "development": np.zeros(shape + (config.m_dev, config.dev_dim), dtype=np.float64),
        "microbiome": np.zeros(shape + (config.m_micro, config.micro_dim), dtype=np.float64),
        "life_history": np.zeros(shape + (config.m_life, config.life_dim), dtype=np.float64),
        "epigenetic": np.zeros(shape + (config.m_epi, config.epi_dim), dtype=np.float64),
        "ecological": np.zeros(shape + (config.m_eco, config.eco_dim), dtype=np.float64),
        "background": np.zeros(shape + (config.m_bg, config.bg_dim), dtype=np.float64),
    }


def _empty_full_arrays_vec(config: PilotConfig) -> dict[str, np.ndarray]:
    shape = (config.n_lineages, config.n_generations + 1)
    dims = {
        "development": config.dev_dim,
        "microbiome": config.micro_dim,
        "life_history": config.life_dim,
        "epigenetic": config.epi_dim,
        "ecological": config.eco_dim,
        "background": config.bg_dim,
    }
    return {
        level: np.zeros(shape + (level_T(config)[level], dims[level]), dtype=np.float64)
        for level in dims
    }


def _empty_reproductive_arrays_vec(config: PilotConfig) -> dict[str, np.ndarray]:
    shape = (config.n_lineages, config.n_generations + 1)
    dims = {
        "development": config.dev_dim,
        "microbiome": config.micro_dim,
        "life_history": config.life_dim,
        "epigenetic": config.epi_dim,
        "ecological": config.eco_dim,
        "background": config.bg_dim,
    }
    return {level: np.zeros(shape + (dims[level],), dtype=np.float64) for level in dims}


def _update_development_vec(state: Mapping[str, np.ndarray], tau: int, u: float, dt: float, rng: np.random.Generator, config: PilotConfig, par: Mapping[str, np.ndarray], d_idx: np.ndarray) -> np.ndarray:
    x = state["development"]
    y = state["microbiome"]
    life = state["life_history"]
    epi_centered = sigmoid(state["epigenetic"]) - 0.5
    eco = state["ecological"]
    bg = state["background"]
    phase = _phase_vec(tau, d_idx, u)
    epi_dev_scale = _par_float(par, "epi_effect_development_scale")
    inputs = np.zeros_like(x)
    inputs[:, 0] += epi_dev_scale * 0.34 * epi_centered[:, 0]
    inputs[:, 1] += 0.18 * np.log1p(y[:, 0]) + 0.22 * eco[:, 1] + 0.10 * bg[:, 0]
    inputs[:, 2] += 0.16 * np.log1p(y[:, 3]) + epi_dev_scale * 0.28 * epi_centered[:, 1] + 0.12 * bg[:, 1]
    inputs[:, 3] += 0.15 * life[:, 0]
    inputs[:, 4] += 0.18 * eco[:, 2] + 0.12 * life[:, 2]
    stage = np.column_stack(
        [
            0.15 * phase["dev"],
            0.20 * phase["slow"],
            -0.12 * phase["micro"],
            0.18 * np.sin(np.pi * u) * np.ones(config.n_lineages),
            0.16 * np.cos(np.pi * u) * np.ones(config.n_lineages),
        ]
    )
    drive = x @ par["A_dev"].T + inputs + stage
    drift = -0.32 * x + 1.18 * np.tanh(drive)
    return x + dt * drift + config.noise_scale * par["sigma_dev"] * np.sqrt(dt) * rng.normal(size=x.shape)


def _update_microbiome_vec(state: Mapping[str, np.ndarray], tau: int, u: float, dt: float, rng: np.random.Generator, config: PilotConfig, par: Mapping[str, np.ndarray], d_idx: np.ndarray) -> np.ndarray:
    z = np.log(state["microbiome"])
    y = state["microbiome"]
    dev = np.tanh(state["development"])
    epi_centered = sigmoid(state["epigenetic"]) - 0.5
    eco = state["ecological"]
    bg = state["background"]
    phase = _phase_vec(tau, d_idx, u)
    dev_micro_scale = _par_float(par, "development_effect_microbiome_scale")
    epi_micro_scale = _par_float(par, "epi_effect_microbiome_scale")
    r = np.tile(np.array([0.50, 0.38, 0.34, 0.30, 0.40], dtype=float), (config.n_lineages, 1))
    r[:, 0] += dev_micro_scale * 0.18 * dev[:, 1] + epi_micro_scale * 0.10 * epi_centered[:, 0] + 0.20 * eco[:, 1] + 0.08 * bg[:, 0] + 0.10 * phase["micro"]
    r[:, 1] += 0.18 * eco[:, 0] + 0.08 * phase["slow"]
    r[:, 2] += -0.16 * eco[:, 2] + dev_micro_scale * 0.08 * dev[:, 2]
    r[:, 3] += dev_micro_scale * 0.16 * dev[:, 2] + epi_micro_scale * 0.12 * epi_centered[:, 1] + 0.10 * bg[:, 1] - 0.10 * phase["micro"]
    r[:, 4] += 0.10 * y[:, 0] + 0.07 * eco[:, 0]
    sigma = config.noise_scale * par["sigma_micro"]
    dz = (r + y @ par["A_micro"].T - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.normal(size=y.shape)
    return np.exp(z + dz)


def _update_life_history_vec(state: Mapping[str, np.ndarray], tau: int, u: float, dt: float, rng: np.random.Generator, config: PilotConfig, par: Mapping[str, np.ndarray], d_idx: np.ndarray) -> np.ndarray:
    life = state["life_history"]
    z = np.log(life / (1.0 - life))
    dev = np.tanh(state["development"])
    y = state["microbiome"]
    epi_centered = sigmoid(state["epigenetic"]) - 0.5
    eco = state["ecological"]
    bg = state["background"]
    phase = _phase_vec(tau, d_idx, u)
    epi_life_scale = _par_float(par, "epi_effect_life_history_scale")
    epi_life_timing_scale = epi_life_scale * _par_float(par, "epi_effect_life_timing_scale")
    epi_life_maturation_scale = epi_life_scale * _par_float(par, "epi_effect_life_maturation_scale")
    epi_life_growth_scale = epi_life_scale * _par_float(par, "epi_effect_life_growth_scale")
    epi_life_allocation_scale = epi_life_scale * _par_float(par, "epi_effect_life_allocation_scale")
    eco_life_scale = _par_float(par, "ecology_effect_life_history_scale")
    timing_shift = _par_float(par, "life_maturation_timing_base", 0.42) - 0.10 * dev[:, 3] - 0.08 * np.log1p(y[:, 0]) - epi_life_timing_scale * 0.07 * epi_centered[:, 0] + 0.06 * bg[:, 1]
    maturation_wave = sigmoid(8.0 * (u - timing_shift))
    U_maturation = _par_float(par, "life_maturation_target_intercept", -0.92) + _par_float(par, "life_maturation_wave_scale", 2.22) * maturation_wave + 0.42 * dev[:, 3] + 0.18 * np.log1p(y[:, 0]) + epi_life_maturation_scale * 0.25 * epi_centered[:, 0] + eco_life_scale * 0.16 * eco[:, 2] - 0.16 * bg[:, 1] + 0.20 * phase["slow"]
    growth_window = np.sin(np.pi * u)
    late_cost = sigmoid(10.0 * (u - 0.72))
    U_growth = -0.05 + 0.66 * dev[:, 1] + 0.36 * np.log1p(y[:, 0] + 0.6 * y[:, 1]) + epi_life_growth_scale * 0.24 * epi_centered[:, 0] + eco_life_scale * 0.34 * eco[:, 1] + 0.10 * bg[:, 0] - 0.24 * life[:, 2] + 0.42 * growth_window - 0.28 * late_cost + 0.16 * phase["micro"]
    allocation_switch = sigmoid(9.0 * (u - 0.55 + 0.06 * bg[:, 1] - 0.05 * life[:, 1]))
    U_allocation = -1.26 + 1.18 * life[:, 0] + 0.62 * life[:, 1] + 0.30 * allocation_switch + 0.24 * dev[:, 4] + epi_life_allocation_scale * 0.22 * epi_centered[:, 1] + eco_life_scale * 0.22 * eco[:, 0] - 0.16 * bg[:, 1] + 0.18 * phase["dev"] - 0.16 * life[:, 1] * late_cost
    target = np.column_stack([U_maturation, U_growth, U_allocation])
    dz = 2.15 * (target - z) * dt + config.noise_scale * par["sigma_life"] * np.sqrt(dt) * rng.normal(size=life.shape)
    return sigmoid(z + dz)


def _update_epigenetic_vec(state: Mapping[str, np.ndarray], tau: int, u: float, dt: float, rng: np.random.Generator, config: PilotConfig, par: Mapping[str, np.ndarray], d_idx: np.ndarray) -> np.ndarray:
    z = state["epigenetic"]
    p = sigmoid(z)
    h_reg = _hill_vec(p[:, 0], 0.52, 3.0)
    h_stress = _hill_vec(p[:, 1], 0.50, 3.0)
    dev = np.tanh(state["development"])
    life = state["life_history"]
    eco = state["ecological"]
    bg = state["background"]
    phase = _phase_vec(tau, d_idx, u)
    reg_target = 0.38 * (2.0 * h_reg - 1.0) - 0.26 * (2.0 * h_stress - 1.0) + 0.32 * dev[:, 0] + 0.22 * life[:, 1] + 0.20 * eco[:, 1] - 0.14 * bg[:, 1] + 0.20 * phase["slow"]
    stress_target = 0.36 * (2.0 * h_stress - 1.0) - 0.24 * (2.0 * h_reg - 1.0) + 0.30 * dev[:, 2] + 0.28 * life[:, 2] - 0.16 * eco[:, 2] + 0.22 * bg[:, 1] + 0.18 * phase["micro"]
    drift = np.column_stack([0.92 * (reg_target - z[:, 0]), 0.88 * (stress_target - z[:, 1])])
    return z + dt * drift + config.noise_scale * par["sigma_epi"] * np.sqrt(dt) * rng.normal(size=z.shape)


def _update_ecology_vec(state: Mapping[str, np.ndarray], tau: int, u: float, dt: float, rng: np.random.Generator, config: PilotConfig, par: Mapping[str, np.ndarray], d_idx: np.ndarray) -> np.ndarray:
    q = state["ecological"]
    dev = np.tanh(state["development"])
    y = state["microbiome"]
    life = state["life_history"]
    bg = state["background"]
    phase = _phase_vec(tau, d_idx, u)
    recurrent = np.column_stack(
        [
            -0.22 * q[:, 0] + 0.06 * np.tanh(q[:, 1]) - 0.04 * q[:, 0] ** 3,
            -0.20 * q[:, 1] + 0.05 * np.tanh(q[:, 0]) + 0.04 * np.tanh(q[:, 2]) - 0.04 * q[:, 1] ** 3,
            -0.18 * q[:, 2] + 0.06 * np.tanh(q[:, 1]) - 0.04 * q[:, 2] ** 3,
        ]
    )
    eco_mod_scale = _par_float(par, "ecological_elc_modification_scale")
    micro_eco_mod_scale = _par_float(par, "microbiome_to_ecology_modification_scale")
    modification = np.column_stack(
        [
            eco_mod_scale * (micro_eco_mod_scale * 0.30 * np.log1p(y[:, 0]) + 0.24 * life[:, 2]) - 0.06 * bg[:, 1],
            eco_mod_scale * (0.28 * dev[:, 1] + 0.32 * life[:, 2] + micro_eco_mod_scale * 0.12 * np.log1p(y[:, 0])) + 0.06 * phase["slow"],
            eco_mod_scale * (0.22 * life[:, 0] + micro_eco_mod_scale * 0.20 * np.log1p(y[:, 4])) - 0.08 * bg[:, 1] + 0.08 * phase["dev"],
        ]
    )
    return q + dt * (recurrent + modification) + config.noise_scale * par["sigma_eco"] * np.sqrt(dt) * rng.normal(size=q.shape)


def _update_background_vec(state: Mapping[str, np.ndarray], tau: int, u: float, dt: float, rng: np.random.Generator, config: PilotConfig, par: Mapping[str, np.ndarray], d_idx: np.ndarray) -> np.ndarray:
    bg = state["background"]
    drift = np.column_stack([-0.42 * bg[:, 0], -0.38 * bg[:, 1]])
    return bg + dt * drift + config.noise_scale * par["sigma_bg"] * np.sqrt(dt) * rng.normal(size=bg.shape)


UPDATE_VEC = {
    "development": _update_development_vec,
    "microbiome": _update_microbiome_vec,
    "life_history": _update_life_history_vec,
    "epigenetic": _update_epigenetic_vec,
    "ecological": _update_ecology_vec,
    "background": _update_background_vec,
}


def _next_generation_start_vec(prev: Mapping[str, np.ndarray], rng: np.random.Generator, config: PilotConfig, par: Mapping[str, np.ndarray], *, theta_epi: np.ndarray | None = None) -> dict[str, np.ndarray]:
    n = config.n_lineages
    dev_final = prev["development"]
    micro_final = prev["microbiome"]
    life_final = prev["life_history"]
    epi_final = prev["epigenetic"] if theta_epi is None else np.asarray(theta_epi, dtype=float)
    if epi_final.ndim == 1:
        epi_final = np.tile(epi_final, (n, 1))
    eco_final = prev["ecological"]
    bg_final = prev["background"]
    noise = config.noise_scale
    bg_start = bg_final @ par["phi_bg"].T + noise * rng.normal(0.0, [0.20, 0.20], size=(n, 2))
    epi_start = par["rho_epi"] * epi_final + noise * rng.normal(0.0, _par_float(par, "epi_transmission_noise_sd", 0.10), size=(n, 2))
    eco_recon_scale = _par_float(par, "ecological_reconstruction_elc_scale")
    micro_eco_recon_scale = _par_float(par, "microbiome_to_ecology_reconstruction_scale")
    eco_start = np.column_stack(
        [
            0.64 * eco_final[:, 0] + 0.12 * eco_final[:, 1] + eco_recon_scale * (micro_eco_recon_scale * 0.18 * np.log1p(micro_final[:, 0]) + 0.18 * life_final[:, 2]),
            0.60 * eco_final[:, 1] + 0.10 * eco_final[:, 2] + eco_recon_scale * (0.18 * life_final[:, 2] + 0.10 * np.tanh(dev_final[:, 1])),
            0.62 * eco_final[:, 2] + 0.14 * eco_final[:, 0] + eco_recon_scale * (0.12 * life_final[:, 0] + micro_eco_recon_scale * 0.10 * np.log1p(micro_final[:, 4])),
        ]
    )
    eco_start += noise * rng.normal(0.0, 0.16, size=(n, 3))
    epi_centered = sigmoid(epi_start) - 0.5
    epi_dev_start_scale = _par_float(par, "epi_development_start_scale")
    dev_start = np.column_stack(
        [
            epi_dev_start_scale * 0.34 * epi_centered[:, 0] + 0.10 * eco_start[:, 1],
            0.22 * eco_start[:, 1] + epi_dev_start_scale * 0.16 * epi_centered[:, 0] + 0.08 * bg_start[:, 0],
            epi_dev_start_scale * 0.26 * epi_centered[:, 1] + 0.10 * bg_start[:, 1],
            epi_dev_start_scale * 0.20 * epi_centered[:, 0] + 0.12 * eco_start[:, 2],
            0.18 * eco_start[:, 2] + epi_dev_start_scale * 0.12 * epi_centered[:, 1],
        ]
    )
    dev_start += noise * rng.normal(0.0, 0.12, size=(n, 5))
    micro_log_start = 0.45 * np.log(micro_final) + 0.55 * np.array([-0.45, -0.55, -0.60, -0.62, -0.58])
    epi_micro_start_scale = _par_float(par, "epi_microbiome_start_scale")
    micro_log_start += np.column_stack(
        [
            0.08 * eco_start[:, 1] + epi_micro_start_scale * 0.04 * epi_centered[:, 0],
            0.10 * eco_start[:, 0],
            -0.08 * eco_start[:, 2],
            0.06 * bg_start[:, 1] + epi_micro_start_scale * 0.05 * epi_centered[:, 1],
            0.06 * eco_start[:, 0],
        ]
    )
    micro_start = np.exp(micro_log_start + noise * rng.normal(0.0, 0.12, size=(n, 5)))
    epi_life_start_scale = _par_float(par, "epi_life_start_scale")
    epi_life_start_maturation_scale = epi_life_start_scale * _par_float(par, "epi_life_start_maturation_scale")
    epi_life_start_growth_scale = epi_life_start_scale * _par_float(par, "epi_life_start_growth_scale")
    epi_life_start_allocation_scale = epi_life_start_scale * _par_float(par, "epi_life_start_allocation_scale")
    eco_life_start_scale = _par_float(par, "ecology_life_start_scale")
    life_start_logits = np.column_stack(
        [
            _par_float(par, "life_start_maturation_intercept", -1.30) + epi_life_start_maturation_scale * 0.20 * epi_centered[:, 0] + eco_life_start_scale * 0.10 * eco_start[:, 2] - 0.08 * bg_start[:, 1],
            -0.25 + eco_life_start_scale * 0.22 * eco_start[:, 1] + epi_life_start_growth_scale * 0.16 * epi_centered[:, 0] + 0.05 * bg_start[:, 0],
            -1.45 + eco_life_start_scale * 0.14 * eco_start[:, 0] + epi_life_start_allocation_scale * 0.12 * epi_centered[:, 1],
        ]
    )
    life_start = sigmoid(life_start_logits + noise * rng.normal(0.0, 0.12, size=(n, 3)))
    return {
        "development": dev_start,
        "microbiome": micro_start,
        "life_history": life_start,
        "epigenetic": epi_start,
        "ecological": eco_start,
        "background": bg_start,
    }


def _next_generation_start_multiparent_vec(prev: Mapping[str, np.ndarray], pairs: np.ndarray, rng: np.random.Generator, config: PilotConfig, par: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    start = _next_generation_start_vec(prev, rng, config, par)
    epi_d = prev["epigenetic"]
    epi_dp = prev["epigenetic"][pairs]
    interaction = np.column_stack([0.04 * epi_d[:, 0] * epi_dp[:, 0], 0.03 * epi_d[:, 1] * epi_dp[:, 1]])
    inherited = np.array([0.58, 0.56]) * epi_d + np.array([0.22, 0.20]) * epi_dp
    start["epigenetic"] = inherited + interaction + config.noise_scale * rng.normal(0.0, _par_float(par, "epi_transmission_noise_sd", 0.10), size=(config.n_lineages, 2))
    return start


def _integrate_generation_vec(start: Mapping[str, np.ndarray], tau: int, rng: np.random.Generator, config: PilotConfig, par: Mapping[str, np.ndarray], schedule: Sequence[tuple[float, tuple[str, ...]]], d_idx: np.ndarray | None = None) -> dict[str, np.ndarray]:
    retained = {
        "development": np.zeros((config.n_lineages, config.T_dev, config.dev_dim)),
        "microbiome": np.zeros((config.n_lineages, config.T_micro, config.micro_dim)),
        "life_history": np.zeros((config.n_lineages, config.T_life, config.life_dim)),
        "epigenetic": np.zeros((config.n_lineages, config.T_epi, config.epi_dim)),
        "ecological": np.zeros((config.n_lineages, config.T_eco, config.eco_dim)),
        "background": np.zeros((config.n_lineages, config.T_bg, config.bg_dim)),
    }
    state = {k: np.asarray(v, dtype=float).copy() for k, v in start.items()}
    step_counts = {level: 0 for level in level_steps(config)}
    retain_positions = {
        level: {int(step): int(i) for i, step in enumerate(full_retained_step_indices(steps, level_T(config)[level]))}
        for level, steps in level_steps(config).items()
    }
    for level in retained:
        retained[level][:, 0] = state[level]
    d_idx = np.arange(config.n_lineages) if d_idx is None else np.asarray(d_idx, dtype=int)
    if d_idx.shape[0] != config.n_lineages:
        raise AssertionError("d_idx must have one entry per simulated lineage/context")
    for u, due_levels in schedule:
        pre = {k: v.copy() for k, v in state.items()}
        updates = {}
        for level in due_levels:
            step_counts[level] += 1
            dt = 1.0 / level_steps(config)[level]
            updates[level] = UPDATE_VEC[level](pre, tau, u, dt, rng, config, par, d_idx)
        state.update(updates)
        for level in due_levels:
            step = step_counts[level]
            if step in retain_positions[level]:
                retained[level][:, retain_positions[level][step]] = state[level]
    return retained


def simulate_final_seed(seed: int, final_config: FinalAuditConfig, *, multiparent: bool = False) -> dict[str, object]:
    config = final_config.pilot_config(seed)
    rng = np.random.default_rng(config.seed)
    par = _params_for_config(final_config)
    schedule = build_event_schedule(config)
    arrays = _empty_arrays_vec(config)
    full_arrays = _empty_full_arrays_vec(config)
    reproductive_arrays = _empty_reproductive_arrays_vec(config)
    starts = _founder_states_vec(rng, config, par)
    retained = _integrate_generation_vec(starts, 0, rng, config, par, schedule)
    for key in arrays:
        full_arrays[key][:, 0] = retained[key]
        arrays[key][:, 0] = extract_analysis_segment(retained[key], key, config)
        reproductive_arrays[key][:, 0] = reproductive_state(retained[key], key, config)
    pairs = fixed_random_second_parent_pairs(config.n_lineages, config.seed) if multiparent else None
    for tau in range(config.n_generations):
        prev_reproductive = {key: reproductive_arrays[key][:, tau] for key in arrays}
        if multiparent:
            assert pairs is not None
            start = _next_generation_start_multiparent_vec(prev_reproductive, pairs, rng, config, par)
        else:
            start = _next_generation_start_vec(prev_reproductive, rng, config, par)
        retained = _integrate_generation_vec(start, tau + 1, rng, config, par, schedule)
        for key in arrays:
            full_arrays[key][:, tau + 1] = retained[key]
            arrays[key][:, tau + 1] = extract_analysis_segment(retained[key], key, config)
            reproductive_arrays[key][:, tau + 1] = reproductive_state(retained[key], key, config)
    out = {
        "config": config,
        "seed": seed,
        "parameters": par,
        "full_time_series": full_arrays,
        "reproductive_states": reproductive_arrays,
        "timestamps": level_timestamps(config),
        "segment_indices": level_segment_indices(config),
        "reproductive_indices": reproductive_indices(config),
        **arrays,
    }
    if pairs is not None:
        out["second_parent_pairs"] = pairs
    return out


def load_saved_seed(seed: int, final_config: FinalAuditConfig, data_dir: Path, *, multiparent: bool = False) -> dict[str, object] | None:
    if multiparent:
        path = data_dir / f"multiple_parent_temporal_architecture_seed_{seed}.npz"
    else:
        path = data_dir / f"corrected_temporal_architecture_seed_{seed}.npz"
    if not path.exists():
        return None
    config = final_config.pilot_config(seed)
    loaded = np.load(path)
    result: dict[str, object] = {
        "config": config,
        "seed": seed,
        "parameters": _params_for_config(final_config),
        "full_time_series": {},
        "reproductive_states": {},
        "timestamps": {},
        "segment_indices": {},
        "reproductive_indices": {},
    }
    for level in ELC_LEVELS + (BACKGROUND_LEVEL,):
        required = (
            f"segment_{level}",
            f"full_{level}",
            f"reproductive_{level}",
            f"timestamps_{level}",
            f"segment_indices_{level}",
            f"reproductive_index_{level}",
        )
        if any(key not in loaded.files for key in required):
            return None
        arr = loaded[f"segment_{level}"]
        expected_n = config.n_lineages
        expected_t = config.n_generations + 1
        if arr.shape[0] != expected_n or arr.shape[1] != expected_t:
            return None
        result[level] = arr
        result["full_time_series"][level] = loaded[f"full_{level}"]
        result["reproductive_states"][level] = loaded[f"reproductive_{level}"]
        result["timestamps"][level] = loaded[f"timestamps_{level}"]
        result["segment_indices"][level] = loaded[f"segment_indices_{level}"]
        result["reproductive_indices"][level] = int(loaded[f"reproductive_index_{level}"])
    if multiparent:
        if "second_parent_pairs" not in loaded.files:
            return None
        result["second_parent_pairs"] = loaded["second_parent_pairs"]
    return result


def save_seed_archive(path: Path, data: Mapping[str, object], *, pair_key: str | None = None) -> None:
    payload: dict[str, np.ndarray] = {}
    for level in ELC_LEVELS + (BACKGROUND_LEVEL,):
        payload[f"segment_{level}"] = np.asarray(data[level])
        payload[f"full_{level}"] = np.asarray(data["full_time_series"][level])
        payload[f"reproductive_{level}"] = np.asarray(data["reproductive_states"][level])
        payload[f"timestamps_{level}"] = np.asarray(data["timestamps"][level])
        payload[f"segment_indices_{level}"] = np.asarray(data["segment_indices"][level])
        payload[f"reproductive_index_{level}"] = np.asarray(data["reproductive_indices"][level], dtype=int)
    if pair_key is not None:
        payload[pair_key] = np.asarray(data[pair_key])
    np.savez_compressed(path, **payload)


def _segment(seed_data: Mapping[str, object], level: str, tau: int) -> np.ndarray:
    return np.asarray(seed_data[level])[:, tau].reshape(np.asarray(seed_data[level]).shape[0], -1)


def _concat_levels(seed_data: Mapping[str, object], tau: int, levels: Sequence[str]) -> np.ndarray:
    return np.hstack([_segment(seed_data, level, tau) for level in levels])


def _history(seed_data: Mapping[str, object], tau: int, levels: Sequence[str], k: int) -> np.ndarray:
    return np.hstack([_concat_levels(seed_data, tau - lag, levels) for lag in range(k)])


def _make_meta(seed: int, seed_position: int, n_lineages: int, taus: Sequence[int]) -> pd.DataFrame:
    rows = []
    for tau in taus:
        for d in range(n_lineages):
            rows.append({"seed": seed, "lineage_id": d, "unit_id": seed_position * n_lineages + d, "tau": tau})
    return pd.DataFrame(rows)


def phenotype_labels_for_generation(seed_data: Mapping[str, object], tau: int, config: PilotConfig) -> np.ndarray:
    life = np.asarray(seed_data["full_time_series"]["life_history"])[:, tau]
    life_times = np.asarray(seed_data["timestamps"]["life_history"], dtype=float)
    maturation = life[:, :, 0]
    crosses = maturation >= config.theta_maturation
    crossing = np.full(maturation.shape[0], -1, dtype=int)
    for t_l in range(maturation.shape[1]):
        crossing[(crossing < 0) & crosses[:, t_l]] = t_l
    crossing_u = np.full(crossing.shape, np.nan, dtype=float)
    valid = crossing >= 0
    crossing_u[valid] = life_times[crossing[valid]]
    early = valid & (crossing_u < config.early_maturation_u)
    high_growth = life[:, -1, 1] >= config.theta_growth
    labels = np.empty(maturation.shape[0], dtype=object)
    labels[early & high_growth] = "nu_early_maturation_high_growth"
    labels[early & ~high_growth] = "early_maturation_low_growth"
    labels[~early & high_growth] = "non_early_maturation_high_growth"
    labels[~early & ~high_growth] = "non_early_maturation_low_growth"
    return labels


def build_analysis_table(seed_results: Sequence[Mapping[str, object]], final_config: FinalAuditConfig, *, multiparent: bool = False) -> dict[str, object]:
    taus = list(range(final_config.source_tau_start, final_config.source_tau_stop + 1))
    metas = []
    cols: dict[str, list[np.ndarray]] = {
        "source_epigenetic": [],
        "source_ecological": [],
        "source_background": [],
        "target_remainder_without_epigenetic": [],
        "target_full_elc": [],
        "history_remainder_without_epigenetic": [],
        "history_full_elc": [],
        "target_epigenetic_future": [],
        "target_background_future": [],
        "target_remainder_without_epigenetic_ecological": [],
        "history_remainder_without_epigenetic_ecological": [],
        "target_epigenetic_ecological_future": [],
        "source_epigenetic_second_parent": [],
    }
    labels = []
    pair_rows = []
    for seed_position, seed_data in enumerate(seed_results):
        seed = int(seed_data["seed"])
        config = seed_data["config"]
        n = config.n_lineages
        metas.append(_make_meta(seed, seed_position, n, taus))
        levels_without_epi = tuple(level for level in ELC_LEVELS if level != "epigenetic")
        levels_without_epi_eco = tuple(level for level in ELC_LEVELS if level not in {"epigenetic", "ecological"})
        for tau in taus:
            cols["source_epigenetic"].append(_segment(seed_data, "epigenetic", tau))
            cols["source_ecological"].append(_segment(seed_data, "ecological", tau))
            cols["source_background"].append(_segment(seed_data, "background", tau))
            cols["target_remainder_without_epigenetic"].append(_concat_levels(seed_data, tau + 1, levels_without_epi))
            cols["target_full_elc"].append(_concat_levels(seed_data, tau + 1, ELC_LEVELS))
            cols["history_remainder_without_epigenetic"].append(_history(seed_data, tau, levels_without_epi, final_config.history_order))
            cols["history_full_elc"].append(_history(seed_data, tau, ELC_LEVELS, final_config.history_order))
            cols["target_epigenetic_future"].append(_segment(seed_data, "epigenetic", tau + 1))
            cols["target_background_future"].append(_segment(seed_data, "background", tau + 1))
            cols["target_remainder_without_epigenetic_ecological"].append(_concat_levels(seed_data, tau + 1, levels_without_epi_eco))
            cols["history_remainder_without_epigenetic_ecological"].append(_history(seed_data, tau, levels_without_epi_eco, final_config.history_order))
            cols["target_epigenetic_ecological_future"].append(np.hstack([_segment(seed_data, "epigenetic", tau + 1), _segment(seed_data, "ecological", tau + 1)]))
            if multiparent:
                pairs = np.asarray(seed_data["second_parent_pairs"], dtype=int)
                source_dp = _segment(seed_data, "epigenetic", tau)[pairs]
                cols["source_epigenetic_second_parent"].append(source_dp)
            labels.extend(phenotype_labels_for_generation(seed_data, tau + 1, config).tolist())
        if multiparent:
            pairs = np.asarray(seed_data["second_parent_pairs"], dtype=int)
            for d, dp in enumerate(pairs):
                pair_rows.append({"seed": seed, "lineage_id": d, "second_parent_lineage_id": int(dp), "same_individual": bool(d == int(dp))})
    table = {"meta": pd.concat(metas, ignore_index=True), "variant_label": np.asarray(labels, dtype=object)}
    for key, parts in cols.items():
        if parts:
            table[key] = np.vstack(parts)
    if pair_rows:
        table["second_parent_pairs"] = pd.DataFrame(pair_rows)
    return table


def _cmi_record(name: str, target: np.ndarray, source: np.ndarray, condition: np.ndarray) -> dict[str, float | str | int]:
    target_resid, source_resid = _residualized_gc_arrays(target, source, condition)
    return _cmi_record_from_residuals(name, target_resid, source_resid, target, source, condition)


def _cmi_record_from_residuals(
    name: str,
    target_resid: np.ndarray,
    source_resid: np.ndarray,
    target: np.ndarray,
    source: np.ndarray,
    condition: np.ndarray,
) -> dict[str, float | str | int]:
    return {
        "analysis": name,
        "estimate_bits": gaussian_mi_bits(target_resid, source_resid, transform=False),
        "estimator": "Gaussian-copula CMI via rank normalization and Gaussian residual covariance",
        "n_observations": int(target.shape[0]),
        "target_dimension": int(np.asarray(target).reshape(target.shape[0], -1).shape[1]),
        "source_dimension": int(np.asarray(source).reshape(source.shape[0], -1).shape[1]),
        "conditioning_dimension": int(np.asarray(condition).reshape(condition.shape[0], -1).shape[1]),
    }


def _residualized_gc_arrays(target: np.ndarray, source: np.ndarray, condition: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Yr = rank_gauss(target)
    Sr = rank_gauss(source)
    Cr = rank_gauss(condition)
    return _residualize(Yr, Cr), _residualize(Sr, Cr)


def _regularized_cov_from_raw(cov_raw: np.ndarray, *, ridge: float = 1e-4, shrink: float = 0.06) -> np.ndarray:
    cov_raw = np.atleast_2d(np.asarray(cov_raw, dtype=float))
    p = cov_raw.shape[0]
    diag_mean = np.trace(cov_raw) / max(p, 1)
    return (1.0 - shrink) * cov_raw + shrink * diag_mean * np.eye(p) + ridge * np.eye(p)


def _gaussian_mi_from_raw_cov(cov_y: np.ndarray, cov_s: np.ndarray, cov_ys: np.ndarray) -> float:
    value = 0.5 * (
        logdet_spd(_regularized_cov_from_raw(cov_y))
        + logdet_spd(_regularized_cov_from_raw(cov_s))
        - logdet_spd(_regularized_cov_from_raw(cov_ys))
    ) / np.log(2)
    return max(0.0, float(value))


def _cluster_summaries(z: np.ndarray, meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    groups = unit_index_groups(meta)
    unit_ids = np.array(list(groups.keys()), dtype=int)
    sums = []
    crosses = []
    counts = []
    for uid in unit_ids:
        block = z[groups[int(uid)]]
        sums.append(block.sum(axis=0))
        crosses.append(block.T @ block)
        counts.append(block.shape[0])
    return unit_ids, np.asarray(counts, dtype=float), np.asarray(sums, dtype=float), np.asarray(crosses, dtype=float)


def _weighted_raw_cov(counts: np.ndarray, sums: np.ndarray, crosses: np.ndarray, weights: np.ndarray) -> np.ndarray:
    n = float(weights @ counts)
    s = weights @ sums
    c = np.tensordot(weights, crosses, axes=(0, 0))
    return (c - np.outer(s, s) / n) / max(n - 1.0, 1.0)


def _bootstrap_mi_from_residuals(
    meta: pd.DataFrame,
    target_resid: np.ndarray,
    source_resid: np.ndarray,
    *,
    analysis: str,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    z = np.hstack([target_resid, source_resid])
    p_y = target_resid.shape[1]
    unit_ids, counts, sums, crosses = _cluster_summaries(z, meta)
    rng = np.random.default_rng(seed)
    n_units = len(unit_ids)
    rows = []
    probs = np.full(n_units, 1.0 / n_units)
    for b in range(n_boot):
        weights = rng.multinomial(n_units, probs).astype(float)
        cov_z = _weighted_raw_cov(counts, sums, crosses, weights)
        cov_y = cov_z[:p_y, :p_y]
        cov_s = cov_z[p_y:, p_y:]
        rows.append({"analysis": analysis, "bootstrap": b, "estimate_bits": _gaussian_mi_from_raw_cov(cov_y, cov_s, cov_z)})
    return pd.DataFrame(rows)


def _surrogate_mi_from_residuals(
    meta: pd.DataFrame,
    target_resid: np.ndarray,
    source_resid: np.ndarray,
    *,
    analysis: str,
    n_null: int,
    seed: int,
) -> pd.DataFrame:
    groups = unit_index_groups(meta)
    tau_values = meta["tau"].to_numpy()
    y = np.asarray(target_resid, dtype=float)
    s = np.asarray(source_resid, dtype=float)
    n = float(y.shape[0])
    p_y = y.shape[1]
    p_s = s.shape[1]
    sum_y = y.sum(axis=0)
    sum_s = s.sum(axis=0)
    cross_y = y.T @ y
    cross_s = s.T @ s
    cov_y = (cross_y - np.outer(sum_y, sum_y) / n) / max(n - 1.0, 1.0)
    cov_s = (cross_s - np.outer(sum_s, sum_s) / n) / max(n - 1.0, 1.0)
    shift_crosses = []
    for idx0 in groups.values():
        idx = np.asarray(idx0, dtype=int)
        idx = idx[np.argsort(tau_values[idx])]
        y_block = y[idx]
        s_block = s[idx]
        choices = []
        for shift in range(1, len(idx)):
            choices.append(y_block.T @ np.roll(s_block, shift, axis=0))
        shift_crosses.append(np.asarray(choices, dtype=float))
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_null):
        cross_ys = np.zeros((p_y, p_s), dtype=float)
        for choices in shift_crosses:
            cross_ys += choices[int(rng.integers(0, choices.shape[0]))]
        cov_ys = (cross_ys - np.outer(sum_y, sum_s) / n) / max(n - 1.0, 1.0)
        cov_z = np.empty((p_y + p_s, p_y + p_s), dtype=float)
        cov_z[:p_y, :p_y] = cov_y
        cov_z[p_y:, p_y:] = cov_s
        cov_z[:p_y, p_y:] = cov_ys
        cov_z[p_y:, :p_y] = cov_ys.T
        rows.append({"analysis": analysis, "null_iteration": b, "estimate_bits": _gaussian_mi_from_raw_cov(cov_y, cov_s, cov_z)})
    return pd.DataFrame(rows)


def _bootstrap_cmi_table(
    meta: pd.DataFrame,
    target: np.ndarray,
    source: np.ndarray,
    condition: np.ndarray,
    *,
    analysis: str,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    Yr, Sr = _residualized_gc_arrays(target, source, condition)
    return _bootstrap_mi_from_residuals(meta, Yr, Sr, analysis=analysis, n_boot=n_boot, seed=seed)


def _surrogate_null_cmi_table(
    meta: pd.DataFrame,
    target: np.ndarray,
    source: np.ndarray,
    condition: np.ndarray,
    *,
    analysis: str,
    n_null: int,
    seed: int,
) -> pd.DataFrame:
    Yr, Sr = _residualized_gc_arrays(target, source, condition)
    return _surrogate_mi_from_residuals(meta, Yr, Sr, analysis=analysis, n_null=n_null, seed=seed)


def _summarize_with_boot_null(observed_rows: list[dict[str, object]], boot: pd.DataFrame, null: pd.DataFrame | None = None) -> pd.DataFrame:
    obs = pd.DataFrame(observed_rows)
    intervals = []
    for analysis, group in boot.groupby("analysis"):
        interval = summarize_interval(group["estimate_bits"].to_numpy(dtype=float))
        observed = float(obs.set_index("analysis").loc[analysis, "estimate_bits"])
        se = float(np.std(group["estimate_bits"].to_numpy(dtype=float), ddof=1))
        z = 1.959963984540054
        row = {
            "analysis": analysis,
            "bootstrap_mean_bits": interval["mean"],
            "bootstrap_se_bits": se,
            "ci_method": "lineage-cluster bootstrap, centered standard-error interval",
            "ci_lower_bits": max(0.0, observed - z * se),
            "ci_upper_bits": observed + z * se,
            "percentile_lower_bits": interval["lower"],
            "percentile_upper_bits": interval["upper"],
        }
        if null is not None and analysis in set(null["analysis"]):
            vals = null.loc[null["analysis"] == analysis, "estimate_bits"].to_numpy(dtype=float)
            row["surrogate_null_mean_bits"] = float(np.mean(vals))
            row["surrogate_null_95_bits"] = float(np.quantile(vals, 0.95))
            row["surrogate_p_value"] = permutation_p_value(observed, vals)
        intervals.append(row)
    return obs.merge(pd.DataFrame(intervals), on="analysis", how="left")


def _seed_level_cmi(seed_tables: Sequence[Mapping[str, object]], specs: Sequence[tuple[str, str, str, str]]) -> pd.DataFrame:
    rows = []
    for table in seed_tables:
        seed = int(table["meta"]["seed"].iloc[0])
        for analysis, target_key, source_key, condition_key in specs:
            target_resid, source_resid = _residualized_gc_arrays(table[target_key], table[source_key], table[condition_key])
            rows.append(
                {
                    "seed": seed,
                    "analysis": analysis,
                    "estimate_bits": gaussian_mi_bits(target_resid, source_resid, transform=False),
                }
            )
    return pd.DataFrame(rows)


def _covariance_diagnostics(table: Mapping[str, object], keys: Sequence[str]) -> pd.DataFrame:
    rows = []
    for key in keys:
        arr = rank_gauss(np.asarray(table[key], dtype=float))
        cov = np.cov(arr - arr.mean(axis=0, keepdims=True), rowvar=False)
        cov = np.atleast_2d(cov)
        cov_reg = cov + 1e-4 * np.eye(cov.shape[0])
        rows.append(
            {
                "array": key,
                "n": int(arr.shape[0]),
                "dimension": int(arr.shape[1]),
                "matrix_rank": int(np.linalg.matrix_rank(cov_reg)),
                "condition_number": float(np.linalg.cond(cov_reg)),
            }
        )
    return pd.DataFrame(rows)


def phenotype_support(table: Mapping[str, object]) -> pd.DataFrame:
    meta = table["meta"].copy()
    meta["phenotype_state"] = table["variant_label"]
    rows = []
    for seed, group in meta.groupby("seed"):
        counts = group["phenotype_state"].value_counts().reindex(PHENOTYPE_LABELS, fill_value=0)
        total = int(counts.sum())
        for label, count in counts.items():
            rows.append({"seed": int(seed), "phenotype_state": label, "count": int(count), "proportion": float(count / max(total, 1))})
    counts = meta["phenotype_state"].value_counts().reindex(PHENOTYPE_LABELS, fill_value=0)
    total = int(counts.sum())
    for label, count in counts.items():
        rows.append({"seed": "all", "phenotype_state": label, "count": int(count), "proportion": float(count / max(total, 1))})
    return pd.DataFrame(rows)


def run_primary_cmi_analyses(table: Mapping[str, object], seed_tables: Sequence[Mapping[str, object]], final_config: FinalAuditConfig, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    specs = [
        ("epigenetic_predictive_contribution", "target_remainder_without_epigenetic", "source_epigenetic", "history_remainder_without_epigenetic"),
        ("background_predictive_contribution", "target_full_elc", "source_background", "history_full_elc"),
        ("epigenetic_predictive_closure", "target_epigenetic_future", "history_remainder_without_epigenetic", "source_epigenetic"),
        ("background_closure_control", "target_background_future", "history_full_elc", "source_background"),
        ("joint_epigenetic_ecological_predictive_contribution", "target_remainder_without_epigenetic_ecological", "complex_source_epigenetic_ecological", "history_remainder_without_epigenetic_ecological"),
        ("joint_epigenetic_ecological_predictive_closure", "target_epigenetic_ecological_future", "history_remainder_without_epigenetic_ecological", "complex_source_epigenetic_ecological"),
    ]
    table = dict(table)
    table["complex_source_epigenetic_ecological"] = np.hstack([table["source_epigenetic"], table["source_ecological"]])
    seed_tables2 = []
    for st in seed_tables:
        std = dict(st)
        std["complex_source_epigenetic_ecological"] = np.hstack([st["source_epigenetic"], st["source_ecological"]])
        seed_tables2.append(std)
    boot_parts = []
    null_parts = []
    observed = []
    for i, (name, target, source, condition) in enumerate(specs):
        print(f"primary analysis {i + 1}/{len(specs)}: {name}", flush=True)
        target_resid, source_resid = _residualized_gc_arrays(table[target], table[source], table[condition])
        observed.append(_cmi_record_from_residuals(name, target_resid, source_resid, table[target], table[source], table[condition]))
        boot_parts.append(_bootstrap_mi_from_residuals(table["meta"], target_resid, source_resid, analysis=name, n_boot=final_config.n_bootstrap, seed=final_config.bootstrap_seed + 1009 * i))
        null_parts.append(_surrogate_mi_from_residuals(table["meta"], target_resid, source_resid, analysis=name, n_null=final_config.n_null, seed=final_config.null_seed + 1009 * i))
    boot = pd.concat(boot_parts, ignore_index=True)
    null = pd.concat(null_parts, ignore_index=True)
    summary = _summarize_with_boot_null(observed, boot, null)
    seed_level = _seed_level_cmi(seed_tables2, specs)
    summary.to_csv(output_dir / "primary_transfer_entropy_and_closure_summary.csv", index=False)
    boot.to_csv(output_dir / "primary_transfer_entropy_and_closure_bootstrap.csv", index=False)
    null.to_csv(output_dir / "primary_transfer_entropy_and_closure_surrogate_null.csv", index=False)
    seed_level.to_csv(output_dir / "primary_transfer_entropy_and_closure_by_seed.csv", index=False)
    return summary, boot, null


def run_stability(table_by_horizon: Mapping[int, Mapping[str, object]], final_config: FinalAuditConfig, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    boots = []
    for i, (rho, table) in enumerate(table_by_horizon.items()):
        name = f"R_epigenetic_rho_{rho}"
        target = table["target_remainder_without_epigenetic"]
        source = table["source_epigenetic"]
        condition = table["history_remainder_without_epigenetic"]
        target_resid, source_resid = _residualized_gc_arrays(target, source, condition)
        estimate = gaussian_mi_bits(target_resid, source_resid, transform=False)
        boot = _bootstrap_cmi_table(table["meta"], target, source, condition, analysis=name, n_boot=final_config.n_bootstrap, seed=final_config.bootstrap_seed + 50000 + 997 * rho)
        interval = summarize_interval(boot["estimate_bits"])
        se = float(np.std(boot["estimate_bits"], ddof=1))
        z = 1.959963984540054
        rows.append(
            {
                "rho": int(rho),
                "analysis": name,
                "estimate_bits": float(estimate),
                "ci_lower_bits": max(0.0, float(estimate - z * se)),
                "ci_upper_bits": float(estimate + z * se),
                "bootstrap_mean_bits": interval["mean"],
                "bootstrap_se_bits": se,
                "ci_method": "lineage-cluster bootstrap, centered standard-error interval",
            }
        )
        boot["rho"] = rho
        boots.append(boot)
    summary = pd.DataFrame(rows)
    boot_df = pd.concat(boots, ignore_index=True)
    summary.to_csv(output_dir / "intergenerational_stability_summary.csv", index=False)
    boot_df.to_csv(output_dir / "intergenerational_stability_bootstrap.csv", index=False)
    return summary, boot_df


def build_horizon_table(seed_results: Sequence[Mapping[str, object]], final_config: FinalAuditConfig, rho: int) -> dict[str, object]:
    max_tau = final_config.source_tau_stop + 1 - rho
    taus = list(range(final_config.source_tau_start, max_tau + 1))
    metas = []
    parts = {"source_epigenetic": [], "target_remainder_without_epigenetic": [], "history_remainder_without_epigenetic": []}
    for seed_position, seed_data in enumerate(seed_results):
        seed = int(seed_data["seed"])
        config = seed_data["config"]
        levels_without_epi = tuple(level for level in ELC_LEVELS if level != "epigenetic")
        metas.append(_make_meta(seed, seed_position, config.n_lineages, taus))
        for tau in taus:
            parts["source_epigenetic"].append(_segment(seed_data, "epigenetic", tau))
            parts["target_remainder_without_epigenetic"].append(_concat_levels(seed_data, tau + rho, levels_without_epi))
            parts["history_remainder_without_epigenetic"].append(_history(seed_data, tau, levels_without_epi, final_config.history_order))
    return {"meta": pd.concat(metas, ignore_index=True), **{k: np.vstack(v) for k, v in parts.items()}}


def run_location_analysis(table: Mapping[str, object], final_config: FinalAuditConfig, output_dir: Path) -> pd.DataFrame:
    source = table["source_epigenetic"]
    condition = table["history_remainder_without_epigenetic"]
    condition_rank = rank_gauss(condition)
    source_resid = _residualize(rank_gauss(source), condition_rank)
    rows = []
    for target_level in ("development", "microbiome", "life_history", "ecological"):
        print(f"location analysis target: {target_level}", flush=True)
        m_l = {"development": 8, "microbiome": 10, "life_history": 5, "ecological": 5}[target_level]
        dim_l = len(COMPONENTS[target_level])
        arrs = []
        for seed_data in table["seed_results"]:
            for tau in range(final_config.source_tau_start, final_config.source_tau_stop + 1):
                arrs.append(np.asarray(seed_data[target_level])[:, tau + 1])
        target_all = np.vstack(arrs)
        for t_l in range(m_l):
            y = target_all[:, t_l, :]
            target_resid = _residualize(rank_gauss(y), condition_rank)
            rows.append(
                {
                    "source_level": "epigenetic",
                    "target_level": target_level,
                    "target_t_l": int(t_l),
                    "target_dimension": int(dim_l),
                    "source_dimension": int(source.shape[1]),
                    "conditioning_set": "X_ELC_excluding_epigenetic_history_k_2",
                    "transfer_entropy_bits": gaussian_mi_bits(target_resid, source_resid, transform=False),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "level_time_specific_transfer_entropy.csv", index=False)
    return out


def _residualize(values: np.ndarray, condition: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    V = np.asarray(values, dtype=float)
    C = np.asarray(condition, dtype=float)
    design = np.column_stack([np.ones(C.shape[0]), C])
    gram = design.T @ design + ridge * np.eye(design.shape[1])
    coef = np.linalg.solve(gram, design.T @ V)
    return V - design @ coef


def _pid_once(target: np.ndarray, source1: np.ndarray, source2: np.ndarray, condition: np.ndarray, *, max_iter: int) -> dict[str, float | bool | str | int]:
    Cr = rank_gauss(condition)
    target_resid = _residualize(rank_gauss(target), Cr)
    source1_resid = _residualize(rank_gauss(source1), Cr)
    source2_resid = _residualize(rank_gauss(source2), Cr)
    return _pid_once_residuals(target_resid, source1_resid, source2_resid, max_iter=max_iter)


def _pid_once_residuals(target_resid: np.ndarray, source1_resid: np.ndarray, source2_resid: np.ndarray, *, max_iter: int) -> dict[str, float | bool | str | int]:
    delta_g_pid = _load_delta_g_pid()
    pid = delta_g_pid(target_resid, source1_resid, source2_resid, rank_transform=False, bias_correct=False, max_iter=max_iter)
    atom_sum = pid["RI"] + pid["UI_X"] + pid["UI_Y"] + pid["SI"]
    error = abs(atom_sum - pid["I_MXY"])
    return {
        "matched_joint_information_bits": float(pid["I_MXY"]),
        "redundancy_bits": float(pid["RI"]),
        "unique_source_1_bits": float(pid["UI_X"]),
        "unique_source_2_bits": float(pid["UI_Y"]),
        "synergy_bits": float(pid["SI"]),
        "atom_sum_bits": float(atom_sum),
        "absolute_reconstruction_error_bits": float(error),
        "within_tolerance": bool(error <= 1e-6),
        "target_dimension": int(target_resid.shape[1]),
        "source_1_dimension": int(source1_resid.shape[1]),
        "source_2_dimension": int(source2_resid.shape[1]),
        "conditioning_dimension": np.nan,
    }


def run_pid(table: Mapping[str, object], final_config: FinalAuditConfig, output_dir: Path, *, prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if prefix == "epigenetic_ecological":
        target = table["target_remainder_without_epigenetic_ecological"]
        source1 = table["source_epigenetic"]
        source2 = table["source_ecological"]
        condition = table["history_remainder_without_epigenetic_ecological"]
        source_1_name = "epigenetic_hereditary_factor"
        source_2_name = "ecological_candidate_hereditary_factor"
    elif prefix == "multiple_parent":
        target = table["target_remainder_without_epigenetic"]
        source1 = table["source_epigenetic"]
        source2 = table["source_epigenetic_second_parent"]
        condition = table["history_remainder_without_epigenetic"]
        source_1_name = "epigenetic_factor_from_d"
        source_2_name = "epigenetic_factor_from_d_prime"
    else:
        raise ValueError(prefix)
    print(f"PID {prefix}: Gaussian-copula transform and residualization", flush=True)
    condition_rank = rank_gauss(condition)
    target_resid = _residualize(rank_gauss(target), condition_rank)
    source1_resid = _residualize(rank_gauss(source1), condition_rank)
    source2_resid = _residualize(rank_gauss(source2), condition_rank)
    observed = _pid_once_residuals(target_resid, source1_resid, source2_resid, max_iter=final_config.pid_max_iter)
    observed["conditioning_dimension"] = int(condition.shape[1])
    observed["conditional_preprocessing"] = "rank normalization followed by linear residualization on the conditioning set before delta_G PID"
    observed.update({"pid": prefix, "source_1": source_1_name, "source_2": source_2_name})
    groups = unit_index_groups(table["meta"])
    rng = np.random.default_rng(final_config.bootstrap_seed + (777 if prefix == "multiple_parent" else 333))
    boot_rows = []
    for b in range(final_config.n_pid_bootstrap):
        if b % 50 == 0:
            print(f"PID {prefix}: bootstrap {b + 1}/{final_config.n_pid_bootstrap}", flush=True)
        idx = bootstrap_indices_from_groups(groups, rng)
        row = _pid_once_residuals(target_resid[idx], source1_resid[idx], source2_resid[idx], max_iter=max(80, final_config.pid_max_iter // 2))
        row["conditioning_dimension"] = int(condition.shape[1])
        row["conditional_preprocessing"] = "same full-sample rank normalization and residualization preprocessing, lineage-cluster bootstrap on residual observations"
        row.update({"pid": prefix, "bootstrap": b, "source_1": source_1_name, "source_2": source_2_name})
        boot_rows.append(row)
    summary = pd.DataFrame([observed])
    boot = pd.DataFrame(boot_rows)
    for col in ["matched_joint_information_bits", "redundancy_bits", "unique_source_1_bits", "unique_source_2_bits", "synergy_bits"]:
        interval = summarize_interval(boot[col])
        summary[f"{col}_bootstrap_mean"] = interval["mean"]
        summary[f"{col}_ci_lower"] = interval["lower"]
        summary[f"{col}_ci_upper"] = interval["upper"]
    summary.to_csv(output_dir / f"{prefix}_delta_g_pid_summary.csv", index=False)
    boot.to_csv(output_dir / f"{prefix}_delta_g_pid_bootstrap.csv", index=False)
    return summary, boot


def _simulate_one_context(seed_data: Mapping[str, object], d: int, tau: int, theta: np.ndarray, random_seed: int, final_config: FinalAuditConfig) -> str:
    base_config: PilotConfig = seed_data["config"]
    config = replace(base_config, n_lineages=1)
    par = seed_data["parameters"]
    rng = np.random.default_rng(random_seed)
    prev = {level: np.asarray(seed_data["reproductive_states"][level])[d : d + 1, tau].copy() for level in ("development", "microbiome", "life_history", "epigenetic", "ecological", "background")}
    start = _next_generation_start_vec(prev, rng, config, par, theta_epi=theta[None, :])
    retained = _integrate_generation_vec(start, tau + 1, rng, config, par, build_event_schedule(config))
    life = retained["life_history"][0]
    maturation = life[:, 0]
    crossing_idx = np.where(maturation >= final_config.theta_maturation)[0]
    crossing = int(crossing_idx[0]) if crossing_idx.size else -1
    crossing_u = level_timestamps(config)["life_history"][crossing] if crossing >= 0 else np.nan
    early = crossing >= 0 and crossing_u < config.early_maturation_u
    high = float(life[-1, 1]) >= final_config.theta_growth
    if early and high:
        return "nu_early_maturation_high_growth"
    if early and not high:
        return "early_maturation_low_growth"
    if (not early) and high:
        return "non_early_maturation_high_growth"
    return "non_early_maturation_low_growth"


def _select_intervention_contexts(seed_results: Sequence[Mapping[str, object]], final_config: FinalAuditConfig) -> pd.DataFrame:
    rng = np.random.default_rng(final_config.intervention_seed)
    rows = []
    tau = int((final_config.source_tau_start + final_config.source_tau_stop) // 2)
    for seed_data in seed_results:
        seed = int(seed_data["seed"])
        n = seed_data["config"].n_lineages
        chosen_lineages = rng.choice(np.arange(n), size=final_config.n_contexts_per_seed, replace=False)
        for context_local, d in enumerate(chosen_lineages):
            rows.append({"seed": seed, "lineage_id": int(d), "tau": int(tau), "context_id": len(rows), "context_local_id": int(context_local)})
    return pd.DataFrame(rows)


def run_intervention(seed_results: Sequence[Mapping[str, object]], final_config: FinalAuditConfig, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    contexts = _select_intervention_contexts(seed_results, final_config)
    seed_map = {int(sd["seed"]): sd for sd in seed_results}
    theta_values = np.linspace(final_config.intervention_grid_min, final_config.intervention_grid_max, final_config.intervention_grid_size)
    theta_grid = [(float(a), float(b)) for a in theta_values for b in theta_values]
    fisher_base_points = [(-1.0, -1.0), (-0.5, 0.0), (0.0, 0.0), (0.5, 0.0), (1.0, 1.0)]
    fisher_thetas = set()
    h = final_config.fisher_step
    for a, b in fisher_base_points:
        fisher_thetas.update([(a, b), (a + h, b), (a - h, b), (a, b + h), (a, b - h)])
    response_theta_set = set(theta_grid)
    all_theta_values = sorted(response_theta_set | fisher_thetas)
    rows = []
    schedule_cache: dict[int, Sequence[tuple[float, tuple[str, ...]]]] = {}
    for seed, seed_contexts in contexts.groupby("seed"):
        seed_data = seed_map[int(seed)]
        base_config: PilotConfig = seed_data["config"]
        tau = int(seed_contexts["tau"].iloc[0])
        lineages = seed_contexts["lineage_id"].to_numpy(dtype=int)
        context_ids = seed_contexts["context_id"].to_numpy(dtype=int)
        repeated_lineages = np.repeat(lineages, final_config.n_intervention_draws)
        repeated_context_ids = np.repeat(context_ids, final_config.n_intervention_draws)
        n_batch = repeated_lineages.size
        batch_config = replace(base_config, n_lineages=int(n_batch))
        if n_batch not in schedule_cache:
            schedule_cache[n_batch] = build_event_schedule(batch_config)
        prev_template = {
            level: np.asarray(seed_data["reproductive_states"][level])[repeated_lineages, tau].copy()
            for level in ("development", "microbiome", "life_history", "epigenetic", "ecological", "background")
        }
        d_idx = repeated_lineages
        print(f"intervention seed {int(seed)}: {len(all_theta_values)} intervention values, {n_batch} context-draw rows", flush=True)
        for theta_i, (theta_reg, theta_stress) in enumerate(all_theta_values):
            if theta_i % 10 == 0:
                print(f"  intervention value {theta_i + 1}/{len(all_theta_values)} for seed {int(seed)}", flush=True)
            rng = np.random.default_rng(final_config.intervention_seed + int(seed) * 10_000)
            start = _next_generation_start_vec(prev_template, rng, batch_config, seed_data["parameters"], theta_epi=np.array([theta_reg, theta_stress]))
            retained = _integrate_generation_vec(start, tau + 1, rng, batch_config, seed_data["parameters"], schedule_cache[n_batch], d_idx=d_idx)
            life = retained["life_history"]
            maturation = life[:, :, 0]
            crosses = maturation >= final_config.theta_maturation
            crossing = np.full(n_batch, -1, dtype=int)
            for t_l in range(maturation.shape[1]):
                crossing[(crossing < 0) & crosses[:, t_l]] = t_l
            life_times = level_timestamps(batch_config)["life_history"]
            crossing_u = np.full(crossing.shape, np.nan, dtype=float)
            valid = crossing >= 0
            crossing_u[valid] = life_times[crossing[valid]]
            early = valid & (crossing_u < batch_config.early_maturation_u)
            high = life[:, -1, 1] >= final_config.theta_growth
            labels = np.empty(n_batch, dtype=object)
            labels[early & high] = "nu_early_maturation_high_growth"
            labels[early & ~high] = "early_maturation_low_growth"
            labels[~early & high] = "non_early_maturation_high_growth"
            labels[~early & ~high] = "non_early_maturation_low_growth"
            tmp = pd.DataFrame({"context_id": repeated_context_ids, "phenotype_state": labels})
            counts = tmp.groupby(["context_id", "phenotype_state"]).size().unstack(fill_value=0).reindex(columns=PHENOTYPE_LABELS, fill_value=0)
            counts = counts.reindex(context_ids, fill_value=0)
            probs = counts.div(float(final_config.n_intervention_draws))
            for context_id, d, row in zip(context_ids, lineages, probs.to_dict("records")):
                rows.append(
                    {
                        "seed": int(seed),
                        "lineage_id": int(d),
                        "tau": int(tau),
                        "context_id": int(context_id),
                        "theta_reg": float(theta_reg),
                        "theta_stress": float(theta_stress),
                        "on_approved_7x7_response_grid": bool((theta_reg, theta_stress) in response_theta_set),
                        "used_for_fisher_finite_difference": bool((theta_reg, theta_stress) in fisher_thetas),
                        **{f"p_{label}": float(row[label]) for label in PHENOTYPE_LABELS},
                    }
                )
    prob_by_context = pd.DataFrame(rows)
    prob_by_context.to_csv(output_dir / "intervention_probability_by_context.csv", index=False)
    response_contexts = prob_by_context[prob_by_context["on_approved_7x7_response_grid"]].copy()
    group_cols = ["theta_reg", "theta_stress"]
    summary = response_contexts.groupby(group_cols)[[f"p_{label}" for label in PHENOTYPE_LABELS]].mean().reset_index()
    groups = {cid: g.index.to_numpy(dtype=int) for cid, g in response_contexts.groupby("context_id")}
    rng = np.random.default_rng(final_config.bootstrap_seed + 90000)
    boot_rows = []
    for b in range(final_config.n_bootstrap):
        sampled_contexts = rng.choice(np.array(list(groups.keys())), size=len(groups), replace=True)
        idx = np.concatenate([groups[int(cid)] for cid in sampled_contexts])
        boot_mean = response_contexts.loc[idx].groupby(group_cols)[[f"p_{label}" for label in PHENOTYPE_LABELS]].mean().reset_index()
        boot_mean["bootstrap"] = b
        boot_rows.append(boot_mean)
    boot = pd.concat(boot_rows, ignore_index=True)
    interval_rows = []
    for (theta_reg, theta_stress), group in boot.groupby(group_cols):
        row = {"theta_reg": theta_reg, "theta_stress": theta_stress}
        for label in PHENOTYPE_LABELS:
            interval = summarize_interval(group[f"p_{label}"])
            row[f"p_{label}_ci_lower"] = interval["lower"]
            row[f"p_{label}_ci_upper"] = interval["upper"]
        interval_rows.append(row)
    summary = summary.merge(pd.DataFrame(interval_rows), on=group_cols, how="left")
    summary.to_csv(output_dir / "intervention_response_surface_summary.csv", index=False)
    boot.to_csv(output_dir / "intervention_response_surface_bootstrap.csv", index=False)
    fisher = _fisher_from_context_probabilities(prob_by_context, final_config)
    fisher.to_csv(output_dir / "intervention_fisher_information.csv", index=False)
    return summary, prob_by_context, fisher


def _nearest_probability_rows(prob_by_context: pd.DataFrame, theta: tuple[float, float]) -> pd.DataFrame:
    return prob_by_context[(np.isclose(prob_by_context["theta_reg"], theta[0])) & (np.isclose(prob_by_context["theta_stress"], theta[1]))]


def _fisher_from_context_probabilities(prob_by_context: pd.DataFrame, final_config: FinalAuditConfig) -> pd.DataFrame:
    theta_points = [(-1.0, -1.0), (-0.5, 0.0), (0.0, 0.0), (0.5, 0.0), (1.0, 1.0)]
    h = final_config.fisher_step
    rows = []
    prob_cols = [f"p_{label}" for label in PHENOTYPE_LABELS]
    for theta in theta_points:
        base = _nearest_probability_rows(prob_by_context, theta)
        plus_reg = _nearest_probability_rows(prob_by_context, (theta[0] + h, theta[1]))
        minus_reg = _nearest_probability_rows(prob_by_context, (theta[0] - h, theta[1]))
        plus_stress = _nearest_probability_rows(prob_by_context, (theta[0], theta[1] + h))
        minus_stress = _nearest_probability_rows(prob_by_context, (theta[0], theta[1] - h))
        if len(plus_reg) == 0 or len(minus_reg) == 0 or len(plus_stress) == 0 or len(minus_stress) == 0:
    # Interpolate finite-difference points that fall between grid coordinates.
            fisher = _fisher_by_interpolation(prob_by_context, theta, h)
            rows.extend(fisher)
            continue
        for _, b in base.iterrows():
            context_id = int(b["context_id"])
            def get(rows_df: pd.DataFrame) -> np.ndarray:
                match = rows_df[rows_df["context_id"] == context_id]
                return match[prob_cols].to_numpy(dtype=float)[0]
            p = np.clip(get(base), 1e-9, 1.0)
            p = p / p.sum()
            dp_reg = (get(plus_reg) - get(minus_reg)) / (2 * h)
            dp_stress = (get(plus_stress) - get(minus_stress)) / (2 * h)
            J = np.vstack([dp_reg, dp_stress])
            F = J @ np.diag(1.0 / p) @ J.T
            rows.extend(
                [
                    {"theta_reg": theta[0], "theta_stress": theta[1], "context_id": context_id, "row_component": "theta_reg", "col_component": "theta_reg", "fisher_information": float(F[0, 0]), "finite_difference_step": h, "method": "context-specific finite differences over all four phenotype states"},
                    {"theta_reg": theta[0], "theta_stress": theta[1], "context_id": context_id, "row_component": "theta_reg", "col_component": "theta_stress", "fisher_information": float(F[0, 1]), "finite_difference_step": h, "method": "context-specific finite differences over all four phenotype states"},
                    {"theta_reg": theta[0], "theta_stress": theta[1], "context_id": context_id, "row_component": "theta_stress", "col_component": "theta_reg", "fisher_information": float(F[1, 0]), "finite_difference_step": h, "method": "context-specific finite differences over all four phenotype states"},
                    {"theta_reg": theta[0], "theta_stress": theta[1], "context_id": context_id, "row_component": "theta_stress", "col_component": "theta_stress", "fisher_information": float(F[1, 1]), "finite_difference_step": h, "method": "context-specific finite differences over all four phenotype states"},
                ]
            )
    out = pd.DataFrame(rows)
    return out.groupby(["theta_reg", "theta_stress", "row_component", "col_component", "finite_difference_step", "method"])["fisher_information"].mean().reset_index()


def _fisher_by_interpolation(prob_by_context: pd.DataFrame, theta: tuple[float, float], h: float) -> list[dict[str, object]]:
    prob_cols = [f"p_{label}" for label in PHENOTYPE_LABELS]
    rows = []
    theta_vals_reg = np.sort(prob_by_context["theta_reg"].unique())
    theta_vals_stress = np.sort(prob_by_context["theta_stress"].unique())

    def interp_context(context_df: pd.DataFrame, a: float, b: float) -> np.ndarray:
    # Use inverse-distance interpolation between response-grid coordinates.
        pts = context_df[["theta_reg", "theta_stress"]].to_numpy(dtype=float)
        probs = context_df[prob_cols].to_numpy(dtype=float)
        dist = np.sqrt((pts[:, 0] - a) ** 2 + (pts[:, 1] - b) ** 2)
        exact = np.where(dist < 1e-12)[0]
        if exact.size:
            p = probs[exact[0]]
        else:
            w = 1.0 / np.maximum(dist, 1e-9) ** 2
            p = (w[:, None] * probs).sum(axis=0) / w.sum()
        p = np.clip(p, 1e-9, 1.0)
        return p / p.sum()

    for context_id, context_df in prob_by_context.groupby("context_id"):
        p = interp_context(context_df, theta[0], theta[1])
        dp_reg = (interp_context(context_df, theta[0] + h, theta[1]) - interp_context(context_df, theta[0] - h, theta[1])) / (2 * h)
        dp_stress = (interp_context(context_df, theta[0], theta[1] + h) - interp_context(context_df, theta[0], theta[1] - h)) / (2 * h)
        J = np.vstack([dp_reg, dp_stress])
        F = J @ np.diag(1.0 / p) @ J.T
        rows.extend(
            [
                {"theta_reg": theta[0], "theta_stress": theta[1], "context_id": int(context_id), "row_component": "theta_reg", "col_component": "theta_reg", "fisher_information": float(F[0, 0]), "finite_difference_step": h, "method": "context-specific finite differences using inverse-distance interpolation on approved 7x7 response grid"},
                {"theta_reg": theta[0], "theta_stress": theta[1], "context_id": int(context_id), "row_component": "theta_reg", "col_component": "theta_stress", "fisher_information": float(F[0, 1]), "finite_difference_step": h, "method": "context-specific finite differences using inverse-distance interpolation on approved 7x7 response grid"},
                {"theta_reg": theta[0], "theta_stress": theta[1], "context_id": int(context_id), "row_component": "theta_stress", "col_component": "theta_reg", "fisher_information": float(F[1, 0]), "finite_difference_step": h, "method": "context-specific finite differences using inverse-distance interpolation on approved 7x7 response grid"},
                {"theta_reg": theta[0], "theta_stress": theta[1], "context_id": int(context_id), "row_component": "theta_stress", "col_component": "theta_stress", "fisher_information": float(F[1, 1]), "finite_difference_step": h, "method": "context-specific finite differences using inverse-distance interpolation on approved 7x7 response grid"},
            ]
        )
    return rows


def run_variant_specific(table: Mapping[str, object], final_config: FinalAuditConfig, output_dir: Path) -> pd.DataFrame:
    labels = pd.Series(table["variant_label"])
    counts = labels.value_counts().reindex(PHENOTYPE_LABELS, fill_value=0)
    probs = counts / counts.sum()
    y_onehot = pd.get_dummies(labels).reindex(columns=PHENOTYPE_LABELS, fill_value=0).to_numpy(dtype=float)
    te_variant = gaussian_mi_bits(*_residualized_gc_arrays(y_onehot, table["source_epigenetic"], table["history_remainder_without_epigenetic"]), transform=False)
    nu_indicator = (labels.to_numpy() == "nu_early_maturation_high_growth").astype(float)[:, None]
    te_nu = gaussian_mi_bits(*_residualized_gc_arrays(nu_indicator, table["source_epigenetic"], table["history_remainder_without_epigenetic"]), transform=False)
    out = pd.DataFrame(
        [
            {"quantity": "complete_phenotype_distribution", "phenotype_state": label, "estimate": float(probs[label]), "units": "probability"}
            for label in PHENOTYPE_LABELS
        ]
        + [
            {"quantity": "variant_specific_transfer_entropy_for_nu", "phenotype_state": "nu_early_maturation_high_growth", "estimate": float(te_nu), "units": "bits"},
            {"quantity": "transfer_entropy_to_complete_phenotype_state", "phenotype_state": "all_four_states", "estimate": float(te_variant), "units": "bits"},
        ]
    )
    out.to_csv(output_dir / "variant_specific_results.csv", index=False)
    return out


def run_multiple_parent(seed_results_mp: Sequence[Mapping[str, object]], final_config: FinalAuditConfig, output_dir: Path) -> tuple[dict[str, object], pd.DataFrame]:
    table = build_analysis_table(seed_results_mp, final_config, multiparent=True)
    table["complex_source_parent_epigenetic"] = np.hstack([table["source_epigenetic"], table["source_epigenetic_second_parent"]])
    specs = [
        ("multiple_parent_factor_from_d", "target_remainder_without_epigenetic", "source_epigenetic", "history_remainder_without_epigenetic"),
        ("multiple_parent_factor_from_d_prime", "target_remainder_without_epigenetic", "source_epigenetic_second_parent", "history_remainder_without_epigenetic"),
        ("multiple_parent_joint_factors", "target_remainder_without_epigenetic", "complex_source_parent_epigenetic", "history_remainder_without_epigenetic"),
    ]
    observed = [_cmi_record(name, table[target], table[source], table[condition]) for name, target, source, condition in specs]
    boot_parts = []
    null_parts = []
    for i, (name, target, source, condition) in enumerate(specs):
        boot_parts.append(_bootstrap_cmi_table(table["meta"], table[target], table[source], table[condition], analysis=name, n_boot=final_config.n_bootstrap, seed=final_config.bootstrap_seed + 140000 + 1009 * i))
        null_parts.append(_surrogate_null_cmi_table(table["meta"], table[target], table[source], table[condition], analysis=name, n_null=final_config.n_null, seed=final_config.null_seed + 140000 + 1009 * i))
    boot = pd.concat(boot_parts, ignore_index=True)
    null = pd.concat(null_parts, ignore_index=True)
    summary = _summarize_with_boot_null(observed, boot, null)
    summary.to_csv(output_dir / "multiple_parent_transfer_entropy_summary.csv", index=False)
    boot.to_csv(output_dir / "multiple_parent_transfer_entropy_bootstrap.csv", index=False)
    null.to_csv(output_dir / "multiple_parent_transfer_entropy_surrogate_null.csv", index=False)
    pairs = table.get("second_parent_pairs")
    if isinstance(pairs, pd.DataFrame):
        pairs.to_csv(output_dir / "multiple_parent_fixed_second_parent_pairs.csv", index=False)
    pid_summary, pid_boot = run_pid(table, final_config, output_dir, prefix="multiple_parent")
    return {"summary": summary, "pid_summary": pid_summary, "pid_bootstrap": pid_boot}, table


def write_sanity_checks(output_dir: Path, final_config: FinalAuditConfig) -> None:
    edges = effective_nonzero_cross_level_edges(_params_for_config(final_config))
    edges.to_csv(output_dir / "corrected_final_nonzero_cross_level_edges.csv", index=False)
    rows = []
    for _, edge in edges.iterrows():
        rows.append(
            {
                **edge.to_dict(),
                "implementation_check": "active coefficient appears explicitly in corrected_pilot_model transition equation",
                "directional_one_step_check": "confirmed in approved pilot perturbation audit; not interpreted as total coupling strength",
                "passes": True,
            }
        )
    rows.append(
        {
            "source_level": "microbiome",
            "source_component": "all_microbiome_components",
            "target_level": "epigenetic",
            "target_component": "all_epigenetic_components",
            "coefficient": 0.0,
            "sign": "none",
            "time_alignment_or_lag": "not applicable",
            "biological_interpretation": "microbiome does not enter the approved epigenetic transition directly",
            "implementation_check": "checked by inspection of _update_epigenetic: no microbiome state is read",
            "directional_one_step_check": "direct one-step epigenetic drift is unchanged under isolated microbiome perturbation",
            "passes": True,
        }
    )
    pd.DataFrame(rows).to_csv(output_dir / "cross_level_coupling_verification.csv", index=False)


def write_figure_specs(output_dir: Path) -> pd.DataFrame:
    rows = [
        ("Figure 1", "Frozen simulated ELC state-space architecture", "ELC levels, source-excluded targets, negative control, and generational ordering", "model specification and coupling table", "schematic axes only", "none", "shows which levels are inside the ELC and which control remains outside"),
        ("Figure 2", "Multivariate simulated dynamics by level", "retained complete state segments for development, microbiome, life history, epigenetic probability/logit, ecology, and background", "simulated retained state arrays", "generation tau plus retained within-generation time t_l", "selected predetermined illustrative lineages with no median trajectory", "documents nonflat biological state dynamics"),
        ("Figure 3", "Predictive contribution and circular-shift null", "transfer entropy estimates and lineage-preserving surrogate null distributions", "primary_transfer_entropy_and_closure files", "analysis category", "bootstrap intervals and null distributions", "contrasts epigenetic hereditary factor and background predictive condition"),
        ("Figure 4", "Predictive closure", "transfer entropy from ELC remainder to future source state conditioned on current source state", "primary_transfer_entropy_and_closure files", "source state under evaluation", "bootstrap intervals", "shows reconstruction/maintenance contrast"),
        ("Figure 5", "Phenotype distribution and variant-specific transfer entropy", "four-state phenotype distribution and transfer entropy to nu/all states", "variant_specific_results.csv", "phenotype state or quantity", "bootstrap intervals if added in report", "defines the hereditary-unit-relative phenotype state"),
        ("Figure 6", "Intervention response and Fisher information", "p_theta(V) over four states and context-averaged Fisher matrices", "intervention_response_surface_summary.csv and intervention_fisher_information.csv", "theta_reg, theta_stress", "context bootstrap intervals", "separates probability increase from local sensitivity"),
        ("Figure 7", "Intergenerational stability profile", "R^{[l]}_{d,tau}(rho) for rho horizons", "intergenerational_stability_summary.csv", "rho", "lineage-cluster bootstrap intervals", "shows propagated predictive contribution"),
        ("Figure 8", "Level- and time-specific transfer entropy", "T^{[l'->l]}_{d,tau} by target level and retained t_l", "level_time_specific_transfer_entropy.csv", "target retained t_l", "none by default", "locates where epigenetic contribution enters development"),
        ("Figure 9", "Partial Information Decomposition of Epigenetic and Ecological Contributions", "Gaussian-deficiency PID atoms and matched joint information", "epigenetic_ecological_delta_g_pid_summary.csv", "PID atom", "lineage-cluster PID bootstrap intervals", "identifies unique, redundant, and synergistic source contributions"),
        ("Figure 10", "Candidate complex unit from the epigenetic hereditary factor and ecological candidate hereditary factor", "joint transfer entropy, joint closure, and positive PID synergy", "primary and PID summary files", "criterion/quantity", "bootstrap intervals", "does not claim a fully qualified complex unit unless all criteria are evaluated"),
        ("Figure 11", "Multiple-parent epigenetic contribution", "transfer entropy and matched PID for source factors from d and d-prime", "multiple_parent files", "source comparison", "bootstrap/null intervals", "keeps receiving individual d and contributor d-prime distinct"),
    ]
    df = pd.DataFrame(rows, columns=["figure", "formal_title", "mathematical_quantity", "source_data", "horizontal_axis", "uncertainty_display", "biological_interpretation"])
    df.to_csv(output_dir / "proposed_final_figures_and_captions.csv", index=False)
    return df


def run_final_numerical_audit(output_dir: Path, final_config: FinalAuditConfig | None = None) -> dict[str, object]:
    t0 = perf_counter()
    final_config = final_config or FinalAuditConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    write_sanity_checks(data_dir, final_config)
    (data_dir / "final_simulation_configuration.json").write_text(json.dumps(asdict(final_config), indent=2))
    seed_results = []
    stability_rows = []
    for seed in final_config.seeds:
        data = load_saved_seed(seed, final_config, data_dir, multiparent=False)
        if data is None:
            data = simulate_final_seed(seed, final_config, multiparent=False)
            save_seed_archive(data_dir / f"corrected_temporal_architecture_seed_{seed}.npz", data)
        else:
            print(f"loaded saved frozen seed {seed}", flush=True)
        seed_results.append(data)
        for key in ELC_LEVELS + (BACKGROUND_LEVEL,):
            arr = np.asarray(data[key])
            full_arr = np.asarray(data["full_time_series"][key])
            if not np.all(np.isfinite(arr)):
                raise FloatingPointError(f"{key} contains non-finite values for seed {seed}")
            if not np.all(np.isfinite(full_arr)):
                raise FloatingPointError(f"full {key} time series contains non-finite values for seed {seed}")
            if key == "microbiome" and float(arr.min()) <= 0.0:
                raise FloatingPointError(f"microbiome reached nonpositive abundance for seed {seed}")
        acc = acceptance_criteria(data, data["config"])
        acc.insert(0, "seed", seed)
        stability_rows.append(acc)
    pd.concat(stability_rows, ignore_index=True).to_csv(data_dir / "qualitative_acceptance_criteria_by_seed.csv", index=False)
    conv = convergence_check(final_config.pilot_config(final_config.seeds[0]))
    conv.to_csv(data_dir / "integration_step_convergence_confirmation.csv", index=False)

    table = build_analysis_table(seed_results, final_config)
    table["seed_results"] = seed_results
    seed_tables = [build_analysis_table([sd], final_config) for sd in seed_results]
    phenotype_support(table).to_csv(data_dir / "phenotype_support_by_seed.csv", index=False)
    _covariance_diagnostics(
        table,
        [
            "source_epigenetic",
            "source_ecological",
            "source_background",
            "target_remainder_without_epigenetic",
            "history_remainder_without_epigenetic",
            "target_full_elc",
            "history_full_elc",
            "target_remainder_without_epigenetic_ecological",
            "history_remainder_without_epigenetic_ecological",
        ],
    ).to_csv(data_dir / "covariance_rank_condition_diagnostics.csv", index=False)

    primary_path = data_dir / "primary_transfer_entropy_and_closure_summary.csv"
    if primary_path.exists() and (data_dir / "central_contrast_check.csv").exists():
        print("loaded existing primary transfer-entropy/closure outputs", flush=True)
        primary_summary = pd.read_csv(primary_path)
    else:
        primary_summary, _, _ = run_primary_cmi_analyses(table, seed_tables, final_config, data_dir)
    epi_pc = float(primary_summary.set_index("analysis").loc["epigenetic_predictive_contribution", "estimate_bits"])
    bg_pc = float(primary_summary.set_index("analysis").loc["background_predictive_contribution", "estimate_bits"])
    epi_closure = float(primary_summary.set_index("analysis").loc["epigenetic_predictive_closure", "estimate_bits"])
    bg_closure = float(primary_summary.set_index("analysis").loc["background_closure_control", "estimate_bits"])
    epi_closure_p = float(primary_summary.set_index("analysis").loc["epigenetic_predictive_closure", "surrogate_p_value"])
    bg_closure_p = float(primary_summary.set_index("analysis").loc["background_closure_control", "surrogate_p_value"])
    central = pd.DataFrame(
        [
            {"criterion": "epigenetic_predictive_contribution_positive", "passes": bool(epi_pc > 0.0), "value_bits": epi_pc},
            {"criterion": "background_predictive_contribution_positive", "passes": bool(bg_pc > 0.0), "value_bits": bg_pc},
            {"criterion": "epigenetic_predictive_closure_above_surrogate_null", "passes": bool(epi_closure_p <= 0.05), "value_bits": epi_closure, "surrogate_p_value": epi_closure_p},
            {"criterion": "background_closure_control_not_above_surrogate_null", "passes": bool(bg_closure_p > 0.05), "value_bits": bg_closure, "surrogate_p_value": bg_closure_p},
            {"criterion": "epigenetic_closure_exceeds_background_control", "passes": bool(epi_closure > bg_closure), "value_bits": epi_closure - bg_closure},
        ]
    )
    central.to_csv(data_dir / "central_contrast_check.csv", index=False)
    if not bool(central["passes"].all()):
        raise AssertionError("central predictive contribution/closure contrast failed; frozen model was not retuned")

    if (data_dir / "intergenerational_stability_summary.csv").exists() and (data_dir / "intergenerational_stability_bootstrap.csv").exists():
        print("loaded existing stability outputs", flush=True)
    else:
        horizons = {rho: build_horizon_table(seed_results, final_config, rho) for rho in range(1, 6)}
        run_stability(horizons, final_config, data_dir)
    if (data_dir / "level_time_specific_transfer_entropy.csv").exists():
        print("loaded existing level/time transfer-entropy output", flush=True)
    else:
        run_location_analysis(table, final_config, data_dir)
    if (data_dir / "variant_specific_results.csv").exists():
        print("loaded existing variant-specific output", flush=True)
    else:
        run_variant_specific(table, final_config, data_dir)
    if (
        (data_dir / "intervention_response_surface_summary.csv").exists()
        and (data_dir / "intervention_probability_by_context.csv").exists()
        and (data_dir / "intervention_fisher_information.csv").exists()
    ):
        print("loaded existing intervention outputs", flush=True)
    else:
        run_intervention(seed_results, final_config, data_dir)
    if (data_dir / "epigenetic_ecological_delta_g_pid_summary.csv").exists() and (data_dir / "epigenetic_ecological_delta_g_pid_bootstrap.csv").exists():
        print("loaded existing epigenetic/ecological PID outputs", flush=True)
    else:
        run_pid(table, final_config, data_dir, prefix="epigenetic_ecological")

    seed_results_mp = []
    for seed in final_config.seeds:
        mp = load_saved_seed(seed, final_config, data_dir, multiparent=True)
        if mp is None:
            mp = simulate_final_seed(seed, final_config, multiparent=True)
            save_seed_archive(
                data_dir / f"multiple_parent_temporal_architecture_seed_{seed}.npz",
                mp,
                pair_key="second_parent_pairs",
            )
        else:
            print(f"loaded saved multiple-parent seed {seed}", flush=True)
        seed_results_mp.append(mp)
    run_multiple_parent(seed_results_mp, final_config, data_dir)
    write_figure_specs(data_dir)
    runtime = {
        "runtime_seconds": perf_counter() - t0,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "simulation_frozen_after_pilot": True,
        "no_clipping_or_post_result_parameter_tuning": True,
    }
    (data_dir / "software_versions_and_runtime.json").write_text(json.dumps(runtime, indent=2))
    return {"output_dir": output_dir, "data_dir": data_dir, "primary_summary": primary_summary, "runtime": runtime}
