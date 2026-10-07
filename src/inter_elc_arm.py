from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence
import hashlib
import json
import platform
import shutil

import numpy as np
import pandas as pd

from .corrected_numerical_audit import (
    CmiFamily,
    CorrectedAuditConfig,
    SourceBlock,
    _bootstrap_family,
    _cov,
    _family_estimates_from_cov,
    _null_family,
    _ranked_joint_arrays,
    _seed_level_family,
    _summarize_family,
)
from .corrected_pilot_model import (
    COMPONENTS,
    PilotConfig,
    _par_float,
    apply_parameter_overrides,
    build_event_schedule,
    convergence_check,
    extract_analysis_segment,
    level_segment_indices,
    level_timestamps,
    reproductive_indices,
    reproductive_state,
    sigmoid,
)
from .final_numerical_audit import (
    BACKGROUND_LEVEL,
    ELC_LEVELS,
    PHENOTYPE_LABELS,
    _empty_arrays_vec,
    _empty_full_arrays_vec,
    _empty_reproductive_arrays_vec,
    _founder_states_vec,
    _history,
    _integrate_generation_vec,
    _next_generation_start_vec,
    _params_for_config,
    _segment,
    _concat_levels,
    phenotype_labels_for_generation,
    phenotype_support,
    save_seed_archive,
)
from .recalibration_audit import calibration_candidates, epigenetic_probability_range


INTER_ELC_CALIBRATION_SEEDS = (101, 202, 303, 404)
PROPOSED_UNTOUCHED_INTER_ELC_FINAL_SEEDS = (6311, 7541, 8779, 9901)
CORRECTED_INTER_ELC_FINAL_SEEDS_V2 = (16319, 17431, 18637, 19843)
ACTIVE_SOURCE_LEVELS = ("epigenetic", "microbiome", "ecological")
SOURCE_LEVELS_WITH_ZERO_B = ("development", "life_history")
INTER_ELC_FINAL_SEEDS = PROPOSED_UNTOUCHED_INTER_ELC_FINAL_SEEDS


@dataclass(frozen=True)
class InterELCBMatrices:
    b_micro: float
    b_eco: float
    b_epi_reg: float = 0.22
    b_epi_stress: float = 0.20
    epi_interaction_reg: float = 0.04
    epi_interaction_stress: float = 0.03

    def matrices(self) -> dict[str, np.ndarray]:
        return {
            "development": np.zeros((5, 5), dtype=float),
            "microbiome": float(self.b_micro) * np.eye(5, dtype=float),
            "life_history": np.zeros((3, 3), dtype=float),
            "epigenetic": np.diag([float(self.b_epi_reg), float(self.b_epi_stress)]),
            "ecological": float(self.b_eco) * np.eye(3, dtype=float),
        }


@dataclass(frozen=True)
class InterELCSourceMeans:
    epigenetic: tuple[float, float]
    log_microbiome: tuple[float, float, float, float, float]
    ecological: tuple[float, float, float]

    @classmethod
    def zero(cls) -> "InterELCSourceMeans":
        return cls(epigenetic=(0.0, 0.0), log_microbiome=(0.0, 0.0, 0.0, 0.0, 0.0), ecological=(0.0, 0.0, 0.0))

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {
            "epigenetic": np.asarray(self.epigenetic, dtype=float),
            "log_microbiome": np.asarray(self.log_microbiome, dtype=float),
            "ecological": np.asarray(self.ecological, dtype=float),
        }


def scaled_epigenetic_B(scale: float, *, b_micro: float = 0.0, b_eco: float = 0.0) -> InterELCBMatrices:
    return InterELCBMatrices(
        b_micro=float(b_micro),
        b_eco=float(b_eco),
        b_epi_reg=0.22 * float(scale),
        b_epi_stress=0.20 * float(scale),
        epi_interaction_reg=0.04 * float(scale),
        epi_interaction_stress=0.03 * float(scale),
    )


def frozen_epigenetic_only_inter_elc_B() -> InterELCBMatrices:
    return InterELCBMatrices(
        b_micro=0.0,
        b_eco=0.0,
        b_epi_reg=0.077,
        b_epi_stress=0.070,
        epi_interaction_reg=0.014,
        epi_interaction_stress=0.0105,
    )


def approved_inter_elc_source_means() -> InterELCSourceMeans:
    return InterELCSourceMeans(
        epigenetic=(0.3471842314766994, -0.05490154474202076),
        log_microbiome=(-0.005310546265322313, -0.10786539896460867, -0.8886079326074503, -0.8346971481478874, -0.05424769765681721),
        ecological=(1.0177854748159245, 0.8253228542701012, 0.9475692050949974),
    )


def final_inter_elc_seeds_are_untouched(prior_seed_sets: Sequence[Sequence[int]] | None = None) -> bool:
    prior = set(INTER_ELC_CALIBRATION_SEEDS)
    if prior_seed_sets is not None:
        for seed_set in prior_seed_sets:
            prior.update(int(x) for x in seed_set)
    return not bool(set(INTER_ELC_FINAL_SEEDS) & prior)


def one_to_one_derangement(n_lineages: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed) + 31579)
    base = np.arange(int(n_lineages), dtype=int)
    for _ in range(10_000):
        perm = rng.permutation(base)
        if np.all(perm != base):
            return perm
    raise RuntimeError("failed to generate one-to-one derangement without self-pairs")


def inter_elc_b_matrix_inventory(b: InterELCBMatrices) -> pd.DataFrame:
    matrices = b.matrices()
    rows: list[dict[str, object]] = []
    interpretations = {
        ("epigenetic", 0): "source regulatory-mark logit contributes to focal regulatory-mark reconstruction",
        ("epigenetic", 1): "source stress-memory logit contributes to focal stress-memory reconstruction",
        ("microbiome", 0): "source support-guild log abundance contributes to focal support-guild carryover",
        ("microbiome", 1): "source fermenter-guild log abundance contributes to focal fermenter-guild carryover",
        ("microbiome", 2): "source stress-tolerant-guild log abundance contributes to focal stress-tolerant-guild carryover",
        ("microbiome", 3): "source opportunist-guild log abundance contributes to focal opportunist-guild carryover",
        ("microbiome", 4): "source cross-feeder-guild log abundance contributes to focal cross-feeder-guild carryover",
        ("ecological", 0): "source soil or nest organic state contributes to focal ecological reconstruction",
        ("ecological", 1): "source food-resource enrichment contributes to focal ecological reconstruction",
        ("ecological", 2): "source microclimate buffering contributes to focal ecological reconstruction",
    }
    for level, matrix in matrices.items():
        comps = COMPONENTS[level]
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = float(matrix[i, j])
                if abs(value) <= 0.0:
                    continue
                rows.append(
                    {
                        "level_l": level,
                        "source_component_d_prime": comps[j],
                        "target_component_d": comps[i],
                        "B_matrix_entry": value,
                        "sign": "+" if value > 0 else "-",
                        "timing": "intergenerational initialization only",
                        "mathematical_term": f"{value:.6g} (x_d_prime_{level}_{j}_final - mu_source_{level}_{j})",
                        "biological_interpretation": interpretations.get((level, i), "same-level inter-ELC contribution"),
                    }
                )
    rows.extend(
        [
            {
                "level_l": "development",
                "source_component_d_prime": "all developmental components",
                "target_component_d": "all developmental components",
                "B_matrix_entry": 0.0,
                "sign": "zero",
                "timing": "not used",
                "mathematical_term": "B^{[l_development]}=0",
                "biological_interpretation": "no direct same-level developmental transmission from d-prime; effects are indirect through active source levels",
            },
            {
                "level_l": "life_history",
                "source_component_d_prime": "all life-history components",
                "target_component_d": "all life-history components",
                "B_matrix_entry": 0.0,
                "sign": "zero",
                "timing": "not used",
                "mathematical_term": "B^{[l_life-history]}=0",
                "biological_interpretation": "no direct same-level life-history transmission from d-prime; effects are indirect through active source levels",
            },
        ]
    )
    rows.append(
        {
            "level_l": "epigenetic",
            "source_component_d_prime": "z_regulatory_mark with focal z_regulatory_mark",
            "target_component_d": "z_regulatory_mark",
            "B_matrix_entry": np.nan,
            "sign": "+",
            "timing": "intergenerational initialization only",
            "mathematical_term": f"{b.epi_interaction_reg:.6g} z_d,reg z_d_prime,reg",
            "biological_interpretation": "nonlinear two-entity epigenetic interaction; documented separately from the fixed linear B matrix",
        }
    )
    rows.append(
        {
            "level_l": "epigenetic",
            "source_component_d_prime": "z_stress_memory_mark with focal z_stress_memory_mark",
            "target_component_d": "z_stress_memory_mark",
            "B_matrix_entry": np.nan,
            "sign": "+",
            "timing": "intergenerational initialization only",
            "mathematical_term": f"{b.epi_interaction_stress:.6g} z_d,stress z_d_prime,stress",
            "biological_interpretation": "nonlinear two-entity epigenetic interaction; documented separately from the fixed linear B matrix",
        }
    )
    return pd.DataFrame(rows)


def inter_elc_level_summary(b: InterELCBMatrices) -> pd.DataFrame:
    inventory = inter_elc_b_matrix_inventory(b)
    nonzero = inventory[inventory["B_matrix_entry"].fillna(0.0).abs() > 0.0]
    rows = []
    for level in ELC_LEVELS:
        rows.append(
            {
                "level_l": level,
                "n_nonzero_B_entries": int((nonzero["level_l"] == level).sum()),
                "B_matrix": np.array2string(b.matrices()[level], precision=3, separator=", "),
                "role": {
                    "development": "no direct same-level inter-ELC term",
                    "microbiome": "partial same-guild microbiome transmission on log-abundance scale",
                    "life_history": "no direct same-level inter-ELC term",
                    "epigenetic": "two-source epigenetic reconstruction at generation initialization",
                    "ecological": "partial continuity of source-modified ecological state",
                }[level],
            }
        )
    return pd.DataFrame(rows)


def inter_elc_calibration_grid() -> tuple[InterELCBMatrices, ...]:
    return tuple(
        InterELCBMatrices(b_micro=b_micro, b_eco=b_eco)
        for b_micro in (0.10, 0.15, 0.20)
        for b_eco in (0.10, 0.15, 0.20)
    )


def inter_elc_epigenetic_isolation_grid() -> tuple[tuple[str, float, InterELCBMatrices], ...]:
    rows = []
    for scale in (0.20, 0.35, 0.50, 0.65):
        label = f"stage1_epi_{int(round(scale * 100)):02d}pct"
        rows.append((label, scale, scaled_epigenetic_B(scale, b_micro=0.0, b_eco=0.0)))
    return tuple(rows)


def inter_elc_small_micro_eco_grid(epigenetic_scale: float) -> tuple[tuple[str, float, InterELCBMatrices], ...]:
    rows = []
    for b_micro, b_eco in ((0.01, 0.01), (0.025, 0.01), (0.01, 0.025), (0.025, 0.025), (0.05, 0.01), (0.01, 0.05)):
        label = f"stage2_epi_{int(round(epigenetic_scale * 100)):02d}pct_B_micro_{b_micro:.3f}_B_eco_{b_eco:.3f}"
        label = label.replace(".", "p")
        rows.append((label, epigenetic_scale, scaled_epigenetic_B(epigenetic_scale, b_micro=b_micro, b_eco=b_eco)))
    return tuple(rows)


def inter_elc_coupling_magnitude(b: InterELCBMatrices) -> float:
    return float(
        np.sqrt(
            b.b_epi_reg**2
            + b.b_epi_stress**2
            + b.epi_interaction_reg**2
            + b.epi_interaction_stress**2
            + 5.0 * b.b_micro**2
            + 3.0 * b.b_eco**2
        )
    )


def _next_generation_start_inter_elc_vec(
    prev: Mapping[str, np.ndarray],
    pairs: np.ndarray,
    rng: np.random.Generator,
    config: PilotConfig,
    par: Mapping[str, np.ndarray],
    b: InterELCBMatrices,
    source_means: InterELCSourceMeans | None = None,
) -> dict[str, np.ndarray]:
    n = config.n_lineages
    pairs = np.asarray(pairs, dtype=int)
    if pairs.shape != (n,):
        raise AssertionError("pairs must contain one d-prime index for each focal lineage")
    if np.any(pairs == np.arange(n)):
        raise AssertionError("inter-ELC derangement contains a self-pair")
    if np.unique(pairs).size != n:
        raise AssertionError("inter-ELC derangement must use every d-prime exactly once")

    dev_final = prev["development"]
    micro_final = prev["microbiome"]
    life_final = prev["life_history"]
    epi_final = prev["epigenetic"]
    eco_final = prev["ecological"]
    bg_final = prev["background"]
    micro_dp = micro_final[pairs]
    epi_dp = epi_final[pairs]
    eco_dp = eco_final[pairs]
    source_mean_arrays = (source_means or InterELCSourceMeans.zero()).as_arrays()
    epi_dp_dev = epi_dp - source_mean_arrays["epigenetic"]
    log_micro_dp_dev = np.log(micro_dp) - source_mean_arrays["log_microbiome"]
    eco_dp_dev = eco_dp - source_mean_arrays["ecological"]
    noise = config.noise_scale

    bg_start = bg_final @ par["phi_bg"].T + noise * rng.normal(0.0, [0.20, 0.20], size=(n, 2))

    epi_linear = epi_dp_dev @ b.matrices()["epigenetic"].T
    epi_interaction = np.column_stack(
        [
            b.epi_interaction_reg * epi_final[:, 0] * epi_dp[:, 0],
            b.epi_interaction_stress * epi_final[:, 1] * epi_dp[:, 1],
        ]
    )
    epi_start = (
        par["rho_epi"] * epi_final
        + epi_linear
        + epi_interaction
        + noise * rng.normal(0.0, _par_float(par, "epi_transmission_noise_sd", 0.10), size=(n, 2))
    )

    eco_recon_scale = _par_float(par, "ecological_reconstruction_elc_scale")
    micro_eco_recon_scale = _par_float(par, "microbiome_to_ecology_reconstruction_scale")
    eco_start = np.column_stack(
        [
            0.64 * eco_final[:, 0] + 0.12 * eco_final[:, 1] + eco_recon_scale * (micro_eco_recon_scale * 0.18 * np.log1p(micro_final[:, 0]) + 0.18 * life_final[:, 2]),
            0.60 * eco_final[:, 1] + 0.10 * eco_final[:, 2] + eco_recon_scale * (0.18 * life_final[:, 2] + 0.10 * np.tanh(dev_final[:, 1])),
            0.62 * eco_final[:, 2] + 0.14 * eco_final[:, 0] + eco_recon_scale * (0.12 * life_final[:, 0] + micro_eco_recon_scale * 0.10 * np.log1p(micro_final[:, 4])),
        ]
    )
    eco_start += eco_dp_dev @ b.matrices()["ecological"].T
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
    micro_log_start += log_micro_dp_dev @ b.matrices()["microbiome"].T
    micro_start = np.exp(micro_log_start + noise * rng.normal(0.0, 0.12, size=(n, 5)))

    epi_life_start_scale = _par_float(par, "epi_life_start_scale")
    epi_life_start_maturation_scale = epi_life_start_scale * _par_float(par, "epi_life_start_maturation_scale")
    epi_life_start_growth_scale = epi_life_start_scale * _par_float(par, "epi_life_start_growth_scale")
    epi_life_start_allocation_scale = epi_life_start_scale * _par_float(par, "epi_life_start_allocation_scale")
    eco_life_start_scale = _par_float(par, "ecology_life_start_scale")
    life_start_logits = np.column_stack(
        [
            -1.30 + epi_life_start_maturation_scale * 0.20 * epi_centered[:, 0] + eco_life_start_scale * 0.10 * eco_start[:, 2] - 0.08 * bg_start[:, 1],
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


def zero_b_matrices() -> InterELCBMatrices:
    return InterELCBMatrices(
        b_micro=0.0,
        b_eco=0.0,
        b_epi_reg=0.0,
        b_epi_stress=0.0,
        epi_interaction_reg=0.0,
        epi_interaction_stress=0.0,
    )


def zero_b_initializer_matches_primary(config: PilotConfig, par: Mapping[str, np.ndarray], seed: int = 12345) -> bool:
    rng = np.random.default_rng(seed)
    prev = _founder_states_vec(rng, config, par)
    pairs = one_to_one_derangement(config.n_lineages, seed)
    ordinary = _next_generation_start_vec(prev, np.random.default_rng(seed + 1), config, par)
    inter_zero = _next_generation_start_inter_elc_vec(prev, pairs, np.random.default_rng(seed + 1), config, par, zero_b_matrices(), InterELCSourceMeans.zero())
    return all(np.allclose(ordinary[level], inter_zero[level]) for level in ELC_LEVELS + (BACKGROUND_LEVEL,))


def simulate_inter_elc_seed(
    seed: int,
    final_config: CorrectedAuditConfig,
    b: InterELCBMatrices,
    source_means: InterELCSourceMeans | None = None,
) -> dict[str, object]:
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
    pairs = one_to_one_derangement(config.n_lineages, config.seed)
    for tau in range(config.n_generations):
        prev_reproductive = {key: reproductive_arrays[key][:, tau] for key in arrays}
        start = _next_generation_start_inter_elc_vec(prev_reproductive, pairs, rng, config, par, b, source_means)
        retained = _integrate_generation_vec(start, tau + 1, rng, config, par, schedule)
        for key in arrays:
            full_arrays[key][:, tau + 1] = retained[key]
            arrays[key][:, tau + 1] = extract_analysis_segment(retained[key], key, config)
            reproductive_arrays[key][:, tau + 1] = reproductive_state(retained[key], key, config)
    return {
        "config": config,
        "seed": seed,
        "parameters": par,
        "inter_elc_pairs": pairs,
        "B": b,
        "full_time_series": full_arrays,
        "reproductive_states": reproductive_arrays,
        "timestamps": level_timestamps(config),
        "segment_indices": level_segment_indices(config),
        "reproductive_indices": reproductive_indices(config),
        **arrays,
    }


def estimate_source_state_means(seed_results: Sequence[Mapping[str, object]], config: CorrectedAuditConfig) -> InterELCSourceMeans:
    taus = list(range(config.source_tau_start, config.source_tau_stop + 1))
    epi_parts = []
    log_micro_parts = []
    eco_parts = []
    for seed_data in seed_results:
        pairs = np.asarray(seed_data["inter_elc_pairs"], dtype=int)
        for tau in taus:
            epi_parts.append(np.asarray(seed_data["reproductive_states"]["epigenetic"])[:, tau, :][pairs])
            log_micro_parts.append(np.log(np.asarray(seed_data["reproductive_states"]["microbiome"])[:, tau, :][pairs]))
            eco_parts.append(np.asarray(seed_data["reproductive_states"]["ecological"])[:, tau, :][pairs])
    epi_mean = np.vstack(epi_parts).mean(axis=0)
    log_micro_mean = np.vstack(log_micro_parts).mean(axis=0)
    eco_mean = np.vstack(eco_parts).mean(axis=0)
    return InterELCSourceMeans(
        epigenetic=tuple(float(x) for x in epi_mean),
        log_microbiome=tuple(float(x) for x in log_micro_mean),
        ecological=tuple(float(x) for x in eco_mean),
    )


def estimate_source_state_means_sequential(config: CorrectedAuditConfig) -> InterELCSourceMeans:
    epi_sum = np.zeros(2, dtype=float)
    log_micro_sum = np.zeros(5, dtype=float)
    eco_sum = np.zeros(3, dtype=float)
    count = 0
    taus = list(range(config.source_tau_start, config.source_tau_stop + 1))
    for seed in config.seeds:
        data = simulate_inter_elc_seed(seed, config, zero_b_matrices(), InterELCSourceMeans.zero())
        pairs = np.asarray(data["inter_elc_pairs"], dtype=int)
        for tau in taus:
            epi = np.asarray(data["reproductive_states"]["epigenetic"])[:, tau, :][pairs]
            micro = np.asarray(data["reproductive_states"]["microbiome"])[:, tau, :][pairs]
            eco = np.asarray(data["reproductive_states"]["ecological"])[:, tau, :][pairs]
            epi_sum += epi.sum(axis=0)
            log_micro_sum += np.log(micro).sum(axis=0)
            eco_sum += eco.sum(axis=0)
            count += epi.shape[0]
        del data
    if count == 0:
        raise AssertionError("no calibration states were available for source centering")
    return InterELCSourceMeans(
        epigenetic=tuple(float(x) for x in epi_sum / count),
        log_microbiome=tuple(float(x) for x in log_micro_sum / count),
        ecological=tuple(float(x) for x in eco_sum / count),
    )


def source_state_means_table(means: InterELCSourceMeans) -> pd.DataFrame:
    arrays = means.as_arrays()
    rows: list[dict[str, object]] = []
    for component, value in zip(COMPONENTS["epigenetic"], arrays["epigenetic"]):
        rows.append({"level": "epigenetic", "component": component, "scale": "latent logit", "source_mean": float(value)})
    for component, value in zip(COMPONENTS["microbiome"], arrays["log_microbiome"]):
        rows.append({"level": "microbiome", "component": component, "scale": "log abundance", "source_mean": float(value)})
    for component, value in zip(COMPONENTS["ecological"], arrays["ecological"]):
        rows.append({"level": "ecological", "component": component, "scale": "state value", "source_mean": float(value)})
    return pd.DataFrame(rows)


def _make_meta(seed: int, seed_position: int, n_lineages: int, taus: Sequence[int]) -> pd.DataFrame:
    rows = []
    for tau in taus:
        for d in range(n_lineages):
            rows.append({"seed": seed, "lineage_id": d, "unit_id": seed_position * n_lineages + d, "tau": tau})
    return pd.DataFrame(rows)


def build_inter_elc_table(seed_results: Sequence[Mapping[str, object]], config: CorrectedAuditConfig) -> dict[str, object]:
    taus = list(range(config.source_tau_start, config.source_tau_stop + 1))
    metas: list[pd.DataFrame] = []
    cols: dict[str, list[np.ndarray]] = {
        "target_full_elc": [],
        "history_full_elc": [],
        "source_dprime_development": [],
        "source_dprime_microbiome": [],
        "source_dprime_life_history": [],
        "source_dprime_epigenetic": [],
        "source_dprime_ecological": [],
    }
    for level in ELC_LEVELS:
        cols[f"target_{level}_future"] = []
    labels: list[str] = []
    pair_rows: list[dict[str, object]] = []
    for seed_position, seed_data in enumerate(seed_results):
        seed = int(seed_data["seed"])
        pcfg: PilotConfig = seed_data["config"]
        pairs = np.asarray(seed_data["inter_elc_pairs"], dtype=int)
        metas.append(_make_meta(seed, seed_position, pcfg.n_lineages, taus))
        for tau in taus:
            cols["target_full_elc"].append(_concat_levels(seed_data, tau + 1, ELC_LEVELS))
            cols["history_full_elc"].append(_history(seed_data, tau, ELC_LEVELS, config.history_order))
            for level in ELC_LEVELS:
                source = _segment(seed_data, level, tau)[pairs]
                cols[f"source_dprime_{level}"].append(source)
                cols[f"target_{level}_future"].append(_segment(seed_data, level, tau + 1))
            labels.extend(phenotype_labels_for_generation(seed_data, tau + 1, pcfg).tolist())
        for d, dp in enumerate(pairs):
            pair_rows.append({"seed": seed, "lineage_id_d": int(d), "source_lineage_d_prime": int(dp), "same_lineage": bool(d == int(dp))})
    table = {"meta": pd.concat(metas, ignore_index=True), "variant_label": np.asarray(labels, dtype=object)}
    for key, parts in cols.items():
        table[key] = np.vstack(parts)
    table["inter_elc_pairs"] = pd.DataFrame(pair_rows)
    table["seed_results"] = list(seed_results)
    return table


def inter_elc_family(target_key: str = "target_full_elc", family_name: str = "inter_elc_whole_source") -> CmiFamily:
    sources = tuple(SourceBlock(level, f"source_dprime_{level}") for level in ELC_LEVELS)
    return CmiFamily(
        family_name,
        target_key,
        "history_full_elc",
        sources,
        (
            ("whole_source_elc_from_d_prime", ELC_LEVELS),
            ("dprime_epigenetic_subsystem", ("epigenetic",)),
            ("dprime_microbiome_subsystem", ("microbiome",)),
            ("dprime_ecological_subsystem", ("ecological",)),
        ),
        "inter-ELC contribution from d-prime through documented same-level B matrices",
    )


def estimate_inter_elc_family(
    table: Mapping[str, object],
    seed_tables: Sequence[Mapping[str, object]],
    config: CorrectedAuditConfig,
    family: CmiFamily,
    *,
    output_dir: Path | None,
    prefix: str,
    jitter: float = 0.0,
    n_bootstrap: int | None = None,
    n_null: int | None = None,
    seed_offset: int = 0,
    compute_seed_level: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    joint, slices, source_arrays = _ranked_joint_arrays(table, family, include_generation=True)
    observed = _family_estimates_from_cov(_cov(joint, jitter), slices, family)
    boot = _bootstrap_family(
        joint,
        slices,
        family,
        table["meta"],
        n_boot=config.n_bootstrap if n_bootstrap is None else int(n_bootstrap),
        seed=config.bootstrap_seed + 180_000 + seed_offset,
        jitter=jitter,
    )
    null = _null_family(
        joint,
        slices,
        family,
        table["meta"],
        n_null=config.n_null if n_null is None else int(n_null),
        seed=config.null_seed + 180_000 + seed_offset,
        jitter=jitter,
    )
    source_dims = {label: source_arrays[label].shape[1] for label in source_arrays}
    summary = _summarize_family(
        family,
        observed,
        boot,
        null,
        target_dim=slices["target"].stop - slices["target"].start,
        source_dims=source_dims,
        condition_dim=slices["condition"].stop - slices["condition"].start,
        jitter=jitter,
    )
    seed_df = (
        _seed_level_family(seed_tables, family, jitter=jitter, include_generation=True)
        if compute_seed_level
        else pd.DataFrame(columns=["seed", "family", "analysis", "raw_estimate_bits"])
    )
    if output_dir is not None:
        summary.to_csv(output_dir / f"{prefix}_summary.csv", index=False)
        boot.to_csv(output_dir / f"{prefix}_bootstrap.csv", index=False)
        null.to_csv(output_dir / f"{prefix}_generation_preserving_null.csv", index=False)
        seed_df.to_csv(output_dir / f"{prefix}_by_seed.csv", index=False)
    return summary, boot, null, seed_df


def phenotype_support_for_seed_results(seed_results: Sequence[Mapping[str, object]], config: CorrectedAuditConfig) -> pd.DataFrame:
    table = build_inter_elc_table(seed_results, config)
    return phenotype_support(table)


def phenotype_fold_support(table: Mapping[str, object], n_folds: int = 5) -> pd.DataFrame:
    meta = table["meta"].copy()
    labels = np.asarray(table["variant_label"], dtype=object)
    units = meta["unit_id"].to_numpy(dtype=int)
    fold = units % int(n_folds)
    rows = []
    for f in range(int(n_folds)):
        mask = fold == f
        present = set(labels[mask].tolist())
        for label in PHENOTYPE_LABELS:
            rows.append({"fold": f, "phenotype_state": label, "count": int(np.sum(mask & (labels == label))), "present": label in present})
    return pd.DataFrame(rows)


def inter_elc_numerical_checks(seed_results: Sequence[Mapping[str, object]], config: CorrectedAuditConfig) -> pd.DataFrame:
    rows = []
    for data in seed_results:
        seed = int(data["seed"])
        for level in ELC_LEVELS + (BACKGROUND_LEVEL,):
            arr = np.asarray(data[level], dtype=float)
            rows.append(
                {
                    "seed": seed,
                    "level": level,
                    "all_finite": bool(np.all(np.isfinite(arr))),
                    "minimum": float(np.min(arr)),
                    "maximum": float(np.max(arr)),
                    "standard_deviation": float(np.std(arr)),
                    "positive_microbiome_if_applicable": bool(np.all(arr > 0.0)) if level == "microbiome" else True,
                }
            )
    return pd.DataFrame(rows)


def inter_elc_candidate_pass_table(
    candidate_summary: pd.DataFrame,
    phenotype: pd.DataFrame,
    fold_support: pd.DataFrame,
    epi_ranges: pd.DataFrame,
    numerical: pd.DataFrame,
) -> pd.DataFrame:
    obs = candidate_summary.set_index("analysis")
    phen_all = phenotype[phenotype["seed"].astype(str) == "all"].set_index("phenotype_state")
    all_probs = phen_all["proportion"].to_numpy(dtype=float)
    complete_excess = float(obs.loc["whole_source_elc_from_d_prime", "null_excess_bits"])
    complete_p = float(obs.loc["whole_source_elc_from_d_prime", "generation_preserving_surrogate_p_value"])
    epi_excess = float(obs.loc["dprime_epigenetic_subsystem", "null_excess_bits"])
    epi_p = float(obs.loc["dprime_epigenetic_subsystem", "generation_preserving_surrogate_p_value"])
    subsystem_raw = obs.loc[
        ["dprime_epigenetic_subsystem", "dprime_microbiome_subsystem", "dprime_ecological_subsystem"],
        "raw_estimate_bits",
    ].to_numpy(dtype=float)
    complete_raw = float(obs.loc["whole_source_elc_from_d_prime", "raw_estimate_bits"])
    rows = [
        ("all_phenotype_states_represented", float(np.min(all_probs)), "all proportions > 0", bool(np.all(all_probs > 0.0))),
        ("all_phenotype_classes_in_each_fold", float(fold_support["present"].mean()), "all folds contain all classes", bool(fold_support["present"].all())),
        ("minimum_phenotype_probability_ge_0.05", float(np.min(all_probs)), ">=0.05", bool(np.min(all_probs) >= 0.05)),
        ("maximum_phenotype_probability_le_0.70", float(np.max(all_probs)), "<=0.70", bool(np.max(all_probs) <= 0.70)),
        ("epigenetic_probabilities_not_saturated_low", float(epi_ranges["minimum"].min()), ">0.01", bool(epi_ranges["minimum"].min() > 0.01)),
        ("epigenetic_probabilities_not_saturated_high", float(epi_ranges["maximum"].max()), "<0.99", bool(epi_ranges["maximum"].max() < 0.99)),
        ("microbiome_abundances_positive", float(numerical.loc[numerical["level"] == "microbiome", "minimum"].min()), ">0", bool(numerical.loc[numerical["level"] == "microbiome", "positive_microbiome_if_applicable"].all())),
        ("microbiome_abundances_not_divergent", float(numerical.loc[numerical["level"] == "microbiome", "maximum"].max()), "<50", bool(numerical.loc[numerical["level"] == "microbiome", "maximum"].max() < 50.0)),
        ("ecological_states_finite", float(numerical.loc[numerical["level"] == "ecological", "maximum"].max()), "finite", bool(numerical.loc[numerical["level"] == "ecological", "all_finite"].all())),
        ("complete_source_detectable", complete_excess, "null-excess >0 and p<=0.05", bool(complete_excess > 0.0 and complete_p <= 0.05)),
        ("complete_source_null_excess_ge_0.01", complete_excess, ">=0.01 bits", bool(complete_excess >= 0.01)),
        ("epigenetic_subsystem_detectable", epi_excess, "null-excess >0 and p<=0.05", bool(epi_excess > 0.0 and epi_p <= 0.05)),
        ("complete_source_raw_ge_active_subsystem_raw", complete_raw - float(np.max(subsystem_raw)), ">=0 bits", bool(complete_raw + 1e-10 >= float(np.max(subsystem_raw)))),
    ]
    return pd.DataFrame(
        [{"criterion": c, "observed_value": v, "target": target, "passes": bool(passes)} for c, v, target, passes in rows]
    )


def inter_elc_final_acceptance_table(
    summary: pd.DataFrame,
    phenotype: pd.DataFrame,
    fold_support: pd.DataFrame,
    epi_ranges: pd.DataFrame,
    numerical: pd.DataFrame,
    pair_summary: pd.DataFrame,
) -> pd.DataFrame:
    obs = summary.set_index("analysis")
    phen_all = phenotype[phenotype["seed"].astype(str) == "all"].set_index("phenotype_state")
    all_probs = phen_all["proportion"].to_numpy(dtype=float)
    complete = obs.loc["whole_source_elc_from_d_prime"]
    epi = obs.loc["dprime_epigenetic_subsystem"]
    raw_values = obs["raw_estimate_bits"].to_dict()
    whole_raw = float(raw_values["whole_source_elc_from_d_prime"])
    contained_raw = [float(v) for k, v in raw_values.items() if k != "whole_source_elc_from_d_prime"]
    rows = [
        ("complete_source_null_excess_ge_0.01", float(complete["null_excess_bits"]), ">=0.01 bits", float(complete["null_excess_bits"]) >= 0.01),
        ("complete_source_surrogate_p_lt_0.05", float(complete["generation_preserving_surrogate_p_value"]), "<0.05", float(complete["generation_preserving_surrogate_p_value"]) < 0.05),
        ("epigenetic_source_distinguishable_from_null", float(epi["null_excess_bits"]), "null-excess >0 and p<0.05", float(epi["null_excess_bits"]) > 0.0 and float(epi["generation_preserving_surrogate_p_value"]) < 0.05),
        ("minimum_phenotype_probability_ge_0.05", float(np.min(all_probs)), ">=0.05", np.min(all_probs) >= 0.05),
        ("maximum_phenotype_probability_le_0.70", float(np.max(all_probs)), "<=0.70", np.max(all_probs) <= 0.70),
        ("no_missing_phenotype_classes_in_folds", float(fold_support["present"].mean()), "all folds contain all classes", bool(fold_support["present"].all())),
        ("epigenetic_probabilities_not_saturated_low", float(epi_ranges["minimum"].min()), ">0.01", float(epi_ranges["minimum"].min()) > 0.01),
        ("epigenetic_probabilities_not_saturated_high", float(epi_ranges["maximum"].max()), "<0.99", float(epi_ranges["maximum"].max()) < 0.99),
        ("all_state_arrays_finite", float(numerical["all_finite"].mean()), "all finite", bool(numerical["all_finite"].all())),
        ("microbiome_positive", float(numerical.loc[numerical["level"] == "microbiome", "minimum"].min()), ">0", bool(numerical.loc[numerical["level"] == "microbiome", "positive_microbiome_if_applicable"].all())),
        ("microbiome_not_divergent", float(numerical.loc[numerical["level"] == "microbiome", "maximum"].max()), "<50", float(numerical.loc[numerical["level"] == "microbiome", "maximum"].max()) < 50.0),
        ("one_to_one_derangement_no_self_pairs", float((~pair_summary["self_pair_present"]).mean()), "all seeds no self-pairs", not bool(pair_summary["self_pair_present"].any())),
        ("one_to_one_derangement_unique_sources", float((pair_summary["receiving_lineages"] == pair_summary["unique_source_lineages"]).mean()), "each source lineage used once", bool((pair_summary["receiving_lineages"] == pair_summary["unique_source_lineages"]).all())),
        ("complete_source_raw_ge_contained_subsystems", whole_raw - max(contained_raw), ">=0 bits", whole_raw + 1e-10 >= max(contained_raw)),
    ]
    return pd.DataFrame(
        [{"criterion": c, "observed_value": float(v), "target": target, "passes": bool(passes)} for c, v, target, passes in rows]
    )


def _write_npz_for_selected(seed_results: Sequence[Mapping[str, object]], data_dir: Path, selected_label: str) -> None:
    for data in seed_results:
        seed = int(data["seed"])
        save_seed_archive(
            data_dir / f"inter_elc_calibration_{selected_label}_temporal_architecture_seed_{seed}.npz",
            data,
            pair_key="inter_elc_pairs",
        )


def _manifest(output_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            payload = path.read_bytes()
            rows.append(
                {
                    "relative_path": str(path.relative_to(output_dir)),
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "inter_elc_calibration_manifest.csv", index=False)
    return df


def exact_inter_elc_equations_text() -> str:
    return r"""# Inter-ELC state-space arm

The additional source lineage \(d'\) enters the reconstruction of the next-generation
ELC of \(d\) through the same-level inter-entity term in the state-space model:
\[
\widetilde{\mathbf{x}}^{[l]}_{d',\tau}
=
\mathbf{x}^{[l]}_{d',t_{l,j_l^R}(\tau)}
-
\boldsymbol{\mu}^{[l]}_{\mathrm{source}},
\]
and
\[
\mathbf{x}^{[l]}_{d,0(\tau+1)}
=
F^{[l]}_0\!\left(\mathcal{X}^{\mathrm{ELC}}_{d,\tau}\right)
+
B^{[l]}_{d,d',0(\tau+1)}
\widetilde{\mathbf{x}}^{[l]}_{d',\tau}
+
\boldsymbol{\epsilon}^{[l]}_{d,\tau}.
\]
This is algebraically equivalent to the uncentered \(B\mathbf{x}\) term with
the constant \(B\boldsymbol{\mu}^{[l]}_{\mathrm{source}}\) absorbed into the
reconstruction intercept.  The source means are estimated from calibration
time series and then fixed before the untouched evaluation.

The \(B\) matrices are applied only when the first state of a new generation is set.  All
within-generation \(B\) matrices are zero in this simulation arm.  Cross-level
effects inside the focal ELC remain the existing \(C^{[l]}_d\) terms of the C8
model.

For \(l_{\mathrm{epigenetic}}\),
\[
\widetilde{\mathbf{z}}_{d',\tau}
=
\mathbf{z}_{d',t_{l_{\mathrm{epigenetic}},j_{\mathrm{epigenetic}}^R}(\tau)}
-
\boldsymbol{\mu}^{[l_{\mathrm{epigenetic}}]}_{\mathrm{source}},
\]
\[
\mathbf{z}_{d,0(\tau+1)}
=
\boldsymbol{\rho}_{\mathrm{epi}}\odot\mathbf{z}_{d,t_{l_{\mathrm{epigenetic}},j_{\mathrm{epigenetic}}^R}(\tau)}
+
\begin{pmatrix}b_{\mathrm{reg}}&0\\0&b_{\mathrm{stress}}\end{pmatrix}
\widetilde{\mathbf{z}}_{d',\tau}
+
\begin{pmatrix}
\gamma_{\mathrm{reg}} z_{d,\mathrm{reg}}z_{d',\mathrm{reg}}\\
\gamma_{\mathrm{stress}} z_{d,\mathrm{stress}}z_{d',\mathrm{stress}}
\end{pmatrix}
+
\boldsymbol{\epsilon}^{[\mathrm{epi}]}_{d,\tau}.
\]
The product term is a separate nonlinear two-entity contribution and is not part
of the fixed linear \(B^{[l_{\mathrm{epigenetic}}]}\) matrix.  In the second
calibration cycle, \((b_{\mathrm{reg}},b_{\mathrm{stress}})\) and
\((\gamma_{\mathrm{reg}},\gamma_{\mathrm{stress}})\) are scaled together.

For \(l_{\mathrm{microbiome}}\), the same-level contribution is on the
log-abundance scale:
\[
\widetilde{\mathbf{y}}_{d',\tau}
=
\log\mathbf{x}^{[l_{\mathrm{microbiome}}]}_{d',t_{l_{\mathrm{microbiome}},j_{\mathrm{microbiome}}^R}(\tau)}
-
\boldsymbol{\mu}^{[l_{\mathrm{microbiome}}]}_{\mathrm{source}},
\]
\[
\log\mathbf{x}^{[l_{\mathrm{microbiome}}]}_{d,0(\tau+1)}
=
F^{[l_{\mathrm{microbiome}}]}_0
+
b_{\mathrm{micro}}I_5
\widetilde{\mathbf{y}}_{d',\tau}
+
\boldsymbol{\epsilon}^{[\mathrm{micro}]}_{d,\tau}.
\]

For \(l_{\mathrm{ecological}}\),
\[
\widetilde{\mathbf{e}}_{d',\tau}
=
\mathbf{x}^{[l_{\mathrm{ecological}}]}_{d',t_{l_{\mathrm{ecological}},j_{\mathrm{ecological}}^R}(\tau)}
-
\boldsymbol{\mu}^{[l_{\mathrm{ecological}}]}_{\mathrm{source}},
\]
\[
\mathbf{x}^{[l_{\mathrm{ecological}}]}_{d,0(\tau+1)}
=
F^{[l_{\mathrm{ecological}}]}_0
+
b_{\mathrm{eco}}I_3
\widetilde{\mathbf{e}}_{d',\tau}
+
\boldsymbol{\epsilon}^{[\mathrm{eco}]}_{d,\tau}.
\]

For development and life history,
\[
B^{[l_{\mathrm{development}}]}_{d,d',0(\tau+1)}=\mathbf{0},
\qquad
B^{[l_{\mathrm{life-history}}]}_{d,d',0(\tau+1)}=\mathbf{0}.
\]
"""


def _evaluate_inter_elc_candidate(
    *,
    label: str,
    stage: str,
    epigenetic_scale: float,
    b: InterELCBMatrices,
    config: CorrectedAuditConfig,
    source_means: InterELCSourceMeans,
    data_dir: Path,
    candidate_index: int,
) -> dict[str, object]:
    print(f"inter-ELC second calibration {stage}: {label}", flush=True)
    seed_results = [simulate_inter_elc_seed(seed, config, b, source_means) for seed in config.seeds]
    table = build_inter_elc_table(seed_results, config)
    seed_tables = [build_inter_elc_table([sd], config) for sd in seed_results]
    family = inter_elc_family()
    summary, _, null, seed_df = estimate_inter_elc_family(
        table,
        seed_tables,
        config,
        family,
        output_dir=None,
        prefix=label,
        jitter=0.0,
        n_bootstrap=150,
        n_null=150,
        seed_offset=220_000 + candidate_index * 1000,
        compute_seed_level=False,
    )
    for df in (summary, null, seed_df):
        df.insert(0, "candidate", label)
        df.insert(1, "stage", stage)
        df.insert(2, "epigenetic_scale", epigenetic_scale)
        df.insert(3, "b_micro", b.b_micro)
        df.insert(4, "b_eco", b.b_eco)
        df.insert(5, "coupling_magnitude", inter_elc_coupling_magnitude(b))
    seed_df.to_csv(data_dir / f"{label}_source_information_by_seed.csv", index=False)
    null.to_csv(data_dir / f"{label}_source_information_generation_preserving_null.csv", index=False)

    phenotype = phenotype_support(table)
    phenotype.insert(0, "candidate", label)
    phenotype.insert(1, "stage", stage)
    fold_support = phenotype_fold_support(table)
    fold_support.insert(0, "candidate", label)
    fold_support.insert(1, "stage", stage)
    epi_range = epigenetic_probability_range(seed_results)
    epi_range.insert(0, "candidate", label)
    epi_range.insert(1, "stage", stage)
    numerical = inter_elc_numerical_checks(seed_results, config)
    numerical.insert(0, "candidate", label)
    numerical.insert(1, "stage", stage)
    pair_summary = table["inter_elc_pairs"].groupby("seed").agg(
        receiving_lineages=("lineage_id_d", "count"),
        unique_source_lineages=("source_lineage_d_prime", "nunique"),
        self_pair_present=("same_lineage", "any"),
    ).reset_index()
    pair_summary.insert(0, "candidate", label)
    pair_summary.insert(1, "stage", stage)

    checks = inter_elc_candidate_pass_table(summary, phenotype, fold_support, epi_range, numerical)
    checks.insert(0, "candidate", label)
    checks.insert(1, "stage", stage)
    checks.insert(2, "epigenetic_scale", epigenetic_scale)
    checks.insert(3, "b_micro", b.b_micro)
    checks.insert(4, "b_eco", b.b_eco)
    checks.insert(5, "coupling_magnitude", inter_elc_coupling_magnitude(b))

    return {
        "label": label,
        "stage": stage,
        "epigenetic_scale": epigenetic_scale,
        "B": b,
        "summary": summary,
        "null": null,
        "seed_df": seed_df,
        "phenotype": phenotype,
        "fold_support": fold_support,
        "epi_range": epi_range,
        "numerical": numerical,
        "pair_summary": pair_summary,
        "checks": checks,
        "seed_results": seed_results,
        "table": table,
        "passes": bool(checks["passes"].all()),
    }


def run_inter_elc_second_calibration_audit(output_dir: Path) -> dict[str, object]:
    t0 = perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    code_dir = output_dir / "analysis_code"
    code_dir.mkdir(exist_ok=True)

    config = CorrectedAuditConfig(
        seeds=INTER_ELC_CALIBRATION_SEEDS,
        parameter_overrides=calibration_candidates()["C8_stronger_growth_allocation"],
    )
    (data_dir / "inter_elc_second_calibration_configuration.json").write_text(json.dumps(asdict(config), indent=2))
    (data_dir / "discard_previous_paired_only_whole_source_results.md").write_text(
        "The previous whole-source ELC estimates paired d-prime with d only during information analysis. "
        "They are invalid for Equation 19 and must not be interpreted or included.\n"
    )
    (data_dir / "inter_elc_centered_equations.md").write_text(exact_inter_elc_equations_text())
    convergence_check(config.pilot_config(config.seeds[0])).to_csv(data_dir / "integration_convergence_confirmation.csv", index=False)

    baseline_results = [simulate_inter_elc_seed(seed, config, zero_b_matrices(), InterELCSourceMeans.zero()) for seed in config.seeds]
    source_means = estimate_source_state_means(baseline_results, config)
    source_state_means_table(source_means).to_csv(data_dir / "inter_elc_second_calibration_source_state_means.csv", index=False)

    evaluations: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    candidate_index = 0

    for label, scale, b in inter_elc_epigenetic_isolation_grid():
        candidate_index += 1
        candidate_rows.append(
            {
                "candidate": label,
                "stage": "stage1_epigenetic_isolation",
                "epigenetic_scale": scale,
                "b_micro": b.b_micro,
                "b_eco": b.b_eco,
                "b_epi_reg": b.b_epi_reg,
                "b_epi_stress": b.b_epi_stress,
                "epi_interaction_reg": b.epi_interaction_reg,
                "epi_interaction_stress": b.epi_interaction_stress,
                "coupling_magnitude": inter_elc_coupling_magnitude(b),
            }
        )
        evaluations.append(
            _evaluate_inter_elc_candidate(
                label=label,
                stage="stage1_epigenetic_isolation",
                epigenetic_scale=scale,
                b=b,
                config=config,
                source_means=source_means,
                data_dir=data_dir,
                candidate_index=candidate_index,
            )
        )

    passing_stage1 = [ev for ev in evaluations if ev["stage"] == "stage1_epigenetic_isolation" and ev["passes"]]
    selected_stage1 = max(passing_stage1, key=lambda ev: float(ev["epigenetic_scale"])) if passing_stage1 else None

    if selected_stage1 is not None:
        stage2_scale = float(selected_stage1["epigenetic_scale"])
        for label, scale, b in inter_elc_small_micro_eco_grid(stage2_scale):
            candidate_index += 1
            candidate_rows.append(
                {
                    "candidate": label,
                    "stage": "stage2_small_microbiome_ecology",
                    "epigenetic_scale": scale,
                    "b_micro": b.b_micro,
                    "b_eco": b.b_eco,
                    "b_epi_reg": b.b_epi_reg,
                    "b_epi_stress": b.b_epi_stress,
                    "epi_interaction_reg": b.epi_interaction_reg,
                    "epi_interaction_stress": b.epi_interaction_stress,
                    "coupling_magnitude": inter_elc_coupling_magnitude(b),
                }
            )
            evaluations.append(
                _evaluate_inter_elc_candidate(
                    label=label,
                    stage="stage2_small_microbiome_ecology",
                    epigenetic_scale=scale,
                    b=b,
                    config=config,
                    source_means=source_means,
                    data_dir=data_dir,
                    candidate_index=candidate_index,
                )
            )

    passing_stage2 = [ev for ev in evaluations if ev["stage"] == "stage2_small_microbiome_ecology" and ev["passes"]]
    selected = min(passing_stage2, key=lambda ev: inter_elc_coupling_magnitude(ev["B"])) if passing_stage2 else None

    pd.DataFrame(candidate_rows).to_csv(data_dir / "inter_elc_second_calibration_candidate_B_matrices.csv", index=False)
    pd.concat([ev["summary"] for ev in evaluations], ignore_index=True).to_csv(data_dir / "inter_elc_second_calibration_source_information_summary.csv", index=False)
    pd.concat([ev["checks"] for ev in evaluations], ignore_index=True).to_csv(data_dir / "inter_elc_second_calibration_candidate_checks.csv", index=False)
    pd.concat([ev["phenotype"] for ev in evaluations], ignore_index=True).to_csv(data_dir / "inter_elc_second_calibration_phenotype_support.csv", index=False)
    pd.concat([ev["fold_support"] for ev in evaluations], ignore_index=True).to_csv(data_dir / "inter_elc_second_calibration_phenotype_fold_support.csv", index=False)
    pd.concat([ev["epi_range"] for ev in evaluations], ignore_index=True).to_csv(data_dir / "inter_elc_second_calibration_epigenetic_probability_ranges.csv", index=False)
    pd.concat([ev["numerical"] for ev in evaluations], ignore_index=True).to_csv(data_dir / "inter_elc_second_calibration_numerical_checks.csv", index=False)
    pd.concat([ev["pair_summary"] for ev in evaluations], ignore_index=True).to_csv(data_dir / "inter_elc_second_calibration_pairing_summary.csv", index=False)

    inv_rows = []
    level_rows = []
    for row in candidate_rows:
        b = InterELCBMatrices(
            b_micro=float(row["b_micro"]),
            b_eco=float(row["b_eco"]),
            b_epi_reg=float(row["b_epi_reg"]),
            b_epi_stress=float(row["b_epi_stress"]),
            epi_interaction_reg=float(row["epi_interaction_reg"]),
            epi_interaction_stress=float(row["epi_interaction_stress"]),
        )
        inv = inter_elc_b_matrix_inventory(b)
        inv.insert(0, "candidate", row["candidate"])
        inv.insert(1, "stage", row["stage"])
        inv_rows.append(inv)
        lvl = inter_elc_level_summary(b)
        lvl.insert(0, "candidate", row["candidate"])
        lvl.insert(1, "stage", row["stage"])
        level_rows.append(lvl)
    pd.concat(inv_rows, ignore_index=True).to_csv(data_dir / "inter_elc_second_calibration_B_component_inventory_all_candidates.csv", index=False)
    pd.concat(level_rows, ignore_index=True).to_csv(data_dir / "inter_elc_second_calibration_B_level_summary_all_candidates.csv", index=False)

    if selected is not None:
        selected_b: InterELCBMatrices = selected["B"]
        _write_npz_for_selected(selected["seed_results"], data_dir, selected["label"])
        selected_inv = inter_elc_b_matrix_inventory(selected_b)
        selected_inv.to_csv(data_dir / "selected_inter_elc_second_B_component_inventory.csv", index=False)
        inter_elc_level_summary(selected_b).to_csv(data_dir / "selected_inter_elc_second_B_level_summary.csv", index=False)
        selected_status = "selected_not_final_run"
        selected_label = str(selected["label"])
    else:
        selected_b = None
        selected_status = "none_selected_no_final_run"
        selected_label = "none"

    pd.DataFrame(
        [
            {
                "selected_candidate": selected_label,
                "selected": bool(selected is not None),
                "selected_stage1_candidate": "none" if selected_stage1 is None else str(selected_stage1["label"]),
                "selected_stage1_epigenetic_scale": np.nan if selected_stage1 is None else float(selected_stage1["epigenetic_scale"]),
                "b_micro": np.nan if selected_b is None else selected_b.b_micro,
                "b_eco": np.nan if selected_b is None else selected_b.b_eco,
                "b_epi_reg": np.nan if selected_b is None else selected_b.b_epi_reg,
                "b_epi_stress": np.nan if selected_b is None else selected_b.b_epi_stress,
                "epi_interaction_reg": np.nan if selected_b is None else selected_b.epi_interaction_reg,
                "epi_interaction_stress": np.nan if selected_b is None else selected_b.epi_interaction_stress,
                "proposed_untouched_final_seeds": ";".join(map(str, PROPOSED_UNTOUCHED_INTER_ELC_FINAL_SEEDS)),
                "final_seed_status": "proposed_not_run",
                "selection_status": selected_status,
            }
        ]
    ).to_csv(data_dir / "selected_inter_elc_second_setting_and_proposed_final_seeds.csv", index=False)
    pd.DataFrame(
        [{"seed": seed, "used_in_prior_c8_work": False, "status": "proposed_untouched_inter_elc_final_seed"} for seed in PROPOSED_UNTOUCHED_INTER_ELC_FINAL_SEEDS]
    ).to_csv(data_dir / "proposed_untouched_inter_elc_final_seeds.csv", index=False)
    pd.DataFrame(
        [
            {
                "check": "zero_B_initializer_matches_primary_initializer",
                "passes": bool(zero_b_initializer_matches_primary(config.pilot_config(config.seeds[0]), _params_for_config(config))),
                "interpretation": "setting all B terms and nonlinear d-prime terms to zero removes the inter-ELC contribution at initialization",
            },
            {
                "check": "source_deviation_means_are_frozen_from_calibration_trajectories",
                "passes": True,
                "interpretation": "source-state means were estimated from zero-B calibration trajectories and saved before evaluating nonzero B candidates",
            },
        ]
    ).to_csv(data_dir / "inter_elc_second_B_implementation_checks.csv", index=False)
    runtime = {
        "runtime_seconds": perf_counter() - t0,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "simulation_arm": "inter_ELC_second_calibration_only",
        "primary_C8_status": "unchanged",
        "source_deviation_centering": "enabled",
    }
    (data_dir / "inter_elc_second_calibration_runtime.json").write_text(json.dumps(runtime, indent=2))
    for source in [
        Path("src/inter_elc_arm.py"),
        Path("src/corrected_numerical_audit.py"),
        Path("src/final_numerical_audit.py"),
        Path("src/corrected_pilot_model.py"),
        Path("src/recalibration_audit.py"),
        Path("run_inter_elc_calibration_audit.py"),
        Path("run_inter_elc_second_calibration_audit.py"),
        Path("pytest.ini"),
    ]:
        if source.exists():
            shutil.copy2(source, code_dir / source.name)
    manifest = _manifest(output_dir)
    return {
        "output_dir": output_dir,
        "data_dir": data_dir,
        "selected": None if selected is None else selected["label"],
        "selected_stage1": None if selected_stage1 is None else selected_stage1["label"],
        "evaluations": evaluations,
        "manifest": manifest,
    }


def run_inter_elc_final_audit(
    output_dir: Path,
    *,
    seeds: Sequence[int] | None = None,
    b: InterELCBMatrices | None = None,
    source_means: InterELCSourceMeans | None = None,
    prior_seed_sets: Sequence[Sequence[int]] | None = None,
    parameter_overrides: Sequence[tuple[str, float]] | None = None,
    calibration_seeds: Sequence[int] | None = None,
) -> dict[str, object]:
    t0 = perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    code_dir = output_dir / "analysis_code"
    code_dir.mkdir(exist_ok=True)

    final_seeds = tuple(int(seed) for seed in (INTER_ELC_FINAL_SEEDS if seeds is None else seeds))
    selected_overrides = (
        calibration_candidates()["C8_stronger_growth_allocation"]
        if parameter_overrides is None
        else tuple(parameter_overrides)
    )
    selected_calibration_seeds = tuple(
        int(seed) for seed in (INTER_ELC_CALIBRATION_SEEDS if calibration_seeds is None else calibration_seeds)
    )
    config = CorrectedAuditConfig(
        seeds=final_seeds,
        parameter_overrides=selected_overrides,
    )
    b = frozen_epigenetic_only_inter_elc_B() if b is None else b
    calibration_config = CorrectedAuditConfig(
        seeds=selected_calibration_seeds,
        parameter_overrides=selected_overrides,
    )
    source_means = estimate_source_state_means_sequential(calibration_config) if source_means is None else source_means
    prior_seeds = set(selected_calibration_seeds)
    if prior_seed_sets is not None:
        for seed_set in prior_seed_sets:
            prior_seeds.update(int(seed) for seed in seed_set)
    if set(final_seeds) & prior_seeds:
        raise AssertionError("final inter-ELC seeds overlap with inter-ELC calibration seeds")

    (data_dir / "inter_elc_final_configuration.json").write_text(json.dumps(asdict(config), indent=2))
    (data_dir / "inter_elc_final_centered_equations.md").write_text(exact_inter_elc_equations_text())
    source_state_means_table(source_means).to_csv(data_dir / "inter_elc_final_frozen_source_state_means.csv", index=False)
    inter_elc_b_matrix_inventory(b).to_csv(data_dir / "inter_elc_final_B_component_inventory.csv", index=False)
    inter_elc_level_summary(b).to_csv(data_dir / "inter_elc_final_B_level_summary.csv", index=False)
    convergence_check(config.pilot_config(config.seeds[0])).to_csv(data_dir / "integration_convergence_confirmation.csv", index=False)
    pd.DataFrame(
        [{"seed": int(seed), "used_in_inter_elc_calibration": bool(seed in prior_seeds), "status": "untouched_inter_elc_final_seed"} for seed in final_seeds]
    ).to_csv(data_dir / "inter_elc_final_seed_verification.csv", index=False)

    seed_results = [simulate_inter_elc_seed(seed, config, b, source_means) for seed in config.seeds]
    table = build_inter_elc_table(seed_results, config)
    seed_tables = [build_inter_elc_table([sd], config) for sd in seed_results]

    family = inter_elc_family()
    summary, boot, null, seed_df = estimate_inter_elc_family(
        table,
        seed_tables,
        config,
        family,
        output_dir=data_dir,
        prefix="inter_elc_final_whole_and_subsystem",
        jitter=0.0,
        n_bootstrap=config.n_bootstrap,
        n_null=config.n_null,
        seed_offset=330_000,
        compute_seed_level=True,
    )

    phenotype = phenotype_support(table)
    phenotype.to_csv(data_dir / "inter_elc_final_phenotype_support.csv", index=False)
    fold_support = phenotype_fold_support(table)
    fold_support.to_csv(data_dir / "inter_elc_final_phenotype_fold_support.csv", index=False)
    epi_ranges = epigenetic_probability_range(seed_results)
    epi_ranges.to_csv(data_dir / "inter_elc_final_epigenetic_probability_ranges.csv", index=False)
    numerical = inter_elc_numerical_checks(seed_results, config)
    numerical.to_csv(data_dir / "inter_elc_final_numerical_checks.csv", index=False)
    pair_summary = table["inter_elc_pairs"].groupby("seed").agg(
        receiving_lineages=("lineage_id_d", "count"),
        unique_source_lineages=("source_lineage_d_prime", "nunique"),
        self_pair_present=("same_lineage", "any"),
    ).reset_index()
    pair_summary.to_csv(data_dir / "inter_elc_final_pairing_summary.csv", index=False)

    acceptance = inter_elc_final_acceptance_table(summary, phenotype, fold_support, epi_ranges, numerical, pair_summary)
    acceptance.to_csv(data_dir / "inter_elc_final_acceptance_checks.csv", index=False)

    profile_parts = []
    profile_boots = []
    profile_nulls = []
    profile_seed = []
    for j, level in enumerate(ELC_LEVELS):
        level_family = inter_elc_family(
            target_key=f"target_{level}_future",
            family_name=f"inter_elc_final_target_profile_{level}",
        )
        level_summary, level_boot, level_null, level_seed = estimate_inter_elc_family(
            table,
            seed_tables,
            config,
            level_family,
            output_dir=None,
            prefix=f"inter_elc_final_target_profile_{level}",
            jitter=0.0,
            n_bootstrap=config.n_bootstrap,
            n_null=config.n_null,
            seed_offset=340_000 + j * 10_000,
            compute_seed_level=True,
        )
        for df in (level_summary, level_boot, level_null, level_seed):
            df.insert(0, "target_level", level)
        profile_parts.append(level_summary)
        profile_boots.append(level_boot)
        profile_nulls.append(level_null)
        profile_seed.append(level_seed)
    pd.concat(profile_parts, ignore_index=True).to_csv(data_dir / "inter_elc_final_target_level_profile_summary.csv", index=False)
    pd.concat(profile_boots, ignore_index=True).to_csv(data_dir / "inter_elc_final_target_level_profile_bootstrap.csv", index=False)
    pd.concat(profile_nulls, ignore_index=True).to_csv(data_dir / "inter_elc_final_target_level_profile_generation_preserving_null.csv", index=False)
    pd.concat(profile_seed, ignore_index=True).to_csv(data_dir / "inter_elc_final_target_level_profile_by_seed.csv", index=False)

    pd.DataFrame(
        [
            {
                "check": "zero_B_initializer_matches_primary_initializer",
                "passes": bool(zero_b_initializer_matches_primary(config.pilot_config(config.seeds[0]), _params_for_config(config))),
                "interpretation": "setting all B terms and nonlinear d-prime terms to zero removes the inter-ELC contribution at initialization",
            },
            {
                "check": "final_B_epigenetic_only",
                "passes": bool(
                    b.b_micro == 0.0
                    and b.b_eco == 0.0
                    and b.b_epi_reg > 0.0
                    and b.b_epi_stress > 0.0
                    and b.epi_interaction_reg > 0.0
                    and b.epi_interaction_stress > 0.0
                ),
                "interpretation": "final inter-ELC arm uses only the sparse same-level epigenetic B matrix selected before the untouched evaluation",
            },
            {
                "check": "source_deviation_means_are_frozen",
                "passes": True,
                "interpretation": "source-state means are fixed to the approved calibration values before evaluating untouched final seeds",
            },
        ]
    ).to_csv(data_dir / "inter_elc_final_implementation_checks.csv", index=False)

    for data in seed_results:
        save_seed_archive(
            data_dir / f"inter_elc_final_temporal_architecture_seed_{int(data['seed'])}.npz",
            data,
            pair_key="inter_elc_pairs",
        )

    runtime = {
        "runtime_seconds": perf_counter() - t0,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "simulation_arm": "inter_ELC_final_epigenetic_only",
        "primary_C8_status": "unchanged",
        "source_deviation_centering": "enabled",
        "final_acceptance_passes": bool(acceptance["passes"].all()),
    }
    (data_dir / "inter_elc_final_runtime.json").write_text(json.dumps(runtime, indent=2))

    for source in [
        Path("src/inter_elc_arm.py"),
        Path("src/corrected_numerical_audit.py"),
        Path("src/final_numerical_audit.py"),
        Path("src/corrected_pilot_model.py"),
        Path("src/recalibration_audit.py"),
        Path("run_inter_elc_final_audit.py"),
        Path("pytest.ini"),
    ]:
        if source.exists():
            shutil.copy2(source, code_dir / source.name)
    manifest = _manifest(output_dir)
    return {
        "output_dir": output_dir,
        "data_dir": data_dir,
        "selected_B": b,
        "summary": summary,
        "acceptance": acceptance,
        "manifest": manifest,
        "passes": bool(acceptance["passes"].all()),
    }


def run_inter_elc_calibration_audit(output_dir: Path) -> dict[str, object]:
    t0 = perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    code_dir = output_dir / "analysis_code"
    code_dir.mkdir(exist_ok=True)

    config = CorrectedAuditConfig(
        seeds=INTER_ELC_CALIBRATION_SEEDS,
        parameter_overrides=calibration_candidates()["C8_stronger_growth_allocation"],
    )
    (data_dir / "inter_elc_calibration_configuration.json").write_text(json.dumps(asdict(config), indent=2))
    (data_dir / "discard_previous_paired_only_whole_source_results.md").write_text(
        "The previous whole-source ELC estimates paired d-prime with d only during information analysis. "
        "They are invalid for Equation 19 and must not be interpreted or included.\n"
    )
    (data_dir / "inter_elc_equations.md").write_text(exact_inter_elc_equations_text())
    convergence_check(config.pilot_config(config.seeds[0])).to_csv(data_dir / "integration_convergence_confirmation.csv", index=False)

    calibration_rows: list[pd.DataFrame] = []
    pass_rows: list[pd.DataFrame] = []
    phenotype_rows: list[pd.DataFrame] = []
    epi_range_rows: list[pd.DataFrame] = []
    numerical_rows: list[pd.DataFrame] = []
    pair_rows: list[pd.DataFrame] = []
    selected: tuple[str, InterELCBMatrices, list[Mapping[str, object]], pd.DataFrame] | None = None
    candidate_settings = []

    for idx, b in enumerate(inter_elc_calibration_grid(), start=1):
        label = f"B_micro_{b.b_micro:.2f}_B_eco_{b.b_eco:.2f}".replace(".", "p")
        candidate_settings.append(
            {
                "candidate": label,
                "b_micro": b.b_micro,
                "b_eco": b.b_eco,
                "b_epi_reg": b.b_epi_reg,
                "b_epi_stress": b.b_epi_stress,
                "epi_interaction_reg": b.epi_interaction_reg,
                "epi_interaction_stress": b.epi_interaction_stress,
            }
        )
        print(f"inter-ELC calibration candidate {idx}/9: {label}", flush=True)
        seed_results = [simulate_inter_elc_seed(seed, config, b) for seed in config.seeds]
        table = build_inter_elc_table(seed_results, config)
        seed_tables = [build_inter_elc_table([sd], config) for sd in seed_results]
        family = inter_elc_family()
        summary, _, null, seed_df = estimate_inter_elc_family(
            table,
            seed_tables,
            config,
            family,
            output_dir=None,
            prefix=label,
            jitter=0.0,
            n_bootstrap=150,
            n_null=150,
            seed_offset=idx * 1000,
            compute_seed_level=False,
        )
        summary.insert(0, "candidate", label)
        seed_df.insert(0, "candidate", label)
        seed_df.to_csv(data_dir / f"{label}_source_information_by_seed.csv", index=False)
        null.to_csv(data_dir / f"{label}_source_information_generation_preserving_null.csv", index=False)
        calibration_rows.append(summary)

        phenotype = phenotype_support(table)
        phenotype.insert(0, "candidate", label)
        phenotype_rows.append(phenotype)
        fold_support = phenotype_fold_support(table)
        fold_support.insert(0, "candidate", label)
        epi_range = epigenetic_probability_range(seed_results)
        epi_range.insert(0, "candidate", label)
        epi_range_rows.append(epi_range)
        numerical = inter_elc_numerical_checks(seed_results, config)
        numerical.insert(0, "candidate", label)
        numerical_rows.append(numerical)
        pair_summary = table["inter_elc_pairs"].groupby("seed").agg(
            receiving_lineages=("lineage_id_d", "count"),
            unique_source_lineages=("source_lineage_d_prime", "nunique"),
            self_pair_present=("same_lineage", "any"),
        ).reset_index()
        pair_summary.insert(0, "candidate", label)
        pair_rows.append(pair_summary)
        checks = inter_elc_candidate_pass_table(summary, phenotype, fold_support, epi_range, numerical)
        checks.insert(0, "candidate", label)
        pass_rows.append(checks)
        if selected is None and bool(checks["passes"].all()):
            selected = (label, b, seed_results, table)

    candidate_df = pd.DataFrame(candidate_settings)
    candidate_df.to_csv(data_dir / "inter_elc_calibration_candidate_B_matrices.csv", index=False)
    calibration_summary = pd.concat(calibration_rows, ignore_index=True)
    calibration_summary.to_csv(data_dir / "inter_elc_calibration_source_information_summary.csv", index=False)
    checks_all = pd.concat(pass_rows, ignore_index=True)
    checks_all.to_csv(data_dir / "inter_elc_calibration_candidate_checks.csv", index=False)
    pd.concat(phenotype_rows, ignore_index=True).to_csv(data_dir / "inter_elc_calibration_phenotype_support.csv", index=False)
    pd.concat(epi_range_rows, ignore_index=True).to_csv(data_dir / "inter_elc_calibration_epigenetic_probability_ranges.csv", index=False)
    pd.concat(numerical_rows, ignore_index=True).to_csv(data_dir / "inter_elc_calibration_numerical_checks.csv", index=False)
    pd.concat(pair_rows, ignore_index=True).to_csv(data_dir / "inter_elc_calibration_pairing_summary.csv", index=False)

    if selected is None:
        selected_label = ""
        selected_b = None
        selected_results = []
        selected_table = None
    else:
        selected_label, selected_b, selected_results, selected_table = selected
        _write_npz_for_selected(selected_results, data_dir, selected_label)
        inter_elc_b_matrix_inventory(selected_b).to_csv(data_dir / "selected_inter_elc_B_component_inventory.csv", index=False)
        inter_elc_level_summary(selected_b).to_csv(data_dir / "selected_inter_elc_B_level_summary.csv", index=False)
        selected_family = inter_elc_family()
        selected_seed_tables = [build_inter_elc_table([sd], config) for sd in selected_results]
        estimate_inter_elc_family(
            selected_table,
            selected_seed_tables,
            config,
            selected_family,
            output_dir=data_dir,
            prefix="selected_inter_elc_whole_and_subsystem",
            jitter=0.0,
            n_bootstrap=config.n_bootstrap,
            n_null=config.n_null,
            seed_offset=77_000,
            compute_seed_level=True,
        )
        profile_parts = []
        profile_boots = []
        profile_nulls = []
        profile_seed = []
        for j, level in enumerate(ELC_LEVELS):
            family = inter_elc_family(
                target_key=f"target_{level}_future",
                family_name=f"inter_elc_target_profile_{level}",
            )
            summary, boot, null, seed_df = estimate_inter_elc_family(
                selected_table,
                selected_seed_tables,
                config,
                family,
                output_dir=None,
                prefix=f"selected_target_profile_{level}",
                jitter=0.0,
                n_bootstrap=200,
                n_null=200,
                seed_offset=88_000 + j * 1000,
                compute_seed_level=True,
            )
            summary.insert(0, "target_level", level)
            boot.insert(0, "target_level", level)
            null.insert(0, "target_level", level)
            seed_df.insert(0, "target_level", level)
            profile_parts.append(summary)
            profile_boots.append(boot)
            profile_nulls.append(null)
            profile_seed.append(seed_df)
        pd.concat(profile_parts, ignore_index=True).to_csv(data_dir / "selected_inter_elc_target_level_profile_summary.csv", index=False)
        pd.concat(profile_boots, ignore_index=True).to_csv(data_dir / "selected_inter_elc_target_level_profile_bootstrap.csv", index=False)
        pd.concat(profile_nulls, ignore_index=True).to_csv(data_dir / "selected_inter_elc_target_level_profile_generation_preserving_null.csv", index=False)
        pd.concat(profile_seed, ignore_index=True).to_csv(data_dir / "selected_inter_elc_target_level_profile_by_seed.csv", index=False)

    pd.DataFrame(
        [
            {
                "selected_candidate": selected_label if selected_label else "none",
                "selected": bool(selected is not None),
                "b_micro": np.nan if selected_b is None else selected_b.b_micro,
                "b_eco": np.nan if selected_b is None else selected_b.b_eco,
                "b_epi_reg": np.nan if selected_b is None else selected_b.b_epi_reg,
                "b_epi_stress": np.nan if selected_b is None else selected_b.b_epi_stress,
                "proposed_untouched_final_seeds": ";".join(map(str, PROPOSED_UNTOUCHED_INTER_ELC_FINAL_SEEDS)),
                "final_seed_status": "proposed_not_run",
            }
        ]
    ).to_csv(data_dir / "selected_inter_elc_setting_and_proposed_final_seeds.csv", index=False)
    pd.DataFrame(
        [{"seed": seed, "used_in_prior_c8_work": False, "status": "proposed_untouched_inter_elc_final_seed"} for seed in PROPOSED_UNTOUCHED_INTER_ELC_FINAL_SEEDS]
    ).to_csv(data_dir / "proposed_untouched_inter_elc_final_seeds.csv", index=False)
    pd.DataFrame(
        [
            {
                "check": "zero_B_initializer_matches_primary_initializer",
                "passes": bool(zero_b_initializer_matches_primary(config.pilot_config(config.seeds[0]), _params_for_config(config))),
                "interpretation": "setting all B terms and nonlinear d-prime terms to zero removes the inter-ELC contribution at initialization",
            }
        ]
    ).to_csv(data_dir / "inter_elc_B_zero_implementation_check.csv", index=False)
    runtime = {
        "runtime_seconds": perf_counter() - t0,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "simulation_arm": "inter_ELC_calibration_only",
        "primary_C8_status": "unchanged",
    }
    (data_dir / "inter_elc_calibration_runtime.json").write_text(json.dumps(runtime, indent=2))
    for source in [
        Path("src/inter_elc_arm.py"),
        Path("src/corrected_numerical_audit.py"),
        Path("src/final_numerical_audit.py"),
        Path("src/corrected_pilot_model.py"),
        Path("src/recalibration_audit.py"),
        Path("run_inter_elc_calibration_audit.py"),
        Path("pytest.ini"),
    ]:
        if source.exists():
            target = code_dir / source.name
            shutil.copy2(source, target)
    manifest = _manifest(output_dir)
    return {
        "output_dir": output_dir,
        "data_dir": data_dir,
        "selected": selected[0] if selected else None,
        "summary": calibration_summary,
        "checks": checks_all,
        "manifest": manifest,
    }
