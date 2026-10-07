from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping
import warnings

import matplotlib

matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COMPONENTS: dict[str, list[str]] = {
    "development": [
        "gene_1_regulatory_initiation",
        "gene_2_somatic_growth",
        "gene_3_stress_response",
        "gene_4_maturation_timing",
        "gene_5_allocation_signal",
    ],
    "microbiome": [
        "guild_1_growth_support",
        "guild_2_fiber_fermenter",
        "guild_3_stress_tolerant",
        "guild_4_opportunist",
        "guild_5_cross_feeder",
    ],
    "life_history": [
        "maturation_progress",
        "growth_capacity",
        "reproductive_allocation",
    ],
    "epigenetic": [
        "z_regulatory_mark",
        "z_stress_memory_mark",
    ],
    "epigenetic_probability": [
        "Pr_regulatory_mark",
        "Pr_stress_memory_mark",
    ],
    "ecological": [
        "soil_organic_matter",
        "food_resource_enrichment",
        "microclimate_buffering",
    ],
    "background": [
        "rainfall_anomaly",
        "temperature_anomaly",
    ],
}


RETAINED_KEYS: dict[str, str] = {
    "development": "development",
    "microbiome": "microbiome",
    "life_history": "life_history",
    "epigenetic": "epigenetic",
    "ecological": "ecological",
    "background": "background",
}


@dataclass(frozen=True)
class PilotConfig:
    seed: int = 20260723
    n_lineages: int = 100
    n_generations: int = 12
    burn_in_generations: int = 4
    illustrative_lineage_d: int = 1
    variability_lineages_d: tuple[int, ...] = (2, 3, 4, 5)
    plot_generation_start: int = 8
    plot_generation_stop: int = 12
    plot_within_generation_tau: int = 10
    dev_dim: int = 5
    micro_dim: int = 5
    life_dim: int = 3
    epi_dim: int = 2
    eco_dim: int = 3
    bg_dim: int = 2
    m_dev: int = 8
    m_micro: int = 10
    m_life: int = 5
    m_epi: int = 5
    m_eco: int = 5
    m_bg: int = 5
    T_dev: int = 57
    T_micro: int = 61
    T_life: int = 41
    T_epi: int = 33
    T_eco: int = 33
    T_bg: int = 33
    dev_steps: int = 280
    micro_steps: int = 360
    life_steps: int = 200
    epi_steps: int = 160
    eco_steps: int = 160
    bg_steps: int = 160
    noise_scale: float = 1.0
    phenotype_maturation_t_l: int = 2
    phenotype_development_t_l: int = 2
    theta_maturation: float = 0.60
    theta_growth: float = 0.65
    t_early_maturation: int = 3
    early_maturation_u: float = 0.60
    reproductive_u: float = 0.75


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: np.ndarray) -> np.ndarray:
    if np.any((p <= 0.0) | (p >= 1.0)):
        raise FloatingPointError("logit input must lie strictly inside (0, 1)")
    return np.log(p / (1.0 - p))


def hill(p: np.ndarray | float, k: float = 0.50, n: float = 3.0) -> np.ndarray | float:
    value = np.asarray(p, dtype=float)
    return value**n / (k**n + value**n)


def level_m(config: PilotConfig) -> dict[str, int]:
    return {
        "development": config.m_dev,
        "microbiome": config.m_micro,
        "life_history": config.m_life,
        "epigenetic": config.m_epi,
        "ecological": config.m_eco,
        "background": config.m_bg,
    }


def level_T(config: PilotConfig) -> dict[str, int]:
    return {
        "development": config.T_dev,
        "microbiome": config.T_micro,
        "life_history": config.T_life,
        "epigenetic": config.T_epi,
        "ecological": config.T_eco,
        "background": config.T_bg,
    }


def level_segment_indices(config: PilotConfig) -> dict[str, np.ndarray]:
    indices = {
        "development": np.arange(6, 14, dtype=int),
        "microbiome": np.arange(36, 46, dtype=int),
        "life_history": np.arange(20, 25, dtype=int),
        "epigenetic": np.arange(20, 25, dtype=int),
        "ecological": np.arange(20, 25, dtype=int),
        "background": np.arange(20, 25, dtype=int),
    }
    for level, selected in indices.items():
        if selected.size != level_m(config)[level]:
            raise AssertionError(f"{level} segment length does not match m_l")
        if selected[0] < 0 or selected[-1] >= level_T(config)[level]:
            raise AssertionError(f"{level} segment indices fall outside the full retained grid")
        if selected.size >= level_T(config)[level]:
            raise AssertionError(f"{level} requires m_l < T_l")
        timestamps = level_timestamps(config)[level][selected]
        if np.any(timestamps > config.reproductive_u + 1e-12):
            raise AssertionError(f"{level} source segment extends beyond u_R")
    return indices


def level_timestamps(config: PilotConfig) -> dict[str, np.ndarray]:
    return {
        level: np.linspace(0.0, 1.0, T_l, dtype=float)
        for level, T_l in level_T(config).items()
    }


def reproductive_indices(config: PilotConfig) -> dict[str, int]:
    out: dict[str, int] = {}
    for level, timestamps in level_timestamps(config).items():
        eligible = np.flatnonzero(timestamps <= config.reproductive_u + 1e-12)
        if eligible.size == 0:
            raise AssertionError(f"{level} has no retained state at or before u_R")
        out[level] = int(eligible[-1])
    return out


def level_steps(config: PilotConfig) -> dict[str, int]:
    return {
        "development": config.dev_steps,
        "microbiome": config.micro_steps,
        "life_history": config.life_steps,
        "epigenetic": config.epi_steps,
        "ecological": config.eco_steps,
        "background": config.bg_steps,
    }


def full_retained_step_indices(n_steps: int, T_l: int) -> np.ndarray:
    if n_steps % (T_l - 1) != 0:
        raise AssertionError("full retained timestamps must align with numerical update steps")
    return np.linspace(0, n_steps, T_l, dtype=int)


def retained_step_indices(n_steps: int, T_l: int) -> np.ndarray:
    """Return the indices of all stored states for a level."""

    return full_retained_step_indices(n_steps, T_l)


def extract_analysis_segment(full_series: np.ndarray, level: str, config: PilotConfig) -> np.ndarray:
    return np.take(full_series, level_segment_indices(config)[level], axis=-2)


def reproductive_state(full_series: np.ndarray, level: str, config: PilotConfig) -> np.ndarray:
    return np.take(full_series, reproductive_indices(config)[level], axis=-2)


def build_event_schedule(config: PilotConfig) -> list[tuple[float, tuple[str, ...]]]:
    events: dict[float, list[str]] = {}
    for level, n_steps in level_steps(config).items():
        for step in range(1, n_steps + 1):
            u = round(step / n_steps, 12)
            events.setdefault(u, []).append(level)
    return [(u, tuple(levels)) for u, levels in sorted(events.items())]


def nonzero_cross_level_edges() -> pd.DataFrame:
    rows = [
        # Developmental inputs.
        ("microbiome", "guild_1_growth_support", "development", "gene_2_somatic_growth", 0.18, "+", "most recent within-generation state", "0.18 log(1+y_1)", "growth-supporting guild raises somatic-growth gene activity", "within_generation"),
        ("microbiome", "guild_4_opportunist", "development", "gene_3_stress_response", 0.16, "+", "most recent within-generation state", "0.16 log(1+y_4)", "opportunist expansion increases developmental stress response", "within_generation"),
        ("epigenetic", "regulatory_mark", "development", "gene_1_regulatory_initiation", 0.34, "+", "most recent within-generation state", "0.34 p_reg^circ", "regulatory mark promotes developmental regulatory initiation", "within_generation"),
        ("epigenetic", "stress_memory_mark", "development", "gene_3_stress_response", 0.28, "+", "most recent within-generation state", "0.28 p_stress^circ", "stress-memory mark promotes stress-response gene activity", "within_generation"),
        ("ecological", "food_resource_enrichment", "development", "gene_2_somatic_growth", 0.22, "+", "most recent within-generation state", "0.22 e_2", "food enrichment supports growth-regulatory expression", "within_generation"),
        ("ecological", "microclimate_buffering", "development", "gene_5_allocation_signal", 0.18, "+", "most recent within-generation state", "0.18 e_3", "buffered microclimate shifts allocation-related development", "within_generation"),
        ("background", "rainfall_anomaly", "development", "gene_2_somatic_growth", 0.10, "+", "most recent within-generation state", "0.10 b_1", "rainfall anomaly affects growth conditions", "within_generation"),
        ("background", "temperature_anomaly", "development", "gene_3_stress_response", 0.12, "+", "most recent within-generation state", "0.12 b_2", "temperature anomaly affects developmental stress", "within_generation"),
        ("life_history", "maturation_progress", "development", "gene_4_maturation_timing", 0.15, "+", "most recent within-generation state", "0.15 q_mat", "maturation progress feeds developmental timing state", "within_generation"),
        ("life_history", "reproductive_allocation", "development", "gene_5_allocation_signal", 0.12, "+", "most recent within-generation state", "0.12 q_repr", "reproductive allocation feeds allocation-related developmental signal", "within_generation"),
        # Microbiome growth inputs.
        ("development", "gene_2_somatic_growth", "microbiome", "guild_1_growth_support", 0.18, "+", "most recent within-generation state", "0.18 xbar_dev,2", "host growth state supports beneficial guild", "within_generation"),
        ("development", "gene_3_stress_response", "microbiome", "guild_3_stress_tolerant", 0.08, "+", "most recent within-generation state", "0.08 xbar_dev,3", "stress-response state supports stress-tolerant guild", "within_generation"),
        ("development", "gene_3_stress_response", "microbiome", "guild_4_opportunist", 0.16, "+", "most recent within-generation state", "0.16 xbar_dev,3", "stress physiology favors opportunist guild", "within_generation"),
        ("epigenetic", "regulatory_mark", "microbiome", "guild_1_growth_support", 0.10, "+", "most recent within-generation state", "0.10 p_reg^circ", "regulatory mark shifts host conditions toward support guild", "within_generation"),
        ("epigenetic", "stress_memory_mark", "microbiome", "guild_4_opportunist", 0.12, "+", "most recent within-generation state", "0.12 p_stress^circ", "stress-memory state shifts host conditions toward opportunists", "within_generation"),
        ("ecological", "soil_organic_matter", "microbiome", "guild_2_fiber_fermenter", 0.18, "+", "most recent within-generation state", "0.18 e_1", "organic matter supports fermenting guild", "within_generation"),
        ("ecological", "soil_organic_matter", "microbiome", "guild_5_cross_feeder", 0.07, "+", "most recent within-generation state", "0.07 e_1", "organic matter supports cross-feeder guild", "within_generation"),
        ("ecological", "food_resource_enrichment", "microbiome", "guild_1_growth_support", 0.20, "+", "most recent within-generation state", "0.20 e_2", "resource enrichment increases support guild growth", "within_generation"),
        ("ecological", "microclimate_buffering", "microbiome", "guild_3_stress_tolerant", -0.16, "-", "most recent within-generation state", "-0.16 e_3", "buffering lowers stress-tolerant guild advantage", "within_generation"),
        ("background", "rainfall_anomaly", "microbiome", "guild_1_growth_support", 0.08, "+", "most recent within-generation state", "0.08 b_1", "rainfall anomaly affects microbial resource flow", "within_generation"),
        ("background", "temperature_anomaly", "microbiome", "guild_4_opportunist", 0.10, "+", "most recent within-generation state", "0.10 b_2", "temperature anomaly favors opportunist growth", "within_generation"),
        # Life-history inputs.
        ("development", "gene_4_maturation_timing", "life_history", "maturation_progress", np.nan, "+", "most recent within-generation state", "2.22 sigma(8(u-eta))+0.42 xbar_dev,4; eta contains -0.10 xbar_dev,4", "maturation-timing gene accelerates threshold-crossing dynamics", "within_generation"),
        ("development", "gene_2_somatic_growth", "life_history", "growth_capacity", 0.66, "+", "most recent within-generation state", "0.66 xbar_dev,2", "somatic-growth gene raises growth capacity", "within_generation"),
        ("development", "gene_5_allocation_signal", "life_history", "reproductive_allocation", 0.24, "+", "most recent within-generation state", "0.24 xbar_dev,5", "allocation-related developmental signal raises reproductive allocation", "within_generation"),
        ("microbiome", "guild_1_growth_support", "life_history", "maturation_progress", np.nan, "+", "most recent within-generation state", "2.22 sigma(8(u-eta))+0.18 log(1+y_1); eta contains -0.08 log(1+y_1)", "support guild accelerates maturation progress", "within_generation"),
        ("microbiome", "guild_1_growth_support", "life_history", "growth_capacity", np.nan, "+", "most recent within-generation state", "0.36 log(1+y_1+0.6y_2)", "support guild contributes to growth capacity with guild 2", "within_generation"),
        ("microbiome", "guild_2_fiber_fermenter", "life_history", "growth_capacity", np.nan, "+", "most recent within-generation state", "0.36 log(1+y_1+0.6y_2)", "fiber fermenter contributes to growth capacity with guild 1", "within_generation"),
        ("epigenetic", "regulatory_mark", "life_history", "maturation_progress", np.nan, "+", "most recent within-generation state", "2.22 sigma(8(u-eta))+0.25 p_reg^circ; eta contains -0.07 p_reg^circ", "regulatory mark accelerates maturation progress", "within_generation"),
        ("epigenetic", "regulatory_mark", "life_history", "growth_capacity", 0.24, "+", "most recent within-generation state", "0.24 p_reg^circ", "regulatory mark raises growth capacity", "within_generation"),
        ("epigenetic", "stress_memory_mark", "life_history", "reproductive_allocation", 0.22, "+", "most recent within-generation state", "0.22 p_stress^circ", "stress-memory mark changes allocation", "within_generation"),
        ("ecological", "microclimate_buffering", "life_history", "maturation_progress", 0.16, "+", "most recent within-generation state", "0.16 e_3", "buffered microclimate supports maturation progress", "within_generation"),
        ("ecological", "food_resource_enrichment", "life_history", "growth_capacity", 0.34, "+", "most recent within-generation state", "0.34 e_2", "food enrichment raises growth capacity", "within_generation"),
        ("ecological", "soil_organic_matter", "life_history", "reproductive_allocation", 0.22, "+", "most recent within-generation state", "0.22 e_1", "modified soil or nest condition supports allocation", "within_generation"),
        ("background", "temperature_anomaly", "life_history", "maturation_progress", np.nan, "-", "most recent within-generation state", "2.22 sigma(8(u-eta))-0.16 b_2; eta contains +0.06 b_2", "temperature anomaly delays maturation progress", "within_generation"),
        ("background", "rainfall_anomaly", "life_history", "growth_capacity", 0.10, "+", "most recent within-generation state", "0.10 b_1", "rainfall anomaly raises growth-capacity input", "within_generation"),
        ("background", "temperature_anomaly", "life_history", "reproductive_allocation", np.nan, "+/-", "most recent within-generation state", "0.30 alpha-0.16 b_2; alpha=sigma(9(u-0.55+0.06 b_2-0.05 q_grow))", "temperature anomaly shifts allocation switch and direct allocation cost", "within_generation"),
        # Epigenetic inputs.
        ("development", "gene_1_regulatory_initiation", "epigenetic", "regulatory_mark", 0.32, "+", "most recent within-generation state", "0.32 xbar_dev,1", "developmental regulatory activity maintains regulatory mark", "within_generation"),
        ("development", "gene_3_stress_response", "epigenetic", "stress_memory_mark", 0.30, "+", "most recent within-generation state", "0.30 xbar_dev,3", "developmental stress response maintains stress-memory mark", "within_generation"),
        ("life_history", "growth_capacity", "epigenetic", "regulatory_mark", 0.22, "+", "most recent within-generation state", "0.22 q_grow", "growth state supports read-write maintenance of regulatory mark", "within_generation"),
        ("life_history", "reproductive_allocation", "epigenetic", "stress_memory_mark", 0.28, "+", "most recent within-generation state", "0.28 q_repr", "allocation state contributes to stress-memory maintenance", "within_generation"),
        ("ecological", "food_resource_enrichment", "epigenetic", "regulatory_mark", 0.20, "+", "most recent within-generation state", "0.20 e_2", "resource-enriched ecology supports regulatory mark reconstruction", "within_generation"),
        ("ecological", "microclimate_buffering", "epigenetic", "stress_memory_mark", -0.16, "-", "most recent within-generation state", "-0.16 e_3", "buffered ecology reduces stress-memory gain", "within_generation"),
        ("background", "temperature_anomaly", "epigenetic", "regulatory_mark", -0.14, "-", "most recent within-generation state", "-0.14 b_2", "temperature anomaly suppresses regulatory-mark maintenance", "within_generation"),
        ("background", "temperature_anomaly", "epigenetic", "stress_memory_mark", 0.22, "+", "most recent within-generation state", "0.22 b_2", "temperature anomaly promotes stress-memory mark", "within_generation"),
        # Ecological inputs.
        ("microbiome", "guild_1_growth_support", "ecological", "soil_organic_matter", 0.30, "+", "most recent within-generation state", "0.30 log(1+y_1)", "microbial support guild contributes to soil or nest organic state", "within_generation"),
        ("microbiome", "guild_1_growth_support", "ecological", "food_resource_enrichment", 0.12, "+", "most recent within-generation state", "0.12 log(1+y_1)", "support guild contributes to resource enrichment", "within_generation"),
        ("microbiome", "guild_5_cross_feeder", "ecological", "microclimate_buffering", 0.20, "+", "most recent within-generation state", "0.20 log(1+y_5)", "cross-feeder guild contributes to buffered microenvironment", "within_generation"),
        ("development", "gene_2_somatic_growth", "ecological", "food_resource_enrichment", 0.28, "+", "most recent within-generation state", "0.28 xbar_dev,2", "somatic-growth state modifies resource enrichment", "within_generation"),
        ("life_history", "reproductive_allocation", "ecological", "soil_organic_matter", 0.24, "+", "most recent within-generation state", "0.24 q_repr", "allocation modifies soil or nest organic state", "within_generation"),
        ("life_history", "reproductive_allocation", "ecological", "food_resource_enrichment", 0.32, "+", "most recent within-generation state", "0.32 q_repr", "organism allocation modifies food-resource enrichment", "within_generation"),
        ("life_history", "maturation_progress", "ecological", "microclimate_buffering", 0.22, "+", "most recent within-generation state", "0.22 q_mat", "maturation state modifies buffering behavior", "within_generation"),
        ("background", "temperature_anomaly", "ecological", "soil_organic_matter", -0.06, "-", "most recent within-generation state", "-0.06 b_2", "temperature anomaly reduces soil or nest organic state", "within_generation"),
        ("background", "temperature_anomaly", "ecological", "microclimate_buffering", -0.08, "-", "most recent within-generation state", "-0.08 b_2", "temperature anomaly reduces buffering", "within_generation"),
        # Inputs used to initialize the next generation.
        ("microbiome", "guild_1_growth_support", "ecological", "soil_organic_matter", 0.18, "+", "state at reproductive transition u_R=0.75", "0.18 log(1+y_1)", "microbiome state at transmission contributes to reconstructed organic state", "intergenerational"),
        ("life_history", "reproductive_allocation", "ecological", "soil_organic_matter", 0.18, "+", "state at reproductive transition u_R=0.75", "0.18 q_repr", "reproductive allocation at u_R represents provisioning that contributes to the reconstructed organic state", "intergenerational"),
        ("life_history", "reproductive_allocation", "ecological", "food_resource_enrichment", 0.18, "+", "state at reproductive transition u_R=0.75", "0.18 q_repr", "reproductive allocation at u_R represents provisioning that contributes to reconstructed resources", "intergenerational"),
        ("development", "gene_2_somatic_growth", "ecological", "food_resource_enrichment", 0.10, "+", "state at reproductive transition u_R=0.75", "0.10 tanh(x_dev,2)", "parental developmental condition at u_R acts indirectly through resource provisioning in the reconstructed ecological state", "intergenerational"),
        ("life_history", "maturation_progress", "ecological", "microclimate_buffering", 0.12, "+", "state at reproductive transition u_R=0.75", "0.12 q_mat", "parental condition at u_R contributes to establishment of the buffered offspring environment", "intergenerational"),
        ("microbiome", "guild_5_cross_feeder", "ecological", "microclimate_buffering", 0.10, "+", "state at reproductive transition u_R=0.75", "0.10 log(1+y_5)", "cross-feeder abundance at the transmission event contributes to reconstructed buffering", "intergenerational"),
        ("epigenetic", "regulatory_mark", "development", "gene_1_regulatory_initiation", 0.34, "+", "next-generation initial reconstruction", "0.34 p_reg^circ", "epigenetic state reconstructs regulatory initiation", "intergenerational"),
        ("epigenetic", "regulatory_mark", "development", "gene_2_somatic_growth", 0.16, "+", "next-generation initial reconstruction", "0.16 p_reg^circ", "epigenetic state reconstructs somatic-growth gene", "intergenerational"),
        ("epigenetic", "stress_memory_mark", "development", "gene_3_stress_response", 0.26, "+", "next-generation initial reconstruction", "0.26 p_stress^circ", "stress-memory state reconstructs stress-response gene", "intergenerational"),
        ("epigenetic", "regulatory_mark", "development", "gene_4_maturation_timing", 0.20, "+", "next-generation initial reconstruction", "0.20 p_reg^circ", "epigenetic state reconstructs maturation-timing gene", "intergenerational"),
        ("epigenetic", "stress_memory_mark", "development", "gene_5_allocation_signal", 0.12, "+", "next-generation initial reconstruction", "0.12 p_stress^circ", "stress-memory state reconstructs allocation signal", "intergenerational"),
        ("ecological", "food_resource_enrichment", "development", "gene_1_regulatory_initiation", 0.10, "+", "next-generation initial reconstruction", "0.10 e_2", "resource enrichment reconstructs regulatory initiation", "intergenerational"),
        ("ecological", "food_resource_enrichment", "development", "gene_2_somatic_growth", 0.22, "+", "next-generation initial reconstruction", "0.22 e_2", "resource enrichment reconstructs somatic-growth gene", "intergenerational"),
        ("ecological", "microclimate_buffering", "development", "gene_4_maturation_timing", 0.12, "+", "next-generation initial reconstruction", "0.12 e_3", "buffering reconstructs maturation-timing gene", "intergenerational"),
        ("ecological", "microclimate_buffering", "development", "gene_5_allocation_signal", 0.18, "+", "next-generation initial reconstruction", "0.18 e_3", "buffering reconstructs allocation signal", "intergenerational"),
        ("background", "rainfall_anomaly", "development", "gene_2_somatic_growth", 0.08, "+", "next-generation initial reconstruction", "0.08 b_1", "rainfall anomaly affects reconstructed growth gene", "intergenerational"),
        ("background", "temperature_anomaly", "development", "gene_3_stress_response", 0.10, "+", "next-generation initial reconstruction", "0.10 b_2", "temperature anomaly affects reconstructed stress gene", "intergenerational"),
        ("epigenetic", "regulatory_mark", "microbiome", "guild_1_growth_support", 0.04, "+", "next-generation initial reconstruction", "0.04 p_reg^circ", "epigenetic state contributes to support-guild carryover", "intergenerational"),
        ("epigenetic", "stress_memory_mark", "microbiome", "guild_4_opportunist", 0.05, "+", "next-generation initial reconstruction", "0.05 p_stress^circ", "stress-memory state contributes to opportunist carryover", "intergenerational"),
        ("ecological", "food_resource_enrichment", "microbiome", "guild_1_growth_support", 0.08, "+", "next-generation initial reconstruction", "0.08 e_2", "resource enrichment contributes to support-guild carryover", "intergenerational"),
        ("ecological", "soil_organic_matter", "microbiome", "guild_2_fiber_fermenter", 0.10, "+", "next-generation initial reconstruction", "0.10 e_1", "organic matter contributes to fermenter carryover", "intergenerational"),
        ("ecological", "microclimate_buffering", "microbiome", "guild_3_stress_tolerant", -0.08, "-", "next-generation initial reconstruction", "-0.08 e_3", "buffering reduces stress-tolerant carryover advantage", "intergenerational"),
        ("background", "temperature_anomaly", "microbiome", "guild_4_opportunist", 0.06, "+", "next-generation initial reconstruction", "0.06 b_2", "temperature anomaly contributes to opportunist carryover", "intergenerational"),
        ("ecological", "soil_organic_matter", "microbiome", "guild_5_cross_feeder", 0.06, "+", "next-generation initial reconstruction", "0.06 e_1", "organic matter contributes to cross-feeder carryover", "intergenerational"),
        ("epigenetic", "regulatory_mark", "life_history", "maturation_progress", 0.20, "+", "next-generation initial reconstruction", "0.20 p_reg^circ", "epigenetic state initializes maturation progress", "intergenerational"),
        ("ecological", "microclimate_buffering", "life_history", "maturation_progress", 0.10, "+", "next-generation initial reconstruction", "0.10 e_3", "buffering initializes maturation progress", "intergenerational"),
        ("background", "temperature_anomaly", "life_history", "maturation_progress", -0.08, "-", "next-generation initial reconstruction", "-0.08 b_2", "temperature anomaly suppresses initial maturation progress", "intergenerational"),
        ("epigenetic", "regulatory_mark", "life_history", "growth_capacity", 0.16, "+", "next-generation initial reconstruction", "0.16 p_reg^circ", "epigenetic state initializes growth capacity", "intergenerational"),
        ("ecological", "food_resource_enrichment", "life_history", "growth_capacity", 0.22, "+", "next-generation initial reconstruction", "0.22 e_2", "resource enrichment initializes growth capacity", "intergenerational"),
        ("background", "rainfall_anomaly", "life_history", "growth_capacity", 0.05, "+", "next-generation initial reconstruction", "0.05 b_1", "rainfall anomaly initializes growth capacity", "intergenerational"),
        ("epigenetic", "stress_memory_mark", "life_history", "reproductive_allocation", 0.12, "+", "next-generation initial reconstruction", "0.12 p_stress^circ", "stress-memory state initializes reproductive allocation", "intergenerational"),
        ("ecological", "soil_organic_matter", "life_history", "reproductive_allocation", 0.14, "+", "next-generation initial reconstruction", "0.14 e_1", "organic state initializes reproductive allocation", "intergenerational"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "source_level",
            "source_component",
            "target_level",
            "target_component",
            "coefficient",
            "sign",
            "time_alignment_or_lag",
            "mathematical_term",
            "biological_interpretation",
            "process",
        ],
    )


def effective_nonzero_cross_level_edges(par: Mapping[str, object]) -> pd.DataFrame:
    """Return the component inventory with the effective calibrated coefficients."""

    edges = nonzero_cross_level_edges().copy()
    edges["mathematical_term"] = (
        edges["mathematical_term"]
        .str.replace(r"\by_([1-5])\b", r"m_\1", regex=True)
    )

    def scale_simple(mask: pd.Series, scale: float) -> None:
        for idx in edges.index[mask]:
            coefficient = float(edges.at[idx, "coefficient"])
            if not np.isfinite(coefficient):
                continue
            effective = coefficient * scale
            old = f"{coefficient:.2f}"
            new = f"{effective:.6g}"
            edges.at[idx, "coefficient"] = effective
            edges.at[idx, "mathematical_term"] = str(edges.at[idx, "mathematical_term"]).replace(old, new)

    within = edges["process"].eq("within_generation")
    between = edges["process"].eq("intergenerational")
    epi = edges["source_level"].eq("epigenetic")
    eco = edges["source_level"].eq("ecological")
    dev = edges["source_level"].eq("development")
    development = edges["target_level"].eq("development")
    microbiome = edges["target_level"].eq("microbiome")
    life = edges["target_level"].eq("life_history")

    scale_simple(within & epi & development, _par_float(par, "epi_effect_development_scale"))
    scale_simple(between & epi & development, _par_float(par, "epi_development_start_scale"))
    scale_simple(within & epi & microbiome, _par_float(par, "epi_effect_microbiome_scale"))
    scale_simple(between & epi & microbiome, _par_float(par, "epi_microbiome_start_scale"))
    scale_simple(within & dev & microbiome, _par_float(par, "development_effect_microbiome_scale"))

    timing_scale = (
        _par_float(par, "epi_effect_life_history_scale")
        * _par_float(par, "epi_effect_life_timing_scale")
    )
    maturation_scale = (
        _par_float(par, "epi_effect_life_history_scale")
        * _par_float(par, "epi_effect_life_maturation_scale")
    )
    growth_scale = (
        _par_float(par, "epi_effect_life_history_scale")
        * _par_float(par, "epi_effect_life_growth_scale")
    )
    allocation_scale = (
        _par_float(par, "epi_effect_life_history_scale")
        * _par_float(par, "epi_effect_life_allocation_scale")
    )
    epi_life_within = within & epi & life
    maturation_row = epi_life_within & edges["target_component"].eq("maturation_progress")
    edges.loc[maturation_row, "mathematical_term"] = (
        f"{_par_float(par, 'life_maturation_wave_scale', 2.22):.6g} sigma(8(u-eta))"
        f"+{0.25 * maturation_scale:.6g} p_reg^circ; "
        f"eta contains -{0.07 * timing_scale:.6g} p_reg^circ"
    )
    scale_simple(epi_life_within & edges["target_component"].eq("growth_capacity"), growth_scale)
    scale_simple(epi_life_within & edges["target_component"].eq("reproductive_allocation"), allocation_scale)

    scale_simple(
        between & epi & life & edges["target_component"].eq("maturation_progress"),
        _par_float(par, "epi_life_start_maturation_scale"),
    )
    scale_simple(
        between & epi & life & edges["target_component"].eq("growth_capacity"),
        _par_float(par, "epi_life_start_growth_scale"),
    )
    scale_simple(
        between & epi & life & edges["target_component"].eq("reproductive_allocation"),
        _par_float(par, "epi_life_start_allocation_scale"),
    )

    scale_simple(within & eco & life, _par_float(par, "ecology_effect_life_history_scale"))
    scale_simple(between & eco & life, _par_float(par, "ecology_life_start_scale"))
    return edges


def intergenerational_equations_text() -> str:
    return r"""# Generation-start equations used in the corrected simulation

All indices use the article's generation index \(\tau\) and level-specific within-generation index \(t_l\).
The state available at the reproductive transition \(u_R=0.75\) is written
\(\mathbf{x}^{[l]}_{d,t_{l,j_l^R}(\tau)}\), where
\(j_l^R=\max\{j:t_{l,j}\leq u_R\}\).  The first retained state in the next generation is
\(\mathbf{x}^{[l]}_{d,t_l=0(\tau+1)}\).  The reproductive state is distinct from both the
selected \(m_l\)-state analysis segment and the terminal state at \(u=1\).

The epigenetic level stores two real-valued state variables
\(x^{[l_{\mathrm{epigenetic}}]}_{d,c,t_l(\tau)}\), with
\(c\in\{\mathrm{reg},\mathrm{stress}\}\).  Applying the logistic function gives the
bounded value \(p^{[l_{\mathrm{epigenetic}}]}_{d,c,t_l(\tau)}=
\sigma(x^{[l_{\mathrm{epigenetic}}]}_{d,c,t_l(\tau)})\) used in selected cross-level terms.

The fixed simulation uses partial epigenetic persistence
\[
x^{[l_{\mathrm{epigenetic}}]}_{d,c,t_l=0(\tau+1)}
=
\rho_c x^{[l_{\mathrm{epigenetic}}]}_{d,c,t_{l,j_l^R}(\tau)}
+
(1-\rho_c)z_{0,c}
+
\epsilon^{[\mathrm{epi,trans}]}_{c,d,\tau}.
\]
Here \(c\in\{\mathrm{reg},\mathrm{stress}\}\), \(\rho_{\mathrm{reg}}=0.64\),
\(\rho_{\mathrm{stress}}=0.60\), \(z_{0,c}=0\), and
\(\epsilon^{[\mathrm{epi,trans}]}_{c,d,\tau}\sim\mathcal{N}(0,0.10^2)\).

Developmental and life-history initial states are reconstructed, not directly inherited:
\[
\mathbf{x}^{[l_{\mathrm{development}}]}_{d,t_l=0(\tau+1)}
=
F^{[\mathrm{dev},0]}\!\left(
\mathbf{x}^{[l_{\mathrm{epigenetic}}]}_{d,t_l=0(\tau+1)},
\mathbf{x}^{[l_{\mathrm{ecological}}]}_{d,t_l=0(\tau+1)},
\mathbf{x}^{[\mathrm{background}]}_{d,t_l=0(\tau+1)}
\right)
+
\boldsymbol{\epsilon}^{[\mathrm{dev},0]}_{d,\tau},
\]
\[
\mathbf{x}^{[l_{\mathrm{life-history}}]}_{d,t_l=0(\tau+1)}
=
F^{[\mathrm{life},0]}\!\left(
\mathbf{x}^{[l_{\mathrm{epigenetic}}]}_{d,t_l=0(\tau+1)},
\mathbf{x}^{[l_{\mathrm{ecological}}]}_{d,t_l=0(\tau+1)},
\mathbf{x}^{[\mathrm{background}]}_{d,t_l=0(\tau+1)}
\right)
+
\boldsymbol{\epsilon}^{[\mathrm{life},0]}_{d,\tau}.
\]

The microbiome has partial carryover:
\[
\log \mathbf{x}^{[l_{\mathrm{microbiome}}]}_{d,t_l=0(\tau+1)}
=
F^{[\mathrm{micro},0]}\!\left(
\log \mathbf{x}^{[l_{\mathrm{microbiome}}]}_{d,t_{l,j_l^R}(\tau)},
\mathbf{x}^{[l_{\mathrm{epigenetic}}]}_{d,t_l=0(\tau+1)},
\mathbf{x}^{[l_{\mathrm{ecological}}]}_{d,t_l=0(\tau+1)},
\mathbf{x}^{[\mathrm{background}]}_{d,t_l=0(\tau+1)}
\right)
+
\boldsymbol{\epsilon}^{[\mathrm{micro},0]}_{d,\tau}.
\]

The ecological level persists with organism-mediated modification:
\[
\mathbf{x}^{[l_{\mathrm{ecological}}]}_{d,t_l=0(\tau+1)}
=
F^{[\mathrm{eco},0]}\!\left(
\mathbf{x}^{[l_{\mathrm{ecological}}]}_{d,t_{l,j_l^R}(\tau)},
\mathbf{x}^{[l_{\mathrm{microbiome}}]}_{d,t_{l,j_l^R}(\tau)},
\mathbf{x}^{[l_{\mathrm{life-history}}]}_{d,t_{l,j_l^R}(\tau)}
\right)
+
\boldsymbol{\epsilon}^{[\mathrm{eco},0]}_{d,\tau}.
\]

The background control remains outside the ELC:
\[
\mathbf{x}^{[\mathrm{background}]}_{d,t_l=0(\tau+1)}
=
\Phi_{\mathrm{bg}}
\mathbf{x}^{[\mathrm{background}]}_{d,t_{l,j_l^R}(\tau)}
+
\boldsymbol{\epsilon}^{[\mathrm{bg},0]}_{d,\tau}.
\]
The fixed simulation uses
\(\Phi_{\mathrm{bg}}=\operatorname{diag}(0.65,0.60)\) and no ELC-dependent term.
"""


def _params() -> dict[str, object]:
    return {
        "A_dev": np.array(
            [
                [-0.80, 0.70, 0.00, -0.40, 0.60],
                [-0.50, -0.90, 0.80, 0.00, -0.20],
                [0.30, -0.60, -0.70, 0.90, 0.00],
                [0.00, 0.40, -0.50, -1.00, 0.60],
                [0.70, 0.00, 0.30, -0.60, -0.80],
            ],
            dtype=float,
        ),
        "A_micro": np.array(
            [
                [-0.74, 0.14, -0.08, -0.12, 0.18],
                [0.10, -0.68, 0.12, -0.08, 0.08],
                [-0.08, 0.10, -0.70, 0.15, -0.06],
                [-0.16, -0.10, 0.12, -0.64, 0.08],
                [0.14, 0.12, -0.06, 0.10, -0.66],
            ],
            dtype=float,
        ),
        "sigma_dev": np.array([0.12, 0.11, 0.13, 0.11, 0.12], dtype=float),
        "sigma_micro": np.array([0.16, 0.15, 0.16, 0.17, 0.15], dtype=float),
        "sigma_life": np.array([0.14, 0.13, 0.15], dtype=float),
        "sigma_epi": np.array([0.16, 0.16], dtype=float),
        "sigma_eco": np.array([0.13, 0.13, 0.12], dtype=float),
        "sigma_bg": np.array([0.16, 0.16], dtype=float),
        "rho_epi": np.array([0.64, 0.60], dtype=float),
        "phi_bg": np.array([[0.65, 0.00], [0.00, 0.60]], dtype=float),
        "epi_founder_sd": 0.45,
        "epi_transmission_noise_sd": 0.10,
        "epi_effect_development_scale": 1.0,
        "development_effect_microbiome_scale": 1.0,
        "epi_effect_microbiome_scale": 1.0,
        "epi_effect_life_history_scale": 1.0,
        "epi_effect_life_timing_scale": 1.0,
        "epi_effect_life_maturation_scale": 1.0,
        "epi_effect_life_growth_scale": 1.0,
        "epi_effect_life_allocation_scale": 1.0,
        "epi_development_start_scale": 1.0,
        "epi_microbiome_start_scale": 1.0,
        "epi_life_start_scale": 1.0,
        "epi_life_start_maturation_scale": 1.0,
        "epi_life_start_growth_scale": 1.0,
        "epi_life_start_allocation_scale": 1.0,
        "life_maturation_timing_base": 0.42,
        "life_maturation_target_intercept": -0.92,
        "life_maturation_wave_scale": 2.22,
        "life_start_maturation_intercept": -1.30,
        "ecology_effect_life_history_scale": 1.0,
        "ecology_life_start_scale": 1.0,
        "ecological_elc_modification_scale": 1.0,
        "ecological_reconstruction_elc_scale": 1.0,
        "microbiome_to_ecology_modification_scale": 1.0,
        "microbiome_to_ecology_reconstruction_scale": 1.0,
    }


def apply_parameter_overrides(par: dict[str, object], overrides: Mapping[str, float] | tuple[tuple[str, float], ...] | None = None) -> dict[str, object]:
    """Apply named calibration changes without changing the model architecture."""

    if overrides is None:
        return par
    items = overrides.items() if hasattr(overrides, "items") else overrides
    for key, value in items:
        value_f = float(value)
        if key == "sigma_epi_reg":
            arr = np.asarray(par["sigma_epi"], dtype=float).copy()
            arr[0] = value_f
            par["sigma_epi"] = arr
        elif key == "sigma_epi_stress":
            arr = np.asarray(par["sigma_epi"], dtype=float).copy()
            arr[1] = value_f
            par["sigma_epi"] = arr
        elif key == "rho_epi_reg":
            arr = np.asarray(par["rho_epi"], dtype=float).copy()
            arr[0] = value_f
            par["rho_epi"] = arr
        elif key == "rho_epi_stress":
            arr = np.asarray(par["rho_epi"], dtype=float).copy()
            arr[1] = value_f
            par["rho_epi"] = arr
        elif key in par:
            par[key] = value_f
        else:
            raise KeyError(f"unknown calibration parameter {key!r}")
    return par


def _par_float(par: Mapping[str, object], key: str, default: float = 1.0) -> float:
    return float(par.get(key, default))


def _phase(tau: int, d: int, u: float) -> dict[str, float]:
    lineage_phase = 2.0 * np.pi * ((d + 1) % 37) / 37.0
    return {
        "dev": float(np.sin(2.0 * np.pi * u + lineage_phase + 0.41 * tau)),
        "micro": float(np.cos(2.0 * np.pi * u - 0.25 * lineage_phase + 0.29 * tau)),
        "slow": float(np.sin(np.pi * u + 0.17 * tau + 0.5 * lineage_phase)),
    }


def _update_development(state: dict[str, np.ndarray], tau: int, d: int, u: float, dt: float, rng: np.random.Generator, config: PilotConfig, par: Mapping[str, np.ndarray]) -> np.ndarray:
    x = state["development"]
    y = state["microbiome"]
    life = state["life_history"]
    epi_centered = sigmoid(state["epigenetic"]) - 0.5
    eco = state["ecological"]
    bg = state["background"]
    phase = _phase(tau, d, u)
    inputs = np.zeros(5)
    epi_dev_scale = _par_float(par, "epi_effect_development_scale")
    inputs[0] += epi_dev_scale * 0.34 * epi_centered[0]
    inputs[1] += 0.18 * np.log1p(y[0]) + 0.22 * eco[1] + 0.10 * bg[0]
    inputs[2] += 0.16 * np.log1p(y[3]) + epi_dev_scale * 0.28 * epi_centered[1] + 0.12 * bg[1]
    inputs[3] += 0.15 * life[0]
    inputs[4] += 0.18 * eco[2] + 0.12 * life[2]
    stage = np.array([0.15 * phase["dev"], 0.20 * phase["slow"], -0.12 * phase["micro"], 0.18 * np.sin(np.pi * u), 0.16 * np.cos(np.pi * u)])
    drive = par["A_dev"] @ x + inputs + stage
    drift = -0.32 * x + 1.18 * np.tanh(drive)
    return x + dt * drift + config.noise_scale * par["sigma_dev"] * np.sqrt(dt) * rng.normal(size=5)


def _update_microbiome(state: dict[str, np.ndarray], tau: int, d: int, u: float, dt: float, rng: np.random.Generator, config: PilotConfig, par: Mapping[str, np.ndarray]) -> np.ndarray:
    z = np.log(state["microbiome"])
    y = state["microbiome"]
    dev = np.tanh(state["development"])
    epi_centered = sigmoid(state["epigenetic"]) - 0.5
    eco = state["ecological"]
    bg = state["background"]
    phase = _phase(tau, d, u)
    r = np.array([0.50, 0.38, 0.34, 0.30, 0.40])
    dev_micro_scale = _par_float(par, "development_effect_microbiome_scale")
    epi_micro_scale = _par_float(par, "epi_effect_microbiome_scale")
    r[0] += dev_micro_scale * 0.18 * dev[1] + epi_micro_scale * 0.10 * epi_centered[0] + 0.20 * eco[1] + 0.08 * bg[0] + 0.10 * phase["micro"]
    r[1] += 0.18 * eco[0] + 0.08 * phase["slow"]
    r[2] += -0.16 * eco[2] + dev_micro_scale * 0.08 * dev[2]
    r[3] += dev_micro_scale * 0.16 * dev[2] + epi_micro_scale * 0.12 * epi_centered[1] + 0.10 * bg[1] - 0.10 * phase["micro"]
    r[4] += 0.10 * y[0] + 0.07 * eco[0]
    sigma = config.noise_scale * par["sigma_micro"]
    dz = (r + par["A_micro"] @ y - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.normal(size=5)
    return np.exp(z + dz)


def _update_life_history(state: dict[str, np.ndarray], tau: int, d: int, u: float, dt: float, rng: np.random.Generator, config: PilotConfig, par: Mapping[str, np.ndarray]) -> np.ndarray:
    life = state["life_history"]
    z = logit(life)
    dev = np.tanh(state["development"])
    y = state["microbiome"]
    epi_centered = sigmoid(state["epigenetic"]) - 0.5
    eco = state["ecological"]
    bg = state["background"]
    phase = _phase(tau, d, u)
    epi_life_scale = _par_float(par, "epi_effect_life_history_scale")
    epi_life_timing_scale = epi_life_scale * _par_float(par, "epi_effect_life_timing_scale")
    epi_life_maturation_scale = epi_life_scale * _par_float(par, "epi_effect_life_maturation_scale")
    epi_life_growth_scale = epi_life_scale * _par_float(par, "epi_effect_life_growth_scale")
    epi_life_allocation_scale = epi_life_scale * _par_float(par, "epi_effect_life_allocation_scale")
    eco_life_scale = _par_float(par, "ecology_effect_life_history_scale")
    timing_shift = (
        _par_float(par, "life_maturation_timing_base", 0.42)
        - 0.10 * dev[3]
        - 0.08 * np.log1p(y[0])
        - epi_life_timing_scale * 0.07 * epi_centered[0]
        + 0.06 * bg[1]
    )
    maturation_wave = sigmoid(8.0 * (u - timing_shift))
    U_maturation = (
        + _par_float(par, "life_maturation_target_intercept", -0.92)
        + _par_float(par, "life_maturation_wave_scale", 2.22) * maturation_wave
        + 0.42 * dev[3]
        + 0.18 * np.log1p(y[0])
        + epi_life_maturation_scale * 0.25 * epi_centered[0]
        + eco_life_scale * 0.16 * eco[2]
        - 0.16 * bg[1]
        + 0.20 * phase["slow"]
    )
    growth_window = np.sin(np.pi * u)
    late_cost = sigmoid(10.0 * (u - 0.72))
    U_growth = (
        -0.05
        + 0.66 * dev[1]
        + 0.36 * np.log1p(y[0] + 0.6 * y[1])
        + epi_life_growth_scale * 0.24 * epi_centered[0]
        + eco_life_scale * 0.34 * eco[1]
        + 0.10 * bg[0]
        - 0.24 * life[2]
        + 0.42 * growth_window
        - 0.28 * late_cost
        + 0.16 * phase["micro"]
    )
    allocation_switch = sigmoid(9.0 * (u - 0.55 + 0.06 * bg[1] - 0.05 * life[1]))
    U_allocation = (
        -1.26
        + 1.18 * life[0]
        + 0.62 * life[1]
        + 0.30 * allocation_switch
        + 0.24 * dev[4]
        + epi_life_allocation_scale * 0.22 * epi_centered[1]
        + eco_life_scale * 0.22 * eco[0]
        - 0.16 * bg[1]
        + 0.18 * phase["dev"]
        - 0.16 * life[1] * late_cost
    )
    target = np.array([U_maturation, U_growth, U_allocation])
    dz = 2.15 * (target - z) * dt + config.noise_scale * par["sigma_life"] * np.sqrt(dt) * rng.normal(size=3)
    return sigmoid(z + dz)


def _update_epigenetic(state: dict[str, np.ndarray], tau: int, d: int, u: float, dt: float, rng: np.random.Generator, config: PilotConfig, par: Mapping[str, np.ndarray]) -> np.ndarray:
    z = state["epigenetic"]
    p = sigmoid(z)
    h_reg = hill(p[0], 0.52, 3.0)
    h_stress = hill(p[1], 0.50, 3.0)
    dev = np.tanh(state["development"])
    life = state["life_history"]
    eco = state["ecological"]
    bg = state["background"]
    phase = _phase(tau, d, u)
    reg_target = (
        0.38 * (2.0 * h_reg - 1.0)
        - 0.26 * (2.0 * h_stress - 1.0)
        + 0.32 * dev[0]
        + 0.22 * life[1]
        + 0.20 * eco[1]
        - 0.14 * bg[1]
        + 0.20 * phase["slow"]
    )
    stress_target = (
        0.36 * (2.0 * h_stress - 1.0)
        - 0.24 * (2.0 * h_reg - 1.0)
        + 0.30 * dev[2]
        + 0.28 * life[2]
        - 0.16 * eco[2]
        + 0.22 * bg[1]
        + 0.18 * phase["micro"]
    )
    dz_reg = 0.92 * (reg_target - z[0])
    dz_stress = 0.88 * (stress_target - z[1])
    drift = np.array([dz_reg, dz_stress])
    return z + dt * drift + config.noise_scale * par["sigma_epi"] * np.sqrt(dt) * rng.normal(size=2)


def _update_ecology(state: dict[str, np.ndarray], tau: int, d: int, u: float, dt: float, rng: np.random.Generator, config: PilotConfig, par: Mapping[str, np.ndarray]) -> np.ndarray:
    q = state["ecological"]
    dev = np.tanh(state["development"])
    y = state["microbiome"]
    life = state["life_history"]
    bg = state["background"]
    phase = _phase(tau, d, u)
    recurrent = np.array(
        [
            -0.22 * q[0] + 0.06 * np.tanh(q[1]) - 0.04 * q[0] ** 3,
            -0.20 * q[1] + 0.05 * np.tanh(q[0]) + 0.04 * np.tanh(q[2]) - 0.04 * q[1] ** 3,
            -0.18 * q[2] + 0.06 * np.tanh(q[1]) - 0.04 * q[2] ** 3,
        ]
    )
    eco_mod_scale = _par_float(par, "ecological_elc_modification_scale")
    micro_eco_mod_scale = _par_float(par, "microbiome_to_ecology_modification_scale")
    modification = np.array(
        [
            eco_mod_scale * (micro_eco_mod_scale * 0.30 * np.log1p(y[0]) + 0.24 * life[2]) - 0.06 * bg[1],
            eco_mod_scale * (0.28 * dev[1] + 0.32 * life[2] + micro_eco_mod_scale * 0.12 * np.log1p(y[0])) + 0.06 * phase["slow"],
            eco_mod_scale * (0.22 * life[0] + micro_eco_mod_scale * 0.20 * np.log1p(y[4])) - 0.08 * bg[1] + 0.08 * phase["dev"],
        ]
    )
    drift = recurrent + modification
    return q + dt * drift + config.noise_scale * par["sigma_eco"] * np.sqrt(dt) * rng.normal(size=3)


def _update_background(state: dict[str, np.ndarray], tau: int, d: int, u: float, dt: float, rng: np.random.Generator, config: PilotConfig, par: Mapping[str, np.ndarray]) -> np.ndarray:
    bg = state["background"]
    drift = np.array([-0.42 * bg[0], -0.38 * bg[1]])
    return bg + dt * drift + config.noise_scale * par["sigma_bg"] * np.sqrt(dt) * rng.normal(size=2)


UPDATE_FUNCS: dict[str, Callable[..., np.ndarray]] = {
    "development": _update_development,
    "microbiome": _update_microbiome,
    "life_history": _update_life_history,
    "epigenetic": _update_epigenetic,
    "ecological": _update_ecology,
    "background": _update_background,
}


def _founder_states(rng: np.random.Generator, config: PilotConfig, par: Mapping[str, object] | None = None) -> list[dict[str, np.ndarray]]:
    states: list[dict[str, np.ndarray]] = []
    par = _params() if par is None else par
    for _ in range(config.n_lineages):
        epi = rng.normal(0.0, _par_float(par, "epi_founder_sd", 0.45), config.epi_dim)
        eco = rng.normal(0.0, 0.38, config.eco_dim)
        bg = rng.normal(0.0, 0.45, config.bg_dim)
        dev = rng.normal(0.0, 0.18, config.dev_dim)
        micro = np.exp(rng.normal(-0.42, 0.18, config.micro_dim))
        life = sigmoid(rng.normal([-1.25, -0.15, -1.35], [0.18, 0.16, 0.16]))
        states.append(
            {
                "development": dev,
                "microbiome": micro,
                "life_history": life,
                "epigenetic": epi,
                "ecological": eco,
                "background": bg,
            }
        )
    return states


def _next_generation_start(prev: Mapping[str, np.ndarray], rng: np.random.Generator, config: PilotConfig, par: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    dev_final = prev["development"]
    micro_final = prev["microbiome"]
    life_final = prev["life_history"]
    epi_final = prev["epigenetic"]
    eco_final = prev["ecological"]
    bg_final = prev["background"]
    noise = config.noise_scale

    bg_start = par["phi_bg"] @ bg_final + noise * rng.normal(0.0, [0.20, 0.20])
    epi_start = par["rho_epi"] * epi_final + (1.0 - par["rho_epi"]) * np.zeros(2) + noise * rng.normal(0.0, _par_float(par, "epi_transmission_noise_sd", 0.10), 2)

    eco_recon_scale = _par_float(par, "ecological_reconstruction_elc_scale")
    micro_eco_recon_scale = _par_float(par, "microbiome_to_ecology_reconstruction_scale")
    eco_start = np.array(
        [
            0.64 * eco_final[0] + 0.12 * eco_final[1] + eco_recon_scale * (micro_eco_recon_scale * 0.18 * np.log1p(micro_final[0]) + 0.18 * life_final[2]),
            0.60 * eco_final[1] + 0.10 * eco_final[2] + eco_recon_scale * (0.18 * life_final[2] + 0.10 * np.tanh(dev_final[1])),
            0.62 * eco_final[2] + 0.14 * eco_final[0] + eco_recon_scale * (0.12 * life_final[0] + micro_eco_recon_scale * 0.10 * np.log1p(micro_final[4])),
        ]
    )
    eco_start += noise * rng.normal(0.0, 0.16, 3)

    epi_centered = sigmoid(epi_start) - 0.5
    epi_dev_start_scale = _par_float(par, "epi_development_start_scale")
    dev_start = np.array(
        [
            epi_dev_start_scale * 0.34 * epi_centered[0] + 0.10 * eco_start[1],
            0.22 * eco_start[1] + epi_dev_start_scale * 0.16 * epi_centered[0] + 0.08 * bg_start[0],
            epi_dev_start_scale * 0.26 * epi_centered[1] + 0.10 * bg_start[1],
            epi_dev_start_scale * 0.20 * epi_centered[0] + 0.12 * eco_start[2],
            0.18 * eco_start[2] + epi_dev_start_scale * 0.12 * epi_centered[1],
        ]
    )
    dev_start += noise * rng.normal(0.0, 0.12, 5)

    epi_micro_start_scale = _par_float(par, "epi_microbiome_start_scale")
    micro_log_start = (
        0.45 * np.log(micro_final)
        + 0.55 * np.array([-0.45, -0.55, -0.60, -0.62, -0.58])
        + np.array([0.08 * eco_start[1] + epi_micro_start_scale * 0.04 * epi_centered[0], 0.10 * eco_start[0], -0.08 * eco_start[2], 0.06 * bg_start[1] + epi_micro_start_scale * 0.05 * epi_centered[1], 0.06 * eco_start[0]])
    )
    micro_start = np.exp(micro_log_start + noise * rng.normal(0.0, 0.12, 5))

    epi_life_start_scale = _par_float(par, "epi_life_start_scale")
    epi_life_start_maturation_scale = epi_life_start_scale * _par_float(par, "epi_life_start_maturation_scale")
    epi_life_start_growth_scale = epi_life_start_scale * _par_float(par, "epi_life_start_growth_scale")
    epi_life_start_allocation_scale = epi_life_start_scale * _par_float(par, "epi_life_start_allocation_scale")
    eco_life_start_scale = _par_float(par, "ecology_life_start_scale")
    life_start_logits = np.array(
        [
            _par_float(par, "life_start_maturation_intercept", -1.30)
            + epi_life_start_maturation_scale * 0.20 * epi_centered[0]
            + eco_life_start_scale * 0.10 * eco_start[2]
            - 0.08 * bg_start[1],
            -0.25 + eco_life_start_scale * 0.22 * eco_start[1] + epi_life_start_growth_scale * 0.16 * epi_centered[0] + 0.05 * bg_start[0],
            -1.45 + eco_life_start_scale * 0.14 * eco_start[0] + epi_life_start_allocation_scale * 0.12 * epi_centered[1],
        ]
    )
    life_start = sigmoid(life_start_logits + noise * rng.normal(0.0, 0.12, 3))

    return {
        "development": dev_start,
        "microbiome": micro_start,
        "life_history": life_start,
        "epigenetic": epi_start,
        "ecological": eco_start,
        "background": bg_start,
    }


def fixed_random_second_parent_pairs(n_lineages: int, seed: int) -> np.ndarray:
    """Pair each lineage with a different lineage using a fixed random seed."""

    rng = np.random.default_rng(seed + 7919)
    pairs = np.empty(n_lineages, dtype=int)
    for d in range(n_lineages):
        choices = np.delete(np.arange(n_lineages), d)
        pairs[d] = int(rng.choice(choices))
    return pairs


def _next_generation_start_multiple_parent(
    prev_receiving: Mapping[str, np.ndarray],
    prev_contributor: Mapping[str, np.ndarray],
    rng: np.random.Generator,
    config: PilotConfig,
    par: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Initialize epigenetic states using contributions from lineages d and d'."""

    start = _next_generation_start(prev_receiving, rng, config, par)
    epi_d = prev_receiving["epigenetic"]
    epi_dp = prev_contributor["epigenetic"]
    interaction = np.array([0.04 * epi_d[0] * epi_dp[0], 0.03 * epi_d[1] * epi_dp[1]])
    inherited = np.array([0.58, 0.56]) * epi_d + np.array([0.22, 0.20]) * epi_dp
    baseline = (1.0 - np.array([0.58 + 0.22, 0.56 + 0.20])) * np.zeros(2)
    start["epigenetic"] = inherited + baseline + interaction + config.noise_scale * rng.normal(0.0, _par_float(par, "epi_transmission_noise_sd", 0.10), 2)
    return start


def _empty_arrays(config: PilotConfig) -> dict[str, np.ndarray]:
    shape = (config.n_lineages, config.n_generations + 1)
    return {
        "development": np.zeros(shape + (config.m_dev, config.dev_dim)),
        "microbiome": np.zeros(shape + (config.m_micro, config.micro_dim)),
        "life_history": np.zeros(shape + (config.m_life, config.life_dim)),
        "epigenetic": np.zeros(shape + (config.m_epi, config.epi_dim)),
        "ecological": np.zeros(shape + (config.m_eco, config.eco_dim)),
        "background": np.zeros(shape + (config.m_bg, config.bg_dim)),
    }


def _empty_full_arrays(config: PilotConfig) -> dict[str, np.ndarray]:
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
        level: np.zeros(shape + (level_T(config)[level], dims[level]))
        for level in dims
    }


def _empty_reproductive_arrays(config: PilotConfig) -> dict[str, np.ndarray]:
    shape = (config.n_lineages, config.n_generations + 1)
    dims = {
        "development": config.dev_dim,
        "microbiome": config.micro_dim,
        "life_history": config.life_dim,
        "epigenetic": config.epi_dim,
        "ecological": config.eco_dim,
        "background": config.bg_dim,
    }
    return {level: np.zeros(shape + (dims[level],)) for level in dims}


def integrate_generation(start: Mapping[str, np.ndarray], tau: int, d: int, rng: np.random.Generator, config: PilotConfig, par: Mapping[str, np.ndarray], schedule: Iterable[tuple[float, tuple[str, ...]]]) -> dict[str, np.ndarray]:
    retained = {
        "development": np.zeros((config.T_dev, config.dev_dim)),
        "microbiome": np.zeros((config.T_micro, config.micro_dim)),
        "life_history": np.zeros((config.T_life, config.life_dim)),
        "epigenetic": np.zeros((config.T_epi, config.epi_dim)),
        "ecological": np.zeros((config.T_eco, config.eco_dim)),
        "background": np.zeros((config.T_bg, config.bg_dim)),
    }
    state = {k: np.asarray(v, dtype=float).copy() for k, v in start.items()}
    step_counts = {level: 0 for level in level_steps(config)}
    retain_steps = {level: set(full_retained_step_indices(steps, level_T(config)[level])) for level, steps in level_steps(config).items()}
    retain_positions = {level: {step: i for i, step in enumerate(sorted(retain_steps[level]))} for level in retain_steps}
    for level in retained:
        retained[level][0] = state[level]

    for u, due_levels in schedule:
        pre = {k: v.copy() for k, v in state.items()}
        updates = {}
        for level in due_levels:
            step_counts[level] += 1
            dt = 1.0 / level_steps(config)[level]
            updates[level] = UPDATE_FUNCS[level](pre, tau, d, u, dt, rng, config, par)
        state.update(updates)
        for level in due_levels:
            step = step_counts[level]
            if step in retain_positions[level]:
                retained[level][retain_positions[level][step]] = state[level]
    return retained


def simulate_pilot(config: PilotConfig) -> dict[str, object]:
    rng = np.random.default_rng(config.seed)
    par = _params()
    schedule = build_event_schedule(config)
    arrays = _empty_arrays(config)
    full_arrays = _empty_full_arrays(config)
    reproductive_arrays = _empty_reproductive_arrays(config)
    starts = _founder_states(rng, config, par)

    for d in range(config.n_lineages):
        retained = integrate_generation(starts[d], 0, d, rng, config, par, schedule)
        for key in arrays:
            full_arrays[key][d, 0] = retained[key]
            arrays[key][d, 0] = extract_analysis_segment(retained[key], key, config)
            reproductive_arrays[key][d, 0] = reproductive_state(retained[key], key, config)

    for tau in range(config.n_generations):
        for d in range(config.n_lineages):
            prev_reproductive = {key: reproductive_arrays[key][d, tau] for key in arrays}
            start = _next_generation_start(prev_reproductive, rng, config, par)
            retained = integrate_generation(start, tau + 1, d, rng, config, par, schedule)
            for key in arrays:
                full_arrays[key][d, tau + 1] = retained[key]
                arrays[key][d, tau + 1] = extract_analysis_segment(retained[key], key, config)
                reproductive_arrays[key][d, tau + 1] = reproductive_state(retained[key], key, config)

    return {
        "config": config,
        "parameters": par,
        "full_time_series": full_arrays,
        "reproductive_states": reproductive_arrays,
        "timestamps": level_timestamps(config),
        "segment_indices": level_segment_indices(config),
        "reproductive_indices": reproductive_indices(config),
        **arrays,
    }


def phenotype_states(data: Mapping[str, object], config: PilotConfig) -> pd.DataFrame:
    rows = []
    for d in range(config.n_lineages):
        for tau in range(config.burn_in_generations + 1, config.n_generations + 1):
            life = data["full_time_series"]["life_history"][d, tau]
            life_times = np.asarray(data["timestamps"]["life_history"], dtype=float)
            maturation_trajectory = life[:, 0]
            crossing = np.where(maturation_trajectory >= config.theta_maturation)[0]
            t_mat = int(crossing[0]) if crossing.size else -1
            crossing_u = float(life_times[t_mat]) if t_mat >= 0 else np.nan
            early = bool(t_mat >= 0 and crossing_u < config.early_maturation_u)
            growth_late = float(life[-1, 1])
            high_growth = growth_late >= config.theta_growth
            if early and high_growth:
                phenotype = "nu_early_maturation_high_growth"
            elif early and not high_growth:
                phenotype = "early_maturation_low_growth"
            elif (not early) and high_growth:
                phenotype = "non_early_maturation_high_growth"
            else:
                phenotype = "non_early_maturation_low_growth"
            rows.append(
                {
                    "d": d + 1,
                    "tau": tau,
                    "maturation_threshold_crossing_t_l": t_mat,
                    "maturation_threshold_crossing_u": crossing_u,
                    "growth_capacity_final_t_l_m_l_minus_1": growth_late,
                    "phenotype_state": phenotype,
                }
            )
    return pd.DataFrame(rows)


def phenotype_count_table(data: Mapping[str, object], config: PilotConfig) -> pd.DataFrame:
    states = phenotype_states(data, config)
    counts = states["phenotype_state"].value_counts().reindex(
        [
            "nu_early_maturation_high_growth",
            "early_maturation_low_growth",
            "non_early_maturation_high_growth",
            "non_early_maturation_low_growth",
        ],
        fill_value=0,
    )
    total = int(counts.sum())
    return pd.DataFrame(
        {
            "phenotype_state": counts.index,
            "count": counts.to_numpy(dtype=int),
            "proportion": counts.to_numpy(dtype=float) / max(total, 1),
        }
    )


def acceptance_criteria(data: Mapping[str, object], config: PilotConfig) -> pd.DataFrame:
    rows = []
    d0 = config.illustrative_lineage_d - 1
    start = config.plot_generation_start
    stop = config.plot_generation_stop
    window = slice(start, stop + 1)
    arrays = {key: np.asarray(data["full_time_series"][key], dtype=float) for key in RETAINED_KEYS}
    reproductive = {key: np.asarray(data["reproductive_states"][key], dtype=float) for key in RETAINED_KEYS}

    def add(name: str, value: float, threshold: float, passed: bool, interpretation: str) -> None:
        rows.append(
            {
                "criterion": name,
                "value": value,
                "threshold": threshold,
                "passes": bool(passed),
                "interpretation": interpretation,
            }
        )

    dev_range = float(np.ptp(arrays["development"][d0, window].reshape(-1, config.dev_dim), axis=0).min())
    add("developmental transitions", dev_range, 0.20, dev_range >= 0.20, "each developmental gene varies across the five-generation illustrative window")

    micro = arrays["microbiome"]
    rel = micro / micro.sum(axis=-1, keepdims=True)
    lineage_rel_ranges = np.ptp(
        rel[:, window].reshape(config.n_lineages, -1, config.micro_dim),
        axis=1,
    ).mean(axis=1)
    rel_range = float(lineage_rel_ranges.mean())
    micro_min = float(micro.min())
    add(
        "microbiome turnover",
        rel_range,
        0.08,
        rel_range >= 0.08,
        "seed-wide mean lineage-level relative-abundance range across microbial guilds",
    )
    add("microbiome persistence without zero collapse", micro_min, 1e-5, micro_min > 1e-5, "minimum abundance remains positive without clipping")

    life_range = float(np.ptp(arrays["life_history"][d0, window].reshape(-1, config.life_dim), axis=0).min())
    add("life-history progression", life_range, 0.08, life_range >= 0.08, "each life-history component changes across the illustrative window")

    epi = arrays["epigenetic"]
    epi_lag = np.corrcoef(reproductive["epigenetic"][:, :-1, 0].ravel(), epi[:, 1:, 0, 0].ravel())[0, 1]
    epi_range = float(np.ptp(epi[d0, window].reshape(-1, config.epi_dim), axis=0).min())
    epi_prob = sigmoid(epi)
    epi_prob_range = float(np.ptp(epi_prob[d0, window].reshape(-1, config.epi_dim), axis=0).min())
    epi_direction_changes = []
    for j in range(config.epi_dim):
        diffs = np.diff(epi[d0, window].reshape(-1, config.epi_dim)[:, j])
        signs = np.sign(diffs[np.abs(diffs) > 1e-8])
        changes = int(np.sum(signs[1:] * signs[:-1] < 0)) if signs.size > 1 else 0
        epi_direction_changes.append(changes)
    min_epi_direction_changes = int(min(epi_direction_changes))
    epi_prob_min = float(np.min(epi_prob[d0, window]))
    epi_prob_max = float(np.max(epi_prob[d0, window]))
    add("epigenetic persistence", float(epi_lag), 0.35, epi_lag >= 0.35, "future initial regulatory mark correlates with previous final mark")
    add("epigenetic within-generation variation", epi_range, 0.08, epi_range >= 0.08, "epigenetic components vary within/across the plotted generations")
    add("epigenetic probability-scale variation", epi_prob_range, 0.05, epi_prob_range >= 0.05, "logit trajectories imply visible bounded probability changes")
    add("epigenetic nonmonotone illustrative trajectory", float(min_epi_direction_changes), 1.0, min_epi_direction_changes >= 1, "each epigenetic logit has at least one direction change in the illustrative window")
    add("epigenetic probabilities not saturated", max(abs(epi_prob_min - 0.5), abs(epi_prob_max - 0.5)), 0.48, epi_prob_min > 0.02 and epi_prob_max < 0.98, "transformed probabilities remain away from the numerical boundaries")

    eco = arrays["ecological"]
    eco_lag = np.corrcoef(reproductive["ecological"][:, :-1, 0].ravel(), eco[:, 1:, 0, 0].ravel())[0, 1]
    eco_range = float(np.ptp(eco[d0, window].reshape(-1, config.eco_dim), axis=0).min())
    add("ecological persistence with modification", float(eco_lag), 0.30, eco_lag >= 0.30, "future initial ecological state correlates with previous final ecological state")
    add("ecological variation", eco_range, 0.12, eco_range >= 0.12, "ecological components vary across the illustrative window")

    bg = arrays["background"]
    bg_lag = np.corrcoef(reproductive["background"][:, :-1, 0].ravel(), bg[:, 1:, 0, 0].ravel())[0, 1]
    bg_range = float(np.ptp(bg[d0, window].reshape(-1, config.bg_dim), axis=0).min())
    add("background autonomous persistence", float(bg_lag), 0.20, bg_lag >= 0.20, "background retains Markov autocorrelation")
    add("background variation", bg_range, 0.10, bg_range >= 0.10, "background components vary without ELC feedback")

    max_abs = max(float(np.max(np.abs(arrays[key]))) for key in arrays if key != "microbiome")
    add("finite nondivergent states", max_abs, 8.0, np.isfinite(max_abs) and max_abs < 8.0, "non-microbial states remain finite and bounded by model dynamics")
    all_finite = all(np.all(np.isfinite(arrays[key])) for key in arrays)
    add("no NaN or infinity", float(all_finite), 1.0, all_finite, "all retained arrays contain finite values")

    all_maturation = arrays["life_history"][:, config.burn_in_generations + 1 :, :, 0]
    crosses = all_maturation >= config.theta_maturation
    crossing_times = np.full(crosses.shape[:2], -1, dtype=int)
    for t_l in range(crosses.shape[2]):
        crossing_times[(crossing_times < 0) & crosses[:, :, t_l]] = t_l
    finite_crossings = crossing_times[crossing_times >= 0]
    crossing_unique_count = int(np.unique(finite_crossings).size) if finite_crossings.size else 0
    life_times = level_timestamps(config)["life_history"]
    crossing_u = np.full(crossing_times.shape, np.nan, dtype=float)
    valid_crossings = crossing_times >= 0
    crossing_u[valid_crossings] = life_times[crossing_times[valid_crossings]]
    early_prop = float(np.mean(valid_crossings & (crossing_u < config.early_maturation_u)))
    non_early_prop = float(1.0 - early_prop)
    growth_trajectory_range = float(np.ptp(arrays["life_history"][d0, window, :, 1]))
    allocation_trajectory_range = float(np.ptp(arrays["life_history"][d0, window, :, 2]))
    life_min = float(np.min(arrays["life_history"][d0, window]))
    life_max = float(np.max(arrays["life_history"][d0, window]))
    immediate_prop = float(np.mean(crossing_times == 0))
    add("maturation threshold-crossing variation", float(crossing_unique_count), 2.0, crossing_unique_count >= 2, "maturation threshold crossing occurs at more than one retained t_l")
    add("growth-capacity trajectory variation", growth_trajectory_range, 0.08, growth_trajectory_range >= 0.08, "growth capacity varies across retained life-history trajectories")
    add("reproductive-allocation trajectory variation", allocation_trajectory_range, 0.08, allocation_trajectory_range >= 0.08, "reproductive allocation varies across retained life-history trajectories")
    add("early maturation support", early_prop, 0.05, early_prop >= 0.05, "at least 5 percent of post-burn-in observations classify as early maturation")
    add("non-early maturation support", non_early_prop, 0.05, non_early_prop >= 0.05, "at least 5 percent of post-burn-in observations classify as threshold crossing after t_early or never crossing")
    add("no immediate maturation saturation", immediate_prop, 0.95, immediate_prop < 0.95, "maturation does not cross the threshold at t_l=0 for nearly all observations")
    add("life-history no boundary artifacts", min(life_min, 1.0 - life_max), 0.005, life_min > 0.005 and life_max < 0.995, "bounded life-history variables do not sit on numerical boundaries")

    counts = phenotype_count_table(data, config)
    min_count = int(counts["count"].min())
    add("phenotype-state support", float(min_count), 40.0, min_count >= 40, "all four phenotype states have at least 5 percent support in the 800 post-burn-in pilot observations")

    return pd.DataFrame(rows)


def convergence_check(config: PilotConfig) -> pd.DataFrame:
    base = replace(config, n_lineages=20, n_generations=5, noise_scale=0.0)
    fine = replace(
        base,
        dev_steps=base.dev_steps * 2,
        micro_steps=base.micro_steps * 2,
        life_steps=base.life_steps * 2,
        epi_steps=base.epi_steps * 2,
        eco_steps=base.eco_steps * 2,
        bg_steps=base.bg_steps * 2,
    )
    base_data = simulate_pilot(base)
    fine_data = simulate_pilot(fine)
    rows = []
    for key in RETAINED_KEYS:
        diff = np.asarray(base_data[key]) - np.asarray(fine_data[key])
        rows.append(
            {
                "level": key,
                "mean_absolute_difference": float(np.mean(np.abs(diff))),
                "max_absolute_difference": float(np.max(np.abs(diff))),
                "passes": bool(np.mean(np.abs(diff)) < 0.015 and np.max(np.abs(diff)) < 0.12),
                "note": "deterministic drift check with identical founder seed and process noise disabled",
            }
        )
    return pd.DataFrame(rows)


def _plot_one_level(data: Mapping[str, object], config: PilotConfig, level: str, out: Path, *, variability: bool = False) -> None:
    arr = np.asarray(data[level], dtype=float)
    components = COMPONENTS[level]
    start, stop = config.plot_generation_start, config.plot_generation_stop
    taus = np.arange(start, stop + 1)
    m_l = arr.shape[2]
    x_positions = []
    for tau in taus:
        for t_l in range(m_l):
            x_positions.append(tau + t_l / max(m_l - 1, 1))
    x_positions = np.array(x_positions)
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.45), gridspec_kw={"width_ratios": [1.38, 1.0]})
    if variability:
        lineages = [d - 1 for d in config.variability_lineages_d]
        for j, comp in enumerate(components):
            for d in lineages:
                values = arr[d, start : stop + 1].reshape(-1, arr.shape[-1])[:, j]
                axes[0].plot(x_positions, values, color=colors[j % len(colors)], alpha=0.36, lw=0.9)
        axes[0].set_title(f"{level.replace('_', ' ')}: predetermined lineages d=2,3,4,5", fontsize=10.5)
    else:
        d = config.illustrative_lineage_d - 1
        for j, comp in enumerate(components):
            values = arr[d, start : stop + 1].reshape(-1, arr.shape[-1])[:, j]
            axes[0].plot(x_positions, values, color=colors[j % len(colors)], lw=1.15, label=comp.replace("_", " "))
        axes[0].set_title(f"{level.replace('_', ' ')}: illustrative simulated lineage d=1", fontsize=10.5)
    for tau in taus:
        axes[0].axvline(tau, color="0.82", lw=0.6, zorder=0)
    axes[0].set_xlabel(r"generation $\tau$ and retained within-generation time $t_l$", fontsize=9.0)
    if level == "microbiome":
        ylabel = "abundance"
    elif level == "epigenetic":
        ylabel = "latent logit"
    elif level == "epigenetic_probability":
        ylabel = "mark probability"
    else:
        ylabel = "state value"
    axes[0].set_ylabel(ylabel)
    axes[0].grid(alpha=0.16, lw=0.5)

    tau_mid = config.plot_within_generation_tau
    t = np.arange(m_l)
    if variability:
        for j, comp in enumerate(components):
            for d in lineages:
                axes[1].plot(t, arr[d, tau_mid, :, j], color=colors[j % len(colors)], alpha=0.36, lw=0.9)
        axes[1].set_title(rf"within generation $\tau={tau_mid}$", fontsize=10.5)
    else:
        d = config.illustrative_lineage_d - 1
        for j, comp in enumerate(components):
            axes[1].plot(t, arr[d, tau_mid, :, j], marker="o", ms=2.6, color=colors[j % len(colors)], lw=1.15, label=comp.replace("_", " "))
        axes[1].legend(frameon=False, bbox_to_anchor=(1.04, 0.5), loc="center left", fontsize=7.2)
        axes[1].set_title(rf"within generation $\tau={tau_mid}$", fontsize=10.5)
    axes[1].set_xlabel(r"retained within-generation time $t_l$", fontsize=9.0)
    axes[1].set_ylabel(ylabel)
    axes[1].grid(alpha=0.16, lw=0.5)
    for ax in axes:
        ax.tick_params(labelsize=8.4)
        ax.yaxis.label.set_size(9.0)
    fig.subplots_adjust(left=0.075, right=0.70, bottom=0.22, top=0.84, wspace=0.36)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=320, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def write_pilot_outputs(data: Mapping[str, object], config: PilotConfig, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    pd.Series(asdict(config)).to_json(output_dir / "pilot_config.json", indent=2)
    pd.DataFrame(
        [
            {
                "threshold": "theta_M",
                "value": config.theta_maturation,
                "variable": "maturation_progress",
                "retained_t_l": f"first full-series time with q_mat >= theta_M; early if u < {config.early_maturation_u}",
                "permitted_pilot_range": "[0.55, 0.65]",
            },
            {
                "threshold": "theta_G",
                "value": config.theta_growth,
                "variable": "growth_capacity",
                "retained_t_l": config.T_life - 1,
                "permitted_pilot_range": "[0.60, 0.70]",
            },
        ]
    ).to_csv(output_dir / "phenotype_thresholds.csv", index=False)
    nonzero_cross_level_edges().to_csv(output_dir / "nonzero_cross_level_edges.csv", index=False)
    (output_dir / "corrected_intergenerational_equations.md").write_text(intergenerational_equations_text())
    counts = phenotype_count_table(data, config)
    counts.to_csv(output_dir / "phenotype_counts.csv", index=False)
    acceptance = acceptance_criteria(data, config)
    acceptance.to_csv(output_dir / "qualitative_acceptance_criteria.csv", index=False)
    convergence = convergence_check(config)
    convergence.to_csv(output_dir / "integration_step_convergence_check.csv", index=False)
    for level in RETAINED_KEYS:
        _plot_one_level(data, config, level, figure_dir / f"pilot_{level}_dynamics.png")
        _plot_one_level(data, config, level, figure_dir / f"pilot_{level}_variability_d2_to_d5.png", variability=True)
    probability_data = dict(data)
    probability_data["epigenetic_probability"] = sigmoid(np.asarray(data["epigenetic"], dtype=float))
    _plot_one_level(
        probability_data,
        config,
        "epigenetic_probability",
        figure_dir / "pilot_epigenetic_probability_dynamics.png",
    )
    _plot_one_level(
        probability_data,
        config,
        "epigenetic_probability",
        figure_dir / "pilot_epigenetic_probability_variability_d2_to_d5.png",
        variability=True,
    )
    np.savez_compressed(
        output_dir / "pilot_temporal_architecture.npz",
        **{f"segment_{key}": np.asarray(data[key]) for key in RETAINED_KEYS},
        **{f"full_{key}": np.asarray(data["full_time_series"][key]) for key in RETAINED_KEYS},
        **{f"reproductive_{key}": np.asarray(data["reproductive_states"][key]) for key in RETAINED_KEYS},
        **{f"timestamps_{key}": np.asarray(data["timestamps"][key]) for key in RETAINED_KEYS},
        **{f"segment_indices_{key}": np.asarray(data["segment_indices"][key]) for key in RETAINED_KEYS},
        **{f"reproductive_index_{key}": np.asarray(data["reproductive_indices"][key]) for key in RETAINED_KEYS},
    )
    return {"counts": counts, "acceptance": acceptance, "convergence": convergence}


def run_corrected_pilot(output_dir: Path, config: PilotConfig | None = None) -> dict[str, object]:
    config = config or PilotConfig()
    data = simulate_pilot(config)
    for key in RETAINED_KEYS:
        arr = np.asarray(data[key])
        if not np.all(np.isfinite(arr)):
            raise FloatingPointError(f"{key} contains non-finite values")
        if key == "microbiome" and float(arr.min()) <= 0.0:
            raise FloatingPointError("microbiome abundance reached a nonpositive value")
    summaries = write_pilot_outputs(data, config, output_dir)
    return {"data": data, **summaries}
