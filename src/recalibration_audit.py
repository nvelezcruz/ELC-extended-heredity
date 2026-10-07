from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import platform
from time import perf_counter
from typing import Mapping

import numpy as np
import pandas as pd

from .corrected_numerical_audit import (
    CorrectedAuditConfig,
    choose_common_jitter,
    continuous_families,
    covariance_diagnostics,
    run_categorical_phenotype_information,
    run_continuous_transfer_entropy,
    run_location_corrected,
    run_pid_corrected,
    central_contrast_from_summary,
)
from .corrected_pilot_model import (
    _params,
    apply_parameter_overrides,
    acceptance_criteria,
    convergence_check,
    sigmoid,
)
from .final_numerical_audit import (
    BACKGROUND_LEVEL,
    ELC_LEVELS,
    FinalAuditConfig,
    PHENOTYPE_LABELS,
    _params_for_config,
    build_analysis_table,
    phenotype_support,
    simulate_final_seed,
)


def calibration_candidates() -> dict[str, tuple[tuple[str, float], ...]]:
    return {
        "baseline": (),
        "C1_epi_variation": (
            ("epi_founder_sd", 0.55),
            ("sigma_epi_reg", 0.19),
            ("sigma_epi_stress", 0.19),
            ("epi_transmission_noise_sd", 0.13),
            ("rho_epi_reg", 0.68),
            ("rho_epi_stress", 0.65),
        ),
        "C2_epi_paths": (
            ("epi_founder_sd", 0.55),
            ("sigma_epi_reg", 0.19),
            ("sigma_epi_stress", 0.19),
            ("epi_transmission_noise_sd", 0.13),
            ("rho_epi_reg", 0.68),
            ("rho_epi_stress", 0.65),
            ("epi_effect_development_scale", 1.45),
            ("epi_effect_life_history_scale", 1.65),
            ("epi_development_start_scale", 1.45),
            ("epi_life_start_scale", 1.65),
            ("epi_effect_microbiome_scale", 1.35),
            ("epi_microbiome_start_scale", 1.35),
        ),
        "C3_epi_paths_eco_balance": (
            ("epi_founder_sd", 0.58),
            ("sigma_epi_reg", 0.20),
            ("sigma_epi_stress", 0.20),
            ("epi_transmission_noise_sd", 0.14),
            ("rho_epi_reg", 0.70),
            ("rho_epi_stress", 0.66),
            ("epi_effect_development_scale", 1.70),
            ("epi_effect_life_history_scale", 2.10),
            ("epi_development_start_scale", 1.70),
            ("epi_life_start_scale", 2.10),
            ("epi_effect_microbiome_scale", 1.60),
            ("epi_microbiome_start_scale", 1.60),
            ("ecology_effect_life_history_scale", 0.82),
            ("ecology_life_start_scale", 0.86),
        ),
        "C4_strong_epi_sparse": (
            ("epi_founder_sd", 0.60),
            ("sigma_epi_reg", 0.21),
            ("sigma_epi_stress", 0.21),
            ("epi_transmission_noise_sd", 0.15),
            ("rho_epi_reg", 0.72),
            ("rho_epi_stress", 0.68),
            ("epi_effect_development_scale", 2.10),
            ("epi_effect_life_history_scale", 2.80),
            ("epi_development_start_scale", 2.00),
            ("epi_life_start_scale", 2.50),
            ("epi_effect_microbiome_scale", 1.90),
            ("epi_microbiome_start_scale", 1.90),
            ("ecology_effect_life_history_scale", 0.72),
            ("ecology_life_start_scale", 0.78),
        ),
        "C5_growth_weighted_epi": (
            ("epi_founder_sd", 0.62),
            ("sigma_epi_reg", 0.22),
            ("sigma_epi_stress", 0.21),
            ("epi_transmission_noise_sd", 0.15),
            ("rho_epi_reg", 0.72),
            ("rho_epi_stress", 0.68),
            ("epi_effect_development_scale", 2.20),
            ("epi_development_start_scale", 2.10),
            ("epi_effect_microbiome_scale", 2.00),
            ("epi_microbiome_start_scale", 2.00),
            ("epi_effect_life_timing_scale", 1.45),
            ("epi_effect_life_maturation_scale", 1.45),
            ("epi_effect_life_growth_scale", 4.20),
            ("epi_effect_life_allocation_scale", 2.10),
            ("epi_life_start_maturation_scale", 1.25),
            ("epi_life_start_growth_scale", 4.00),
            ("epi_life_start_allocation_scale", 2.00),
            ("ecology_effect_life_history_scale", 0.82),
            ("ecology_life_start_scale", 0.86),
        ),
        "C6_growth_weighted_no_eco_reduction": (
            ("epi_founder_sd", 0.62),
            ("sigma_epi_reg", 0.22),
            ("sigma_epi_stress", 0.21),
            ("epi_transmission_noise_sd", 0.15),
            ("rho_epi_reg", 0.72),
            ("rho_epi_stress", 0.68),
            ("epi_effect_development_scale", 2.20),
            ("epi_development_start_scale", 2.10),
            ("epi_effect_microbiome_scale", 2.00),
            ("epi_microbiome_start_scale", 2.00),
            ("epi_effect_life_timing_scale", 1.35),
            ("epi_effect_life_maturation_scale", 1.35),
            ("epi_effect_life_growth_scale", 4.40),
            ("epi_effect_life_allocation_scale", 2.10),
            ("epi_life_start_maturation_scale", 1.15),
            ("epi_life_start_growth_scale", 4.20),
            ("epi_life_start_allocation_scale", 2.00),
        ),
        "C7_moderate_growth_weighted": (
            ("epi_founder_sd", 0.60),
            ("sigma_epi_reg", 0.21),
            ("sigma_epi_stress", 0.20),
            ("epi_transmission_noise_sd", 0.14),
            ("rho_epi_reg", 0.70),
            ("rho_epi_stress", 0.66),
            ("epi_effect_development_scale", 2.00),
            ("epi_development_start_scale", 1.90),
            ("epi_effect_microbiome_scale", 1.80),
            ("epi_microbiome_start_scale", 1.80),
            ("epi_effect_life_timing_scale", 1.25),
            ("epi_effect_life_maturation_scale", 1.25),
            ("epi_effect_life_growth_scale", 3.70),
            ("epi_effect_life_allocation_scale", 1.90),
            ("epi_life_start_maturation_scale", 1.10),
            ("epi_life_start_growth_scale", 3.60),
            ("epi_life_start_allocation_scale", 1.80),
            ("ecology_effect_life_history_scale", 0.90),
            ("ecology_life_start_scale", 0.92),
        ),
        "C8_stronger_growth_allocation": (
            ("epi_founder_sd", 0.68),
            ("sigma_epi_reg", 0.24),
            ("sigma_epi_stress", 0.24),
            ("epi_transmission_noise_sd", 0.16),
            ("rho_epi_reg", 0.72),
            ("rho_epi_stress", 0.70),
            ("epi_effect_development_scale", 2.80),
            ("epi_development_start_scale", 3.00),
            ("epi_effect_microbiome_scale", 2.50),
            ("epi_microbiome_start_scale", 2.40),
            ("epi_effect_life_timing_scale", 1.35),
            ("epi_effect_life_maturation_scale", 1.35),
            ("epi_effect_life_growth_scale", 5.20),
            ("epi_effect_life_allocation_scale", 3.50),
            ("epi_life_start_maturation_scale", 1.15),
            ("epi_life_start_growth_scale", 5.00),
            ("epi_life_start_allocation_scale", 3.40),
            ("ecology_effect_life_history_scale", 0.82),
            ("ecology_life_start_scale", 0.86),
        ),
        "C9_stronger_no_eco_balance": (
            ("epi_founder_sd", 0.68),
            ("sigma_epi_reg", 0.24),
            ("sigma_epi_stress", 0.24),
            ("epi_transmission_noise_sd", 0.16),
            ("rho_epi_reg", 0.72),
            ("rho_epi_stress", 0.70),
            ("epi_effect_development_scale", 2.80),
            ("epi_development_start_scale", 3.00),
            ("epi_effect_microbiome_scale", 2.50),
            ("epi_microbiome_start_scale", 2.40),
            ("epi_effect_life_timing_scale", 1.25),
            ("epi_effect_life_maturation_scale", 1.25),
            ("epi_effect_life_growth_scale", 5.40),
            ("epi_effect_life_allocation_scale", 3.50),
            ("epi_life_start_maturation_scale", 1.10),
            ("epi_life_start_growth_scale", 5.20),
            ("epi_life_start_allocation_scale", 3.40),
        ),
        "C10_growth_allocation_support": (
            ("epi_founder_sd", 0.66),
            ("sigma_epi_reg", 0.23),
            ("sigma_epi_stress", 0.23),
            ("epi_transmission_noise_sd", 0.16),
            ("rho_epi_reg", 0.72),
            ("rho_epi_stress", 0.69),
            ("epi_effect_development_scale", 2.60),
            ("epi_development_start_scale", 2.80),
            ("epi_effect_microbiome_scale", 2.40),
            ("epi_microbiome_start_scale", 2.30),
            ("epi_effect_life_timing_scale", 1.20),
            ("epi_effect_life_maturation_scale", 1.20),
            ("epi_effect_life_growth_scale", 5.00),
            ("epi_effect_life_allocation_scale", 4.20),
            ("epi_life_start_maturation_scale", 1.00),
            ("epi_life_start_growth_scale", 4.80),
            ("epi_life_start_allocation_scale", 4.00),
            ("ecology_effect_life_history_scale", 0.88),
            ("ecology_life_start_scale", 0.90),
        ),
        "C11_pre_reproductive_phenotype": (
            ("epi_founder_sd", 0.68),
            ("sigma_epi_reg", 0.24),
            ("sigma_epi_stress", 0.24),
            ("epi_transmission_noise_sd", 0.16),
            ("rho_epi_reg", 0.72),
            ("rho_epi_stress", 0.70),
            ("epi_effect_development_scale", 2.80),
            ("epi_development_start_scale", 3.00),
            ("epi_effect_microbiome_scale", 2.50),
            ("epi_microbiome_start_scale", 2.40),
            ("epi_effect_life_timing_scale", 4.50),
            ("epi_effect_life_maturation_scale", 3.00),
            ("epi_effect_life_growth_scale", 7.00),
            ("epi_effect_life_allocation_scale", 3.50),
            ("epi_life_start_maturation_scale", 1.15),
            ("epi_life_start_growth_scale", 5.00),
            ("epi_life_start_allocation_scale", 3.40),
            ("life_maturation_timing_base", 0.25),
            ("ecology_effect_life_history_scale", 0.82),
            ("ecology_life_start_scale", 0.86),
        ),
        "C12_complete_graph_recovery": (
            ("epi_founder_sd", 0.68),
            ("sigma_epi_reg", 0.24),
            ("sigma_epi_stress", 0.24),
            ("epi_transmission_noise_sd", 0.16),
            ("rho_epi_reg", 0.72),
            ("rho_epi_stress", 0.70),
            ("epi_effect_development_scale", 2.80),
            ("epi_development_start_scale", 3.00),
            ("development_effect_microbiome_scale", 7.00),
            ("epi_effect_microbiome_scale", 2.50),
            ("epi_microbiome_start_scale", 2.40),
            ("epi_effect_life_timing_scale", 4.50),
            ("epi_effect_life_maturation_scale", 3.00),
            ("epi_effect_life_growth_scale", 7.00),
            ("epi_effect_life_allocation_scale", 3.50),
            ("epi_life_start_maturation_scale", 1.15),
            ("epi_life_start_growth_scale", 5.00),
            ("epi_life_start_allocation_scale", 3.40),
            ("life_maturation_timing_base", 0.25),
            ("ecology_effect_life_history_scale", 0.82),
            ("ecology_life_start_scale", 0.86),
        ),
        "C13_robust_complete_graph_recovery": (
            ("epi_founder_sd", 0.68),
            ("sigma_epi_reg", 0.24),
            ("sigma_epi_stress", 0.24),
            ("epi_transmission_noise_sd", 0.16),
            ("rho_epi_reg", 0.72),
            ("rho_epi_stress", 0.70),
            ("epi_effect_development_scale", 2.80),
            ("epi_development_start_scale", 3.00),
            ("development_effect_microbiome_scale", 30.00),
            ("epi_effect_microbiome_scale", 2.50),
            ("epi_microbiome_start_scale", 2.40),
            ("epi_effect_life_timing_scale", 4.50),
            ("epi_effect_life_maturation_scale", 3.00),
            ("epi_effect_life_growth_scale", 7.00),
            ("epi_effect_life_allocation_scale", 3.50),
            ("epi_life_start_maturation_scale", 1.15),
            ("epi_life_start_growth_scale", 5.00),
            ("epi_life_start_allocation_scale", 3.40),
            ("life_maturation_timing_base", 0.25),
            ("ecology_effect_life_history_scale", 0.82),
            ("ecology_life_start_scale", 0.86),
        ),
    }


def parameter_change_table(candidate: str, overrides: tuple[tuple[str, float], ...]) -> pd.DataFrame:
    base = _params()
    changed = apply_parameter_overrides(_params(), overrides)
    rows = []
    reason = {
        "epi_founder_sd": "increase between-lineage epigenetic variation while retaining bounded probability-scale marks",
        "sigma_epi_reg": "increase within-generation variation of the regulatory epigenetic logit",
        "sigma_epi_stress": "increase within-generation variation of the stress-memory epigenetic logit",
        "epi_transmission_noise_sd": "increase between-generation stochastic variation in inherited epigenetic logits",
        "rho_epi_reg": "preserve partial intergenerational persistence of the regulatory mark",
        "rho_epi_stress": "preserve partial intergenerational persistence of the stress-memory mark",
        "epi_effect_development_scale": "strengthen existing epigenetic effects on developmental regulatory variables",
        "development_effect_microbiome_scale": "strengthen the existing host-development effects on microbial guild growth without adding a pathway",
        "epi_development_start_scale": "strengthen existing epigenetic effects in developmental reconstruction",
        "epi_effect_microbiome_scale": "strengthen existing indirect epigenetic effects on microbial guild dynamics",
        "epi_microbiome_start_scale": "strengthen existing epigenetic effects in microbiome carryover",
        "epi_effect_life_timing_scale": "strengthen the existing regulatory effect on maturation timing without changing thresholds",
        "epi_effect_life_maturation_scale": "strengthen the existing regulatory effect on maturation progress",
        "epi_effect_life_growth_scale": "strengthen the existing regulatory effect on growth capacity",
        "epi_effect_life_allocation_scale": "strengthen the existing stress-memory effect on reproductive allocation",
        "epi_life_start_maturation_scale": "strengthen existing epigenetic effect on initial maturation progress",
        "epi_life_start_growth_scale": "strengthen existing epigenetic effect on initial growth capacity",
        "epi_life_start_allocation_scale": "strengthen existing epigenetic effect on initial reproductive allocation",
        "life_maturation_timing_base": "align the existing maturation wave with the fixed pre-reproductive early-maturation boundary",
        "life_maturation_target_intercept": "adjust the baseline of the existing maturation target without changing the phenotype threshold",
        "life_maturation_wave_scale": "adjust the amplitude of the existing maturation wave without adding a new pathway",
        "life_start_maturation_intercept": "adjust the initial maturation state while preserving the existing reconstruction equation",
        "ecology_effect_life_history_scale": "modestly reduce direct ecological effects on life-history state so they do not overwhelm epigenetic variation",
        "ecology_life_start_scale": "modestly reduce ecological effects in life-history reconstruction while preserving ecological contribution",
    }
    for key, value in overrides:
        if key in {"sigma_epi_reg", "sigma_epi_stress", "rho_epi_reg", "rho_epi_stress"}:
            old_value = np.nan
            if key.startswith("sigma_epi"):
                old_value = float(np.asarray(base["sigma_epi"])[0 if key.endswith("reg") else 1])
            if key.startswith("rho_epi"):
                old_value = float(np.asarray(base["rho_epi"])[0 if key.endswith("reg") else 1])
        else:
            old_value = float(base.get(key, 1.0))
        rows.append(
            {
                "candidate": candidate,
                "parameter": key,
                "old_value": old_value,
                "new_value": float(value),
                "biological_reason": reason.get(key, "calibration parameter in an existing transition term"),
                "changed_parameter_value_in_selected_model": float(changed[key][0] if isinstance(changed.get(key), np.ndarray) and np.asarray(changed[key]).ndim else changed.get(key, value)) if key in changed else float(value),
            }
        )
    return pd.DataFrame(rows)


def epigenetic_probability_range(seed_results: list[Mapping[str, object]]) -> pd.DataFrame:
    rows = []
    for data in seed_results:
        seed = int(data["seed"])
        probs = sigmoid(np.asarray(data["epigenetic"], dtype=float))
        for i, component in enumerate(("regulatory_mark_probability", "stress_memory_probability")):
            vals = probs[..., i].reshape(-1)
            rows.append(
                {
                    "seed": seed,
                    "component": component,
                    "minimum": float(vals.min()),
                    "q01": float(np.quantile(vals, 0.01)),
                    "median": float(np.median(vals)),
                    "q99": float(np.quantile(vals, 0.99)),
                    "maximum": float(vals.max()),
                }
            )
    return pd.DataFrame(rows)


def selected_target_checks(
    categorical_summary: pd.DataFrame,
    categorical_seed: pd.DataFrame,
    phenotype: pd.DataFrame,
    continuous_summary: pd.DataFrame,
    location: pd.DataFrame,
    pid_summary: pd.DataFrame,
    pid_seed: pd.DataFrame,
) -> pd.DataFrame:
    cat = categorical_summary.set_index("quantity")
    phen_all = phenotype[phenotype["seed"].astype(str) == "all"].set_index("phenotype_state")
    cont = continuous_summary.set_index("analysis")
    loc = location.copy()
    loc_max = loc.groupby("target_level")["null_excess_bits"].max().to_dict()
    rows = [
        ("complete_four_state_information_ge_0.020", float(cat.loc["complete_phenotype_transfer_entropy", "estimate_bits"]), ">=0.020", float(cat.loc["complete_phenotype_transfer_entropy", "estimate_bits"]) >= 0.020),
        ("binary_nu_information_ge_0.010", float(cat.loc["binary_nu_transfer_entropy", "estimate_bits"]), ">=0.010", float(cat.loc["binary_nu_transfer_entropy", "estimate_bits"]) >= 0.010),
        ("focal_nu_log_ratio_positive_every_seed", float(categorical_seed["focal_nu_log_ratio_given_nu_bits"].min()), ">0 in every seed", bool((categorical_seed["focal_nu_log_ratio_given_nu_bits"] > 0).all())),
        ("complete_information_ge_binary_nu_information", float(cat.loc["complete_phenotype_transfer_entropy", "estimate_bits"] - cat.loc["binary_nu_transfer_entropy", "estimate_bits"]), ">=0", float(cat.loc["complete_phenotype_transfer_entropy", "estimate_bits"]) + 1e-12 >= float(cat.loc["binary_nu_transfer_entropy", "estimate_bits"])),
        ("minimum_phenotype_state_probability_ge_0.05", float(phen_all["proportion"].min()), ">=0.05", float(phen_all["proportion"].min()) >= 0.05),
        ("maximum_phenotype_state_probability_le_0.70", float(phen_all["proportion"].max()), "<=0.70", float(phen_all["proportion"].max()) <= 0.70),
        ("epigenetic_predictive_contribution_above_null", float(cont.loc["epigenetic_predictive_contribution", "null_excess_bits"]), ">0 and p<=0.05", bool(cont.loc["epigenetic_predictive_contribution", "null_excess_bits"] > 0 and cont.loc["epigenetic_predictive_contribution", "generation_preserving_surrogate_p_value"] <= 0.05)),
        ("epigenetic_predictive_closure_above_null", float(cont.loc["epigenetic_predictive_closure", "null_excess_bits"]), ">0 and p<=0.05", bool(cont.loc["epigenetic_predictive_closure", "null_excess_bits"] > 0 and cont.loc["epigenetic_predictive_closure", "generation_preserving_surrogate_p_value"] <= 0.05)),
        ("background_predictive_contribution_above_null", float(cont.loc["background_predictive_contribution", "null_excess_bits"]), ">0 and p<=0.05", bool(cont.loc["background_predictive_contribution", "null_excess_bits"] > 0 and cont.loc["background_predictive_contribution", "generation_preserving_surrogate_p_value"] <= 0.05)),
        ("background_closure_not_above_null", float(cont.loc["background_closure_control", "null_excess_bits"]), "not >0 with p<=0.05", not bool(cont.loc["background_closure_control", "null_excess_bits"] > 0 and cont.loc["background_closure_control", "generation_preserving_surrogate_p_value"] <= 0.05)),
        ("development_location_positive_ge_0.010_some_time", float(loc_max.get("development", np.nan)), ">=0.010 at one retained time", float(loc_max.get("development", -np.inf)) >= 0.010),
        ("life_history_location_positive_ge_0.015_some_time", float(loc_max.get("life_history", np.nan)), ">=0.015 at one retained time", float(loc_max.get("life_history", -np.inf)) >= 0.015),
        ("microbiome_location_positive_ge_0.002_some_time", float(loc_max.get("microbiome", np.nan)), ">=0.002 at one retained time", float(loc_max.get("microbiome", -np.inf)) >= 0.002),
        ("pid_synergy_positive_pooled", float(pid_summary.loc[0, "synergy_bits"]), ">0", float(pid_summary.loc[0, "synergy_bits"]) > 0),
        ("pid_synergy_positive_every_seed", float(pid_seed["synergy_bits"].min()), ">0 in every seed", bool((pid_seed["synergy_bits"] > 0).all())),
    ]
    return pd.DataFrame(
        [
            {"criterion": criterion, "observed_value": value, "target": target, "passes": bool(passes)}
            for criterion, value, target, passes in rows
        ]
    )


def run_selected_recalibration_audit(
    output_dir: Path,
    *,
    selected_candidate: str = "C8_stronger_growth_allocation",
    config: CorrectedAuditConfig | None = None,
) -> dict[str, object]:
    t0 = perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    candidates = calibration_candidates()
    if selected_candidate not in candidates:
        raise KeyError(selected_candidate)
    base_config = config or CorrectedAuditConfig()
    cfg = CorrectedAuditConfig(**{**asdict(base_config), "parameter_overrides": candidates[selected_candidate]})
    (data_dir / "recalibration_configuration.json").write_text(json.dumps(asdict(cfg), indent=2))
    setting_rows = []
    for name, overrides in candidates.items():
        setting_rows.append({"candidate": name, "n_changed_parameters": len(overrides), "parameter_overrides": json.dumps(dict(overrides), sort_keys=True)})
    pd.DataFrame(setting_rows).to_csv(data_dir / "candidate_parameter_settings.csv", index=False)
    parameter_change_table(selected_candidate, candidates[selected_candidate]).to_csv(data_dir / "selected_parameter_changes.csv", index=False)
    pd.DataFrame(
        [
            {
                "selected_candidate": selected_candidate,
                "new_untouched_final_seeds_proposed": "1207;2411;3613;4817",
                "calibration_seeds": ";".join(map(str, cfg.seeds)),
                "status": "selected_for_user_approval_before_new_final_run",
            }
        ]
    ).to_csv(data_dir / "selected_setting_and_proposed_final_seeds.csv", index=False)
    seed_results: list[Mapping[str, object]] = []
    acceptance_rows = []
    for seed in cfg.seeds:
        print(f"selected recalibration simulation seed {seed}", flush=True)
        data = simulate_final_seed(seed, cfg, multiparent=False)
        seed_results.append(data)
        np.savez_compressed(
            data_dir / f"calibration_retained_segments_{selected_candidate}_seed_{seed}.npz",
            **{level: np.asarray(data[level]) for level in ELC_LEVELS + (BACKGROUND_LEVEL,)},
        )
        acc = acceptance_criteria(data, data["config"])
        acc.insert(0, "seed", seed)
        acceptance_rows.append(acc)
    pd.concat(acceptance_rows, ignore_index=True).to_csv(data_dir / "calibration_qualitative_acceptance_criteria_by_seed.csv", index=False)
    convergence_check(cfg.pilot_config(cfg.seeds[0])).to_csv(data_dir / "calibration_integration_step_convergence_confirmation.csv", index=False)
    epigenetic_probability_range(seed_results).to_csv(data_dir / "calibration_epigenetic_probability_range_by_seed.csv", index=False)
    table = build_analysis_table(seed_results, cfg)
    table["seed_results"] = seed_results
    table["complex_source_epigenetic_ecological"] = np.hstack([table["source_epigenetic"], table["source_ecological"]])
    seed_tables = [build_analysis_table([sd], cfg) for sd in seed_results]
    for seed_table in seed_tables:
        seed_table["complex_source_epigenetic_ecological"] = np.hstack([seed_table["source_epigenetic"], seed_table["source_ecological"]])
    phenotype = phenotype_support(table)
    phenotype.to_csv(data_dir / "calibration_phenotype_support_by_seed.csv", index=False)
    covariance_diagnostics(table, data_dir)
    families = continuous_families()
    selected_jitter = choose_common_jitter(table, families, cfg, data_dir)
    (data_dir / "calibration_selected_common_covariance_jitter.txt").write_text(f"{selected_jitter:.12g}\n")
    continuous_summary, continuous_boot, continuous_null, continuous_seed = run_continuous_transfer_entropy(table, seed_tables, cfg, data_dir, jitter=selected_jitter)
    central = central_contrast_from_summary(continuous_summary, data_dir)
    categorical_summary, categorical_boot, categorical_seed, categorical_decomp = run_categorical_phenotype_information(table, cfg, data_dir)
    location_summary, location_boot, location_null = run_location_corrected(table, cfg, data_dir, jitter=selected_jitter)
    pid_summary, pid_boot, pid_seed, pid_sensitivity = run_pid_corrected(
        table,
        seed_tables,
        cfg,
        data_dir,
        prefix="epigenetic_ecological",
        jitter=selected_jitter,
    )
    checks = selected_target_checks(
        categorical_summary,
        categorical_seed,
        phenotype,
        continuous_summary,
        location_summary,
        pid_summary,
        pid_seed,
    )
    checks.to_csv(data_dir / "selected_calibration_target_checks.csv", index=False)
    runtime = {
        "runtime_seconds": perf_counter() - t0,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "biological_dynamics_status": "calibration_run_not_final_worked_example",
        "selected_candidate": selected_candidate,
    }
    (data_dir / "calibration_software_versions_and_runtime.json").write_text(json.dumps(runtime, indent=2))
    return {
        "data_dir": data_dir,
        "selected_candidate": selected_candidate,
        "phenotype": phenotype,
        "continuous_summary": continuous_summary,
        "categorical_summary": categorical_summary,
        "location_summary": location_summary,
        "pid_summary": pid_summary,
        "pid_seed": pid_seed,
        "checks": checks,
        "runtime": runtime,
    }
