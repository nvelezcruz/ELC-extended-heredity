from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import gc
import json

import pandas as pd

from src.corrected_numerical_audit import CorrectedAuditConfig
from src.inter_elc_arm import (
    CORRECTED_INTER_ELC_FINAL_SEEDS_V2,
    INTER_ELC_CALIBRATION_SEEDS,
    INTER_ELC_FINAL_SEEDS,
    _evaluate_inter_elc_candidate,
    estimate_source_state_means_sequential,
    inter_elc_b_matrix_inventory,
    inter_elc_level_summary,
    run_inter_elc_final_audit,
    scaled_epigenetic_B,
    source_state_means_table,
)
from src.recalibration_audit import calibration_candidates
from run_c12_final_untouched import FINAL_SEEDS as C12_PRIMARY_FINAL_SEEDS


CALIBRATION_ROOT = Path("outputs/c12_inter_elc_calibration")
FINAL_ROOT = Path("outputs/c12_inter_elc_final_untouched")
FINAL_SEEDS = (212345789, 223456789, 234567899, 245678911)
CANDIDATE_SCALES = (0.35, 0.50, 0.65)
CALIBRATION_BUFFER_BITS = 0.012
CANDIDATE = "C12_complete_graph_recovery"


def main() -> None:
    t0 = perf_counter()
    data_dir = CALIBRATION_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    overrides = calibration_candidates()[CANDIDATE]
    config = CorrectedAuditConfig(
        seeds=INTER_ELC_CALIBRATION_SEEDS,
        parameter_overrides=overrides,
    )
    (data_dir / "calibration_configuration.json").write_text(
        json.dumps(asdict(config), indent=2)
    )

    source_means = estimate_source_state_means_sequential(config)
    source_state_means_table(source_means).to_csv(
        data_dir / "fixed_source_centering_means.csv", index=False
    )

    candidate_rows: list[dict[str, object]] = []
    selected = None
    candidate_prefix = CANDIDATE.split("_", 1)[0].lower()
    for index, scale in enumerate(CANDIDATE_SCALES, start=1):
        label = f"{candidate_prefix}_epigenetic_{int(round(scale * 100)):03d}pct"
        b = scaled_epigenetic_B(scale, b_micro=0.0, b_eco=0.0)
        evaluation = _evaluate_inter_elc_candidate(
            label=label,
            stage=f"{candidate_prefix}_epigenetic_only",
            epigenetic_scale=scale,
            b=b,
            config=config,
            source_means=source_means,
            data_dir=data_dir,
            candidate_index=index,
        )
        complete = evaluation["summary"].loc[
            evaluation["summary"]["analysis"] == "whole_source_elc_from_d_prime"
        ].iloc[0]
        passes = bool(evaluation["checks"]["passes"].all())
        buffer_passes = float(complete["null_excess_bits"]) >= CALIBRATION_BUFFER_BITS
        candidate_rows.append(
            {
                "candidate": label,
                "epigenetic_scale": scale,
                "b_epi_reg": b.b_epi_reg,
                "b_epi_stress": b.b_epi_stress,
                "epi_interaction_reg": b.epi_interaction_reg,
                "epi_interaction_stress": b.epi_interaction_stress,
                "whole_source_raw_bits": float(complete["raw_estimate_bits"]),
                "randomized_source_mean_bits": float(complete["surrogate_null_mean_bits"]),
                "information_above_randomized_mean_bits": float(
                    complete["null_excess_bits"]
                ),
                "randomized_source_p_value": float(
                    complete["generation_preserving_surrogate_p_value"]
                ),
                "all_acceptance_checks_pass": passes,
                "calibration_buffer_pass": buffer_passes,
                "selected": passes and buffer_passes,
            }
        )
        if passes and buffer_passes:
            selected = (label, b)
            break
        del evaluation
        gc.collect()

    pd.DataFrame(candidate_rows).to_csv(data_dir / "candidate_summary.csv", index=False)
    if selected is None:
        raise AssertionError(
            "no C12 inter-ELC candidate passed every gate and the calibration buffer"
        )
    label, selected_b = selected
    inter_elc_b_matrix_inventory(selected_b).to_csv(
        data_dir / "selected_B_component_inventory.csv", index=False
    )
    inter_elc_level_summary(selected_b).to_csv(
        data_dir / "selected_B_level_summary.csv", index=False
    )
    (data_dir / "selected_setting.json").write_text(
        json.dumps(
            {
                "candidate": label,
                "parameter_setting": CANDIDATE,
                "calibration_seeds": list(INTER_ELC_CALIBRATION_SEEDS),
                "untouched_final_seeds": list(FINAL_SEEDS),
                **selected_b.__dict__,
            },
            indent=2,
        )
    )

    prior = (
        set(INTER_ELC_CALIBRATION_SEEDS)
        | set(INTER_ELC_FINAL_SEEDS)
        | set(CORRECTED_INTER_ELC_FINAL_SEEDS_V2)
        | set(C12_PRIMARY_FINAL_SEEDS)
    )
    verification = pd.DataFrame(
        [
            {
                "seed": seed,
                "used_in_calibration_or_prior_evaluation": seed in prior,
                "prior_project_occurrences_before_final_assignment": 0,
                "status": "untouched_inter_elc_final_seed",
            }
            for seed in FINAL_SEEDS
        ]
    )
    verification.to_csv(data_dir / "untouched_final_seed_verification.csv", index=False)
    if bool(verification["used_in_calibration_or_prior_evaluation"].any()):
        raise AssertionError("C12 inter-ELC final seed overlaps calibration or evaluation")

    result = run_inter_elc_final_audit(
        FINAL_ROOT,
        seeds=FINAL_SEEDS,
        b=selected_b,
        source_means=source_means,
        prior_seed_sets=(
            INTER_ELC_FINAL_SEEDS,
            CORRECTED_INTER_ELC_FINAL_SEEDS_V2,
            C12_PRIMARY_FINAL_SEEDS,
        ),
        parameter_overrides=overrides,
        calibration_seeds=INTER_ELC_CALIBRATION_SEEDS,
    )
    if not bool(result["passes"]):
        raise AssertionError(
            "untouched C12 inter-ELC run failed; final seeds were not used for tuning"
        )

    (data_dir / "runtime.json").write_text(
        json.dumps(
            {"runtime_seconds": perf_counter() - t0, "status": "passed"}, indent=2
        )
    )
    print(f"selected={label}; outputs={FINAL_ROOT}", flush=True)


if __name__ == "__main__":
    main()
