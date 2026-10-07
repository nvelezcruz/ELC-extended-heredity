from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

import numpy as np
import pandas as pd

from run_final_c8_untouched_audit import _flatten_parameters, _phenotype_targets_pass
from src.corrected_numerical_audit import (
    CorrectedAuditConfig,
    run_categorical_phenotype_information,
    run_corrected_numerical_audit,
)
from src.final_draft32_completeness_audit import run_final_draft32_completeness_audit
from src.final_numerical_audit import (
    _params_for_config,
    build_analysis_table,
    phenotype_support,
    save_seed_archive,
    simulate_final_seed,
)
from src.recalibration_audit import calibration_candidates


FINAL_SEEDS = (254989981, 265358979, 276563425, 287271829)
CALIBRATION_SEEDS = (101, 202, 303, 404)
CANDIDATE = "C13_robust_complete_graph_recovery"


def main() -> None:
    root = Path(__file__).resolve().parent
    output_root = root / "outputs" / "c13_final_untouched"
    source_dir = output_root / "source_data"
    output_root.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    if set(FINAL_SEEDS) & set(CALIBRATION_SEEDS):
        raise AssertionError("final evaluation seeds overlap calibration seeds")
    seed_status = pd.DataFrame(
        [
            {
                "seed": seed,
                "used_in_primary_calibration": seed in CALIBRATION_SEEDS,
                "prior_project_occurrences_before_final_assignment": 0,
                "status": "untouched_before_final_evaluation",
            }
            for seed in FINAL_SEEDS
        ]
    )
    seed_status.to_csv(output_root / "untouched_final_seed_verification.csv", index=False)

    config = CorrectedAuditConfig(
        seeds=FINAL_SEEDS,
        parameter_overrides=calibration_candidates()[CANDIDATE],
    )
    (output_root / "final_configuration.json").write_text(
        json.dumps(asdict(config), indent=2)
    )
    (output_root / "superseded_c12_untouched_run.json").write_text(
        json.dumps(
            {
                "path": "outputs/c12_final_untouched",
                "status": "failed_untouched_graph_gate_preserved_not_used_for_tuning",
                "missed_encoded_dependency": "development_to_microbiome",
                "randomized_source_p_value": 0.017964071856287425,
                "required_alpha": 0.01,
            },
            indent=2,
        )
    )

    base = _flatten_parameters(
        _params_for_config(CorrectedAuditConfig(seeds=FINAL_SEEDS))
    ).rename(columns={"value": "previous_value"})
    selected = _flatten_parameters(_params_for_config(config)).rename(
        columns={"value": "fixed_c13_value"}
    )
    comparison = base.merge(selected, on=["parameter", "index"], how="outer")
    comparison["changed"] = ~np.isclose(
        comparison["previous_value"], comparison["fixed_c13_value"], equal_nan=True
    )
    selected.to_csv(source_dir / "fixed_c13_parameter_set.csv", index=False)
    comparison.to_csv(source_dir / "fixed_c13_parameter_comparison.csv", index=False)

    seed_results: list[dict[str, object]] = []
    for seed in FINAL_SEEDS:
        print(f"simulating untouched C13 final seed {seed}", flush=True)
        data = simulate_final_seed(seed, config, multiparent=False)
        save_seed_archive(
            source_dir / f"corrected_temporal_architecture_seed_{seed}.npz", data
        )
        seed_results.append(data)

    table = build_analysis_table(seed_results, config)
    support = phenotype_support(table)
    support.to_csv(output_root / "phenotype_support_by_seed_precheck.csv", index=False)
    categorical_summary, _, categorical_by_seed, _ = run_categorical_phenotype_information(
        table, config, output_root
    )
    ok, checks = _phenotype_targets_pass(
        categorical_summary, categorical_by_seed, support
    )
    checks.to_csv(output_root / "final_phenotype_acceptance_checks.csv", index=False)
    print(checks.to_string(index=False), flush=True)
    if not ok:
        raise AssertionError(
            "untouched C13 phenotype gate failed; these seeds cannot be used for calibration"
        )

    corrected = run_corrected_numerical_audit(
        output_dir=output_root / "corrected_numerical_audit",
        source_archive=source_dir,
        config=config,
    )
    completeness = run_final_draft32_completeness_audit(
        output_root=output_root / "draft32_completeness_audit",
        corrected_archive=Path(corrected["data_dir"]),
        frozen_source_archive=source_dir,
        config=config,
    )
    print(completeness["complex_unit_qualification"].to_string(index=False), flush=True)
    print(f"outputs={output_root}", flush=True)


if __name__ == "__main__":
    main()
