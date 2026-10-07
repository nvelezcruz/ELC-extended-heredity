from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.corrected_numerical_audit import (
    CmiFamily,
    CorrectedAuditConfig,
    SourceBlock,
    _cov,
    _family_estimates_from_cov,
    _null_family,
    _ranked_joint_arrays,
    _summarize_family,
    _bootstrap_family,
    choose_common_jitter,
    continuous_families,
    run_categorical_phenotype_information,
    run_corrected_numerical_audit,
)
from src.final_draft32_completeness_audit import run_final_draft32_completeness_audit
from src.final_numerical_audit import (
    BACKGROUND_LEVEL,
    ELC_LEVELS,
    _concat_levels,
    _history,
    _params_for_config,
    build_analysis_table,
    load_saved_seed,
    phenotype_support,
    save_seed_archive,
    simulate_final_seed,
)
from src.recalibration_audit import calibration_candidates


UNTOUCHED_SEEDS = (1207, 2411, 3613, 4817)
CALIBRATION_SEEDS = (101, 202, 303, 404)


def _flatten_parameters(par: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, value in sorted(par.items()):
        arr = np.asarray(value)
        if arr.ndim == 0:
            rows.append({"parameter": key, "index": "", "value": float(arr)})
        else:
            for idx in np.ndindex(arr.shape):
                rows.append({"parameter": key, "index": ",".join(str(i) for i in idx), "value": float(arr[idx])})
    return pd.DataFrame(rows)


def _write_c8_parameter_tables(config: CorrectedAuditConfig, data_dir: Path) -> None:
    base_config = CorrectedAuditConfig(seeds=config.seeds)
    base = _flatten_parameters(_params_for_config(base_config)).rename(columns={"value": "previous_value"})
    c8 = _flatten_parameters(_params_for_config(config)).rename(columns={"value": "frozen_c8_value"})
    merged = base.merge(c8, on=["parameter", "index"], how="outer")
    merged["changed"] = ~np.isclose(
        merged["previous_value"].fillna(np.nan),
        merged["frozen_c8_value"].fillna(np.nan),
        equal_nan=True,
    )
    c8.to_csv(data_dir / "frozen_c8_parameter_set.csv", index=False)
    merged.to_csv(data_dir / "frozen_c8_parameter_comparison_to_previous_model.csv", index=False)


def _verify_untouched_seeds(output_root: Path) -> pd.DataFrame:
    rows = []
    for seed in UNTOUCHED_SEEDS:
        rows.append(
            {
                "seed": seed,
                "in_calibration_seed_set": seed in CALIBRATION_SEEDS,
                "status": "untouched_for_calibration" if seed not in CALIBRATION_SEEDS else "not_untouched",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(output_root / "untouched_seed_verification.csv", index=False)
    if bool(out["in_calibration_seed_set"].any()):
        raise AssertionError("a proposed final seed was used in calibration")
    return out


def _simulate_source_archives(config: CorrectedAuditConfig, source_dir: Path) -> list[dict[str, object]]:
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "final_c8_untouched_source_configuration.json").write_text(json.dumps(asdict(config), indent=2))
    _write_c8_parameter_tables(config, source_dir)
    seed_results = []
    for seed in config.seeds:
        data = load_saved_seed(seed, config, source_dir, multiparent=False)
        if data is None:
            print(f"simulating untouched C8 seed {seed}", flush=True)
            data = simulate_final_seed(seed, config, multiparent=False)
            save_seed_archive(source_dir / f"corrected_temporal_architecture_seed_{seed}.npz", data)
        else:
            print(f"loaded untouched C8 seed {seed}", flush=True)
        seed_results.append(data)
    return seed_results


def _phenotype_targets_pass(summary: pd.DataFrame, by_seed: pd.DataFrame, support: pd.DataFrame) -> tuple[bool, pd.DataFrame]:
    q = summary.set_index("quantity")
    pooled_complete = float(q.loc["complete_phenotype_transfer_entropy", "estimate_bits"])
    pooled_binary = float(q.loc["binary_nu_transfer_entropy", "estimate_bits"])
    min_seed_focal = float(by_seed["focal_nu_log_ratio_given_nu_bits"].min())
    all_support = support[support["seed"].astype(str) == "all"]["proportion"].to_numpy(dtype=float)
    rows = [
        ("pooled_complete_four_state_information_ge_0.010", pooled_complete, ">=0.010", pooled_complete >= 0.010),
        ("pooled_binary_nu_information_ge_0.010", pooled_binary, ">=0.010", pooled_binary >= 0.010),
        ("seed_level_focal_nu_log_ratio_positive", min_seed_focal, ">0 in every seed", min_seed_focal > 0.0),
        ("complete_information_ge_binary_nu_information", pooled_complete - pooled_binary, ">=0", pooled_complete + 1e-12 >= pooled_binary),
        ("minimum_phenotype_state_probability_ge_0.05", float(all_support.min()), ">=0.05", float(all_support.min()) >= 0.05),
        ("maximum_phenotype_state_probability_le_0.70", float(all_support.max()), "<=0.70", float(all_support.max()) <= 0.70),
    ]
    checks = pd.DataFrame(
        [{"criterion": name, "observed_value": value, "target": target, "passes": bool(passes)} for name, value, target, passes in rows]
    )
    return bool(checks["passes"].all()), checks


def _build_whole_source_table(seed_results: list[dict[str, object]], config: CorrectedAuditConfig) -> dict[str, object]:
    taus = list(range(config.source_tau_start, config.source_tau_stop + 1))
    metas = []
    targets = []
    sources = []
    conditions = []
    pair_rows = []
    for seed_position, seed_data in enumerate(seed_results):
        seed = int(seed_data["seed"])
        pilot = seed_data["config"]
        rng = np.random.default_rng(1_300_000 + seed)
        for _ in range(10000):
            pairs = rng.permutation(pilot.n_lineages)
            if np.all(pairs != np.arange(pilot.n_lineages)):
                break
        else:
            raise RuntimeError(f"failed to generate derangement for seed {seed}")
        for d, dp in enumerate(pairs):
            pair_rows.append({"seed": seed, "lineage_id": d, "source_lineage_d_prime": int(dp), "same_individual": bool(d == int(dp))})
        for tau in taus:
            meta = pd.DataFrame(
                {
                    "seed": seed,
                    "lineage_id": np.arange(pilot.n_lineages, dtype=int),
                    "unit_id": seed_position * pilot.n_lineages + np.arange(pilot.n_lineages, dtype=int),
                    "tau": tau,
                }
            )
            metas.append(meta)
            targets.append(_concat_levels(seed_data, tau + 1, ELC_LEVELS))
            source_current = _concat_levels(seed_data, tau, ELC_LEVELS)
            sources.append(source_current[pairs])
            conditions.append(_history(seed_data, tau, ELC_LEVELS, config.history_order))
    return {
        "meta": pd.concat(metas, ignore_index=True),
        "whole_source_target_full_elc_future": np.vstack(targets),
        "whole_source_elc_from_d_prime": np.vstack(sources),
        "whole_source_receiving_full_elc_history": np.vstack(conditions),
        "whole_source_derangement_pairs": pd.DataFrame(pair_rows),
    }


def run_whole_source_elc_contribution(
    seed_results: list[dict[str, object]],
    config: CorrectedAuditConfig,
    output_dir: Path,
    *,
    jitter: float,
) -> pd.DataFrame:
    table = _build_whole_source_table(seed_results, config)
    family = CmiFamily(
        "whole_source_elc_contribution_from_d_prime",
        "whole_source_target_full_elc_future",
        "whole_source_receiving_full_elc_history",
        (SourceBlock("complete_elc_from_d_prime", "whole_source_elc_from_d_prime"),),
        (("whole_source_elc_from_d_prime", ("complete_elc_from_d_prime",)),),
        "whole-source-ELC contribution from d' to the receiving lineage's future ELC",
    )
    joint, slices, source_arrays = _ranked_joint_arrays(table, family, include_generation=True)
    observed = _family_estimates_from_cov(_cov(joint, jitter), slices, family)
    boot = _bootstrap_family(joint, slices, family, table["meta"], n_boot=config.n_bootstrap, seed=config.bootstrap_seed + 202000, jitter=jitter)
    null = _null_family(joint, slices, family, table["meta"], n_null=config.n_null, seed=config.null_seed + 202000, jitter=jitter)
    summary = _summarize_family(
        family,
        observed,
        boot,
        null,
        target_dim=slices["target"].stop - slices["target"].start,
        source_dims={label: source_arrays[label].shape[1] for label in source_arrays},
        condition_dim=slices["condition"].stop - slices["condition"].start,
        jitter=jitter,
    )
    seed_rows = []
    for seed_data in seed_results:
        seed_table = _build_whole_source_table([seed_data], config)
        seed_joint, seed_slices, _ = _ranked_joint_arrays(seed_table, family, include_generation=True)
        seed_observed = _family_estimates_from_cov(_cov(seed_joint, jitter), seed_slices, family)
        seed_rows.append({"seed": int(seed_data["seed"]), "analysis": "whole_source_elc_from_d_prime", "raw_estimate_bits": float(seed_observed["whole_source_elc_from_d_prime"])})
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "whole_source_elc_transfer_entropy_summary.csv", index=False)
    boot.to_csv(output_dir / "whole_source_elc_transfer_entropy_bootstrap.csv", index=False)
    null.to_csv(output_dir / "whole_source_elc_transfer_entropy_generation_preserving_null.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(output_dir / "whole_source_elc_transfer_entropy_by_seed.csv", index=False)
    table["whole_source_derangement_pairs"].to_csv(output_dir / "whole_source_elc_derangement_pairs.csv", index=False)
    pair_summary = table["whole_source_derangement_pairs"].groupby("seed").agg(
        n_receiving=("lineage_id", "count"),
        n_unique_sources=("source_lineage_d_prime", "nunique"),
        any_self_pair=("same_individual", "any"),
    ).reset_index()
    pair_summary.to_csv(output_dir / "whole_source_elc_derangement_pair_summary.csv", index=False)
    return summary


def main() -> None:
    root = Path(__file__).resolve().parent
    output_root = root / "outputs" / "corrected_temporal_architecture_final_strict_early"
    output_root.mkdir(parents=True, exist_ok=True)
    config = CorrectedAuditConfig(
        seeds=UNTOUCHED_SEEDS,
        parameter_overrides=calibration_candidates()["C8_stronger_growth_allocation"],
    )
    (output_root / "final_c8_untouched_configuration.json").write_text(json.dumps(asdict(config), indent=2))
    seed_verification = _verify_untouched_seeds(output_root)
    print(seed_verification.to_string(index=False), flush=True)

    source_dir = output_root / "source_data"
    seed_results = _simulate_source_archives(config, source_dir)
    table = build_analysis_table(seed_results, config)
    support = phenotype_support(table)
    support.to_csv(output_root / "phenotype_support_by_seed_precheck.csv", index=False)
    categorical_summary_path = output_root / "categorical_phenotype_information_summary.csv"
    categorical_by_seed_path = output_root / "categorical_phenotype_by_seed.csv"
    if categorical_summary_path.exists() and categorical_by_seed_path.exists():
        categorical_summary = pd.read_csv(categorical_summary_path)
        categorical_by_seed = pd.read_csv(categorical_by_seed_path)
        print("loaded corrected categorical phenotype outputs from the current architecture run", flush=True)
    else:
        categorical_summary, _, categorical_by_seed, _ = run_categorical_phenotype_information(
            table,
            config,
            output_root,
        )
    ok, checks = _phenotype_targets_pass(categorical_summary, categorical_by_seed, support)
    checks.to_csv(output_root / "final_phenotype_acceptance_checks.csv", index=False)
    print(checks.to_string(index=False), flush=True)
    if not ok:
        raise AssertionError(
            "phenotype acceptance gate failed after the temporal-architecture correction; "
            "the untouched evaluation must be reported as failed and must not be retuned"
        )

    corrected_root = output_root / "corrected_numerical_audit"
    result = run_corrected_numerical_audit(
        output_dir=corrected_root,
        source_archive=source_dir,
        config=config,
    )
    corrected_data = Path(result["data_dir"])
    selected_jitter = float((corrected_data / "selected_common_covariance_jitter.txt").read_text().strip())

    pd.DataFrame(
        [
            {
                "analysis": "independently_paired_whole_source_elc",
                "status": "not_run",
                "reason": "superseded by the generative inter-ELC arm in which d-prime enters through B during simulation",
            }
        ]
    ).to_csv(output_root / "superseded_analysis_status.csv", index=False)

    completeness = run_final_draft32_completeness_audit(
        output_root=output_root / "draft32_completeness_audit",
        corrected_archive=corrected_data,
        frozen_source_archive=source_dir,
        config=config,
    )
    print("complex-unit qualification", flush=True)
    print(completeness["complex_unit_qualification"].to_string(index=False), flush=True)
    print(f"outputs={output_root}", flush=True)


if __name__ == "__main__":
    main()
