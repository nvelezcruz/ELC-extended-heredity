from __future__ import annotations

from pathlib import Path

import pandas as pd

from run_c13_final_untouched import FINAL_SEEDS
from src.final_conditional_dependence_reconstruction import (
    GraphReconstructionConfig,
    build_graph_reconstruction_table,
    estimate_graph_dependencies,
    known_dependency_table,
)
from src.final_numerical_audit import load_saved_seed
from src.recalibration_audit import calibration_candidates


def main() -> None:
    project_root = Path(__file__).resolve().parent
    source_dir = project_root / "outputs" / "c13_final_untouched" / "source_data"
    output_dir = (
        project_root / "outputs" / "c13_final_untouched" / "graph_reconstruction"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    config = GraphReconstructionConfig(
        seeds=FINAL_SEEDS,
        parameter_overrides=calibration_candidates()[
            "C13_robust_complete_graph_recovery"
        ],
    )
    seed_results = []
    for seed in FINAL_SEEDS:
        data = load_saved_seed(int(seed), config, source_dir, multiparent=False)
        if data is None:
            raise FileNotFoundError(f"missing untouched C13 source arrays for seed {seed}")
        seed_results.append(data)

    table = build_graph_reconstruction_table(seed_results, config)
    source_rows = []
    for kind in ("sources", "targets"):
        for level, values in table[kind].items():
            source_rows.append(
                {
                    "kind": kind[:-1],
                    "level": level,
                    "n_observations": int(values.shape[0]),
                    "dimension": int(values.shape[1]),
                }
            )
    matrix, summary, null = estimate_graph_dependencies(table, config)
    known_dependency_table().to_csv(
        output_dir / "graph_reconstruction_known_dependencies.csv", index=False
    )
    matrix.to_csv(output_dir / "graph_reconstruction_information_matrix.csv", index=False)
    summary.to_csv(output_dir / "graph_reconstruction_recovery_summary.csv", index=False)
    null.to_csv(
        output_dir / "graph_reconstruction_generation_preserving_null.csv", index=False
    )
    pd.DataFrame(source_rows).to_csv(
        output_dir / "graph_reconstruction_source_table.csv", index=False
    )
    print(summary.to_string(index=False), flush=True)
    missed = matrix[matrix["edge_status"] == "expected_missed"]
    if len(missed):
        print(missed.to_string(index=False), flush=True)
        raise AssertionError(
            f"untouched C13 final graph missed {len(missed)} encoded dependencies"
        )


if __name__ == "__main__":
    main()
