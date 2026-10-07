from __future__ import annotations

from pathlib import Path

import run_c12_inter_elc as runner
from run_c13_final_untouched import FINAL_SEEDS as C13_PRIMARY_FINAL_SEEDS


def main() -> None:
    runner.CALIBRATION_ROOT = Path("outputs/c13_inter_elc_calibration")
    runner.FINAL_ROOT = Path("outputs/c13_inter_elc_final_untouched")
    runner.FINAL_SEEDS = (298463771, 309016999, 320377241, 331662479)
    runner.CANDIDATE = "C13_robust_complete_graph_recovery"
    runner.C12_PRIMARY_FINAL_SEEDS = C13_PRIMARY_FINAL_SEEDS
    runner.main()


if __name__ == "__main__":
    main()
