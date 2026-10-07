# An Information Theory of Extended Heredity

Code and publication files for the simulated Extended Life Cycle (ELC) analysis in *An Information Theory of Extended Heredity*.

The repository contains the final C13 simulation, information analyses, intervention analyses, partial information decompositions, Fisher-information calculations, additional-lineage analysis, validation tests, and publication figures. Large simulated arrays and intermediate calibration runs are not versioned.

## Publication record

- `publication/manuscript/` contains the final manuscript source, figures, and PDF.
- `publication/supplement/` contains the final supplementary source, figures, and PDF.
- `results/` contains the fixed final configuration, acceptance checks, and figure source tables.

The publication files correspond to the October 7, 2026 manuscript and supplement supplied with this repository.

## Installation

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Tests

```bash
python -m pytest -q
```

The tests cover the asynchronous time architecture, additional-lineage analysis, dependency recovery, time-profile completeness, and Fisher-information checks.

## Final workflows

```bash
python run_c13_final_untouched.py
python run_c13_final_full_graph.py
python run_c13_inter_elc.py
python run_c13_framework_acceptance_gate.py
python run_publication_supplement.py
```

The last command regenerates the publication figures after the final simulation outputs are available. The full simulation requires substantial memory, storage, and runtime. Generated files are written under `outputs/`, which is excluded from version control.

## Repository structure

- `src/` contains the simulation and analysis implementation.
- `scripts/` contains final robustness and post-processing analyses.
- `reference_code/` contains the Gaussian-deficiency PID implementation used by the analysis.
- `tests/` contains executable tests and compact result fixtures.
- `publication/` contains the final manuscript and supplement.
- `results/` contains compact final summaries and figure source tables.
