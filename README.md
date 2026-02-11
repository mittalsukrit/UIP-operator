# UIP-operator

Unified Innovized Progress (UIP) operator integrated with NSGA-III (and related evolutionary algorithms), implemented in a research-oriented fork of `pymoo`.

This repository accompanies the published manuscript in *IEEE Transactions on Evolutionary Computation*:

- Paper link: https://ieeexplore.ieee.org/document/10269130

---

## What this repository contains

This codebase is organized around a local copy/fork of `pymoo` with UIP-related algorithmic extensions and ready-to-run notebooks for constrained benchmark problems.

### High-level structure

- `pymoo/` – optimization framework sources (algorithms, operators, performance indicators, utilities).
- `pymoo/algorithms/nsga3_uip.py` – NSGA-III implementation adapted to use the custom UIP-enabled GA backend.
- `pymoo/algorithms/genetic_algorithm_uip.py` – core UIP-enabled genetic algorithm logic (CO/DO learning, adaptive repair, and internal termination heuristics).
- `run_*.ipynb` – runnable experiment notebooks:
  - `run_DASCMOP1.ipynb`
  - `run_DASCMOP9.ipynb`
  - `run_MW5.ipynb`
  - `run_CIBN2 (custom problem).ipynb`
- `data/` – example output arrays (`.npy`) for decision variables (`x`), objective values (`f`), and constraint values (`c`).

---

## Core idea: UIP with NSGA-III

At a practical level, the workflow in this repository is:

1. Define a constrained multi-objective problem.
2. Run `NSGA3` from `pymoo.algorithms.nsga3_uip` using reference directions.
3. Let the UIP-enhanced genetic engine adaptively trigger and apply learning/repair behavior while evolution proceeds.
4. Save final arrays (`X`, `F`, `CV`) and visualize Pareto approximations.

The notebooks use intentionally high generation limits (e.g., `n_gen = 15000`), while the algorithm includes internal logic that can stop earlier when conditions are met.

---

## Requirements

### Python

- Python 3.8+

### Python packages

The project uses a local `pymoo` package and imports common scientific Python dependencies in algorithms/notebooks, including:

- `numpy`
- `scipy`
- `scikit-learn`
- `autograd`
- `matplotlib`
- `joblib`

> Note: this repository does not currently provide a pinned `requirements.txt` or `pyproject.toml`; install the dependencies into your environment manually.

### Recommended setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy scipy scikit-learn autograd matplotlib joblib jupyter
```

---

## Quick start

### 1) Clone repository

```bash
git clone <your-fork-or-this-repo-url>
cd UIP-operator
```

### 2) Launch Jupyter

```bash
jupyter notebook
```

### 3) Run any example notebook

Open and execute cells in one of:

- `run_DASCMOP1.ipynb`
- `run_DASCMOP9.ipynb`
- `run_MW5.ipynb`
- `run_CIBN2 (custom problem).ipynb`

Outputs are saved under `data/` using the naming convention:

```text
data/<problem-name>-<key>-<seed>.npy
```

Where `<key>` is typically one of:

- `x` (decision vectors)
- `f` (objective vectors)
- `c` (constraint values)

---

## Reproducing notebook experiments

All provided notebooks follow a similar pattern:

1. Import `NSGA3` from `pymoo.algorithms.nsga3_uip`.
2. Build or load a benchmark problem.
3. Construct reference directions with `get_reference_directions("das-dennis", ...)`.
4. Instantiate the algorithm (`pop_size` commonly set to `len(ref_dirs)`).
5. Call `minimize(...)` with a generation-based termination tuple.
6. Save `X/F/CV` arrays into `data/` and optionally plot the final population.

To repeat experiments with a different seed/problem:

- change seed and problem definition cells,
- rerun all cells,
- compare generated `.npy` files and plots.

---

## Using UIP-NSGA-III from a Python script

If you prefer scripts over notebooks, you can use the same API pattern:

```python
from pymoo.algorithms.nsga3_uip import NSGA3
from pymoo.factory import get_problem, get_reference_directions
from pymoo.optimize import minimize

problem = get_problem("dascmop1")  # or your custom problem
ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=99 if problem.n_obj == 2 else 13)

algorithm = NSGA3(ref_dirs=ref_dirs, pop_size=len(ref_dirs))

res = minimize(
    problem,
    algorithm,
    ("n_gen", 15000),
    seed=1,
    save_history=False,
    verbose=True,
)

# res.X, res.F and constraint info can then be saved/visualized
```

---

## Notes on repository scope

- This repository includes a broad internal `pymoo` code tree; most research usage in this project centers around the UIP-enabled algorithm files and notebooks.
- The bundled code version reports `pymoo` version `0.4.2.rc1`.
- The notebooks are the intended entry points for reproducing reported behavior.

---

## Troubleshooting

### Import or module errors

- Ensure you run notebooks from repository root (`UIP-operator/`) so local `pymoo` is importable.
- Verify required packages are installed in the active environment.

### Very long runs

- Problem difficulty and generation limits can make runs slow.
- Start with lower generation counts for smoke tests.

### Plot issues in headless environments

- Use a notebook backend or save figures to files instead of interactive display.

---

## Contact

If you face issues running the code, you can reach out to:

- dhish.saxena@me.iitr.ac.in
- mittalsukrit@gmail.com
