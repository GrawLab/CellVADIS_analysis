# CellVADIS
Code needed to reproduce figures of the CellVADIS manuscript
* Wicker L., Varga S., Hasenauer J., Graw F.: **CellVADIS: Inferring cross-tissue immune cell trafficking dynamics from single-cell snapshots**, bioRxiv 2026, https://doi.org/10.64898/2026.11.18.767220.


The codebase combines:

- a stochastic mathematical model allowing for generation of synthetic snapshot data of marker-driven cell migration and differentiation,
- Evaluation workflows for trafficking prediction based on simulated data with known ground truth,
- experimental analysis notebooks for real datasets (Wong et al. 2016 and Conde et al. 2022),

to simulate and infer immune-cell trafficking dynamics from single-cell data.

If you are interested in applying CellVADIS to your own data, please use https://github.com/GrawLab/CellVADIS


## Project overview

- synthetic data generation with known ground truth,
- benchmarking inference methods such as OT-based location fate estimation,
- reproducing figures and analyses used in related experimental or methodological work,
- applying the framework to real immune trafficking datasets.

---

## Repository structure

```text
OT_trafficking/
├── README.md
├── requirements.txt
├── data/
│   ├── analyse_atlas.ipynb        (Figure 5, Supplementary Figure 7)
│   ├── analyse_TICA_CD8.ipynb     (Supplementary Figure 6)
│   ├── analyse_Wong_data.ipynb   (Figure 4, Supplementary Figure 5)
│   ├── TICA/     (https://www.tissueimmunecellatlas.org/)
│   └── Wongetal/ (http://flowrepository.org/id/FR-FCM-ZZTM)
├── simulator/
│   ├── __init__.py
│   ├── backend.py
│   ├── differentiation_functions.py
│   ├── migration_functions.py
│   ├── model.py
│   ├── simulations_figures.ipynb    (Supplementary Figure 1)
│   └── data/
├── simulator_unobserved/
│   ├── __init__.py
│   ├── backend.py
│   ├── differentiation_functions.py
│   ├── migration_functions.py
│   ├── model.py
│   ├── simulations_figures.ipynb    (Supplementary Figure 3)
│   └── figures/
├── trafficking_inference/
│   ├── __init__.py
│   ├── CellVADIS.py
│   ├── helper_functions.py
│   ├── method_testing/        (Figure 2, Suppl. Figure 2)
│   └── method_testing_unobserved/    (Figure 3, Suppl. Figure 4)
```

### Main components

#### `simulator/`
Core simulation model for marker-driven immune-cell trafficking.

This directory contains:

- `model.py` — main model orchestration,
- `differentiation_functions.py` — equations governing marker dynamics,
- `migration_functions.py` — transitions between compartments,
- `backend.py` — backend switching logic, including optional GPU use,
- `simulations_figures.py` — Usage example

The model is built around stochastic transition rates and Hill-function dependencies that depend on marker expression and copmartment dwell time.
- The simulator supports both regular NumPy and optional CuPy GPU execution via the backend switch in `simulator/backend.py`.


#### `simulator_unobserved/`
A variant of the simulator intended to analyse the effects of unobserved tissues on fate prediction.

#### `trafficking_inference/`
Inference and evaluation code for trafficking problems.

This includes:

- `CellVADIS.py` — main CellVADIS methods, StitchOT and JointOT
- `helper_functions.py` — evaluation utilities, fate calculations, accuracy metrics, and general analysis helpers,
- `method_testing/` — benchmarking experiments,
- `method_testing_unobserved/` — benchmarking accounting for unobserved tissues


#### `data/`
Application of CellVADIS to TICA (Conde et al.) and MASS (Wong et al.) datasets, as well as celltype specific analysis.


## Setup

Create a virtual environment and install dependencies (Code was tested with Python 3.12.11):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
