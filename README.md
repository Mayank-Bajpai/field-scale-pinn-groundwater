# Field-scale PINNs for groundwater: Bayesian optimisation and FiLM modulation

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Framework](https://img.shields.io/badge/DeepXDE-PyTorch%20backend-orange)

Code accompanying the manuscript

> Bajpai, M., Kumar, R., Singh, K., Gaur, S. **Scaling PINNs for Field-Scale Hydrogeology: Comparative Analysis of Bayesian Optimization and FiLM Modulation.** *Manuscript submitted.*

The repository implements physics-informed neural networks (PINNs) that jointly **predict groundwater head** $h(x, y, t)$
and **estimate aquifer parameters** — anisotropic hydraulic conductivity $(K_x, K_y)$ and storage — from sparse field
observations. It compares three configurations on a real, data-scarce alluvial aquifer (Varuna River basin, India):

| # | Configuration | Hyperparameters | Notebook |
|---|---|---|---|
| 1 | Baseline (simple) PINN — Fourier-feature MLP | fixed | [`notebooks/1_Simple_PINN.ipynb`](notebooks/1_Simple_PINN.ipynb) |
| 2 | Same PINN with Bayesian hyperparameter optimisation (Optuna, TPE) | tuned | [`notebooks/2_Optuna_PINN.ipynb`](notebooks/2_Optuna_PINN.ipynb) |
| 3 | FiLM-conditioned PINN with Bayesian hyperparameter optimisation | tuned | [`notebooks/3_Optuna_PINN_FiLM.ipynb`](notebooks/3_Optuna_PINN_FiLM.ipynb) |

All three share one module, [`pinn_unified.py`](pinn_unified.py).

---

## Contents

- [Method in brief](#method-in-brief)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Data](#data)
- [Quick start](#quick-start)
- [Validation splits](#validation-splits)
- [Outputs](#outputs)
- [Reproducibility](#reproducibility)
- [FAIR statement](#fair-statement)
- [How to cite](#how-to-cite)
- [License and acknowledgements](#license-and-acknowledgements)

---

## Method in brief

**Governing equation.** Two-dimensional transient groundwater flow,

```math
S \frac{\partial h}{\partial t} - \frac{\partial}{\partial x}\left(K_x \frac{\partial h}{\partial x}\right) - \frac{\partial}{\partial y}\left(K_y \frac{\partial h}{\partial y}\right) = q
```

is enforced as a residual loss on collocation points in the non-dimensionalised domain
$(\hat x, \hat y, \hat t) \in [0,1]^3$, with the chain-rule factors $1/L_t$, $1/L_x^2$ and $1/L_y^2$ applied explicitly
(`train_one_iteration → pde`).

**Networks.**

- **Head network** (`SingleHeadDecomposedNet`): Fourier-feature embedding of $(x, y, t)$ followed by a tanh MLP.
  With `use_film=True`, $(x, y)$ feed the main network while $t$ feeds an auxiliary network that emits feature-wise scale and
  shift parameters, `h*_l = (1 + γ(t)) ⊙ h_l + β(t)`, applied after
  every hidden linear layer (FiLM layers are zero-initialised, so training starts from the unmodulated network).
- **Parameter fields** (`AnisotropicHybridK`, `HybridFieldScalar`): a learnable anchor grid (bilinear interpolation) blended
  with a small coordinate MLP through a learnable weight; outputs are mapped exponentially to physical bounds
  ($K \in [10^{-2}, 5\times10^{2}]$ m d⁻¹; storage $S \in [0.02, 0.35]$, specific yield) and the anisotropy ratio is
  constrained.

**Loss.** PDE residual + river-stage Dirichlet condition + head observations + optional initial condition
(sampled from a kriged head raster), plus gradient, curvature, temporal-clipping and latent-field regularisation terms. Loss
weights and penalty strengths are the main targets of the Bayesian optimisation (`build_search_space`).

**Training.** DeepXDE (PyTorch backend), Adam; a warm-up stage on the full training partition followed by main iterations.

## Repository structure

```
field-scale-pinn-groundwater/
├── pinn_unified.py               # all model, training, optimisation and plotting code
├── notebooks/
│   ├── 1_Simple_PINN.ipynb       # baseline PINN (fixed hyperparameters)
│   ├── 2_Optuna_PINN.ipynb       # Bayesian-optimised PINN
│   └── 3_Optuna_PINN_FiLM.ipynb  # Bayesian-optimised FiLM PINN
├── examples/
│   └── make_synthetic_data.py    # writes a synthetic dataset with the required schema
├── data/
│   └── README.md                 # required input files and column schema (data not distributed)
├── requirements.txt / environment.yml
├── CITATION.cff / codemeta.json / .zenodo.json   # citation and machine-readable metadata
├── CHANGELOG.md
└── LICENSE
```

## Installation

Python ≥ 3.9. A CUDA GPU is strongly recommended for the optimisation notebooks; the code falls back to CPU.

```bash
git clone https://github.com/Mayank-Bajpai/field-scale-pinn-groundwater.git
cd field-scale-pinn-groundwater

# conda
conda env create -f environment.yml
conda activate pinn-gw

# or pip (install the PyTorch build for your platform first: https://pytorch.org/get-started/)
pip install -r requirements.txt
```

DeepXDE must use the PyTorch backend. The module sets `DDE_BACKEND=pytorch` before importing DeepXDE; if DeepXDE has
already been imported in the same session with another backend, restart the kernel.

## Data

The field observations (groundwater heads from monitoring wells and DGPS-surveyed river stage) are held by the Central
Ground Water Board (Government of India) and the Smart Lab on Clean Rivers (SLCR), IIT (BHU) Varanasi, and **are not
distributed with this repository**. Access requires a research collaboration with SLCR.

- The required files, column names and units are documented in [`data/README.md`](data/README.md).
- To run the full workflow without the restricted data, generate a **synthetic dataset with the same schema**:

  ```bash
  python examples/make_synthetic_data.py --out data
  ```

  The synthetic values are artificial and are intended only for testing the software.

## Quick start

**Notebooks** (run from the `notebooks/` folder; they look for input files in `../data/`):

```bash
jupyter lab notebooks/
```

**Python**

```python
import pinn_unified as pu

stage_s, gwt_s, well_s, stats_df = pu.load_and_preprocess_data(data_dir="data")
study, retrain = pu.optimize_pinn(
    n_trials=30, split_type="random", use_film=False,
    storage="sqlite:///study.db", output_dir="results_random", retrain=True,
)
```

**Command line**

```bash
python pinn_unified.py --data-dir data --split-type temporal --n-trials 30 \
       --storage sqlite:///study.db --output-dir results_temporal --retrain
```

Useful options: `--use-film`, `--sampler {tpe,botorch}`, `--fixed-split-seed`, `--global-seed`, `--force-cpu`,
`--amp`, `--low-mem`, `--no-plot-per-trial`. Run `python pinn_unified.py --help` for the full list.

## Validation splits

The groundwater-head observations are partitioned **once** into train / validation / test sets and the partition is held
fixed across all Optuna trials. The Optuna objective is the **Kling–Gupta efficiency (KGE) on the validation set**; the test
set is scored only when the best trial is retrained (`retrain=True`).

| `split_type` | Train / validation / test | Construction |
|---|---|---|
| `random` | 70 / 15 / 15 % | random permutation of all observations |
| `temporal` | 70 / 15 / 15 % | records sorted by time; a contiguous block in the middle of the record is withheld (validation, then test) — tests interpolation across a temporal gap |
| `temporal_lowdata` | 30 / 15 / 55 % | as `temporal`, with only 30 % of the record used for training |

## Outputs

`optimize_pinn` writes to `output_dir` (default `optuna_pinn_results_random_split`):

| File | Content |
|---|---|
| `trial_<n>/metrics.json` | train and validation RMSE, MAE, R², KGE and the sampled hyperparameters |
| `trial_<n>/predictions_train.csv`, `predictions_val.csv` | observed and predicted heads |
| `trial_<n>/trial_<n>_penalties.csv/.png`, `_loss.png` | per-epoch loss and penalty history |
| `trial_<n>/fig_*.pdf` | head maps, scatter plots, well time series, K and S fields |
| `trial_<n>/iter*_net.pt`, `*_K_aniso.pt`, `*_S_field.pt` | model weights (PyTorch state dicts) |
| `best_trial_summary.json`, `all_trials_metrics.csv` | study summary |
| `best_retrain/` | retrained best configuration, test metrics and predictions (`retrain=True`) |
| `best_retrain/aquifer_properties_K_S.csv` | inferred Kx, Ky (m d⁻¹) and S on a 100 × 100 grid (scaled and physical coordinates) |

The Optuna study itself is stored in the SQLite database given by `storage`, so studies can be resumed and analysed
(e.g. with `optuna.importance.get_param_importances`).

## Reproducibility

- Seeds: `global_seed` (Python, NumPy, PyTorch; cuDNN deterministic mode) and `fixed_split_seed` (data partition).
- Optuna: `TPESampler(multivariate=True)` with `MedianPruner`; persistent SQLite storage.
- Iteration budgets are module constants: `FIXED_ADAM_ITERS`, `FIXED_MAIN_NUM_ITERS`, `FIXED_HYBRID_N_ANCHOR`.
- Resource controls: at import the module limits the process to 25 % of GPU memory
  (`torch.cuda.set_per_process_memory_fraction(0.25)`); environment variables `PINN_LOW_MEM_CHECKPOINT`,
  `PINN_LOW_MEM_AMP`, `PINN_MEM_LOG` and `PINN_OFFLOAD_AFTER_ITER` (`"1"` to enable) control checkpointing,
  mixed precision, memory logging and offloading.
- Exact package versions used for a run can be recorded with `pip freeze > requirements-lock.txt`.

## FAIR statement

| Principle | How it is addressed |
|---|---|
| **Findable** | Public GitHub repository with descriptive metadata; [`CITATION.cff`](CITATION.cff), [`codemeta.json`](codemeta.json) and [`.zenodo.json`](.zenodo.json) so that each release can be archived on Zenodo with a persistent DOI. |
| **Accessible** | Open source under the MIT license; retrievable over standard protocols (git/HTTPS). Restricted field data are described, with the access route stated in [Data](#data). |
| **Interoperable** | Plain Python with widely used libraries (PyTorch, DeepXDE, Optuna); open formats for inputs and outputs (CSV, GeoTIFF, JSON, SQLite, PDF/PNG). |
| **Reusable** | Documented input schema, a synthetic dataset generator, pinned dependency ranges, fixed seeds, a changelog and a clear license. |

## How to cite

If you use this code, please cite the article (details will be updated on publication) and the software release. GitHub
shows a **"Cite this repository"** button generated from [`CITATION.cff`](CITATION.cff).

```bibtex
@software{bajpai_field_scale_pinn_groundwater,
  author  = {Bajpai, Mayank and Kumar, Ranveer and Singh, Kamal and Gaur, Shishir},
  title   = {Field-scale PINNs for groundwater: Bayesian optimisation and FiLM modulation},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/Mayank-Bajpai/field-scale-pinn-groundwater}
}
```

## License and acknowledgements

Released under the [MIT License](LICENSE).

The authors acknowledge the Prime Minister's Research Fellowship (PMRF), Ministry of Education, Government of India; the
Smart Lab on Clean Rivers (SLCR), IIT (BHU) Varanasi; and the PARAM Shivay facility under the National Supercomputing
Mission, Government of India, at IIT (BHU).

Questions and bug reports: please open a [GitHub issue](https://github.com/Mayank-Bajpai/field-scale-pinn-groundwater/issues).
