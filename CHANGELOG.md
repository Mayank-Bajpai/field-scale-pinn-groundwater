# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [1.0.0] — 2026-09-22

First public release accompanying the revised manuscript.

### Changed (model behaviour)
- `optimize_pinn(use_film=True)` / `--use-film` now trains the FiLM-conditioned network. Previously the search space
  forced `use_film=False`, so the FiLM configuration was never used in optimisation; the retrained best
  configuration (`retrain=True`) now also keeps the FiLM setting.
- Storage maps in plots (`fig_K_fields.pdf`) and in `sample_K_S_maps()` use the same range as training
  (S ∈ [0.02, 0.35]); previously they were mapped to 1e-8–5e-3.
- `derivative_scaling_factors.csv` with the `derivative, w_max` schema is now read, so the gradient-penalty search ranges
  are scaled as intended (previously only the `parameter, scale_factor` schema was accepted and the file was ignored).
- `include_forcings` defaults to `True` and is passed as `True` in all training calls.
- `retrain_best()` writes `aquifer_properties_K_S.csv` (Kx, Ky, S on a 100 × 100 grid); the previous call failed silently.

### Added
- `load_and_preprocess_data()` — loads and scales the observation files and makes them available to the training and
  optimisation functions (previously executed at import time).
- `DATA_DIR` / `data_path()` and the `PINN_DATA_DIR` environment variable to locate input files; `--data-dir` and
  `--output-dir` command-line options; `output_dir` argument of `optimize_pinn()`.
- `examples/make_synthetic_data.py` — synthetic dataset with the same file names and column schema as the restricted
  field data.
- Documentation (`README.md`, `data/README.md`) and metadata (`CITATION.cff`, `codemeta.json`, `.zenodo.json`),
  `requirements.txt`, `environment.yml`, `.gitignore`, MIT license.

### Changed
- The notebooks import `pinn_unified.py` instead of embedding a full copy of it, and call the module's actual API.
- `DDE_BACKEND=pytorch` is set before DeepXDE is first imported.
- The initial-condition raster is looked up in the data directory instead of an absolute local path.
- Data loading raises exceptions instead of calling `sys.exit()`, so failures surface normally in notebooks.
- Per-trial validation predictions are written to `predictions_val.csv` (previously skipped because of an undefined
  variable), and `all_trials_metrics.csv` reports validation metrics (`kge_val`, `rmse_val`, …) that the objective records.
- Per-iteration metadata is saved as `iter<N>_meta.json` (the file name previously contained a stray space).
- The command-line summary reports the best **validation** KGE (the quantity being optimised).

Apart from the items under *Changed (model behaviour)*, no change was made to the network architectures, loss
formulation or training procedure.
