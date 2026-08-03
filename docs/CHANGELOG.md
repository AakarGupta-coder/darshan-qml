# Changelog

All notable changes to the Darshan Hybrid Quantum-Classical Machine Learning Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.0.0] - 2026-08-03

### Added
- **Hardware Acceleration (High-Performance Backends)**: 
  - Integrated support for the C++ `lightning.qubit` simulator, drastically reducing quantum circuit execution times on standard CPUs.
  - Added support for NVIDIA cuQuantum via the `lightning.gpu` simulator (requires Linux/WSL2 environment).
  - *Reference: See [Installation & Environment Setup](README.md#14-installation--environment-setup) in the main README.*
- **Global Configuration Engine**: 
  - Introduced `darshan_config.yaml` at the root level to act as a single source of truth for global state.
  - Implemented `utils/config.py` parser to load properties such as `theme`, `backend`, `random_seed`, and execution flags globally, avoiding hardcoded instantiations.
- **Automated Unit Testing Suite (`pytest`)**:
  - Created `tests/test_data.py` containing parameterized tests for the dataset loader (Iris, Wine, Moons) ensuring correct output shapes, PCA reduction behavior, and proper subsampling fractions.
  - Created `tests/test_models.py` with comprehensive smoke tests for model pipelines (e.g., verifying that `fit()` and `predict()` loops complete without crashing).
- **Project Packaging**:
  - Added `pyproject.toml` defining build system requirements (`setuptools`) and project metadata.
  - Enabled `pip install -e .` installation so Darshan components can be seamlessly imported from external scripts or Jupyter notebooks.
- **Git Hooks & CI Automation**:
  - Added `.pre-commit-config.yaml` configured to run the `ruff` linter and formatter on every commit to enforce code quality automatically.

### Changed
- **Robust Device Instantiation Architecture**: 
  - Refactored all quantum and hybrid model constructors (`models/ananta_vqc.py`, `models/ananta_pro.py`, `models/samyoga_svm.py`, etc.) to support dynamic backend passing.
  - Implemented a graceful `try-except` fallback loop: models will attempt to instantiate the configured `lightning` backend, but safely fallback to `default.qubit` or `default.mixed` if dependencies are missing.
- **Strict Python Type Hinting**:
  - Added explicit typing annotations across the `data/loader.py` module (e.g., `X: np.ndarray`, `y: np.ndarray`, returning complex dictionaries).
  - Enforced type safety in core model constructors and method signatures to improve IDE autocompletion and static analysis.
- **Codebase Linting & Standardization**:
  - Formatted 100% of the codebase to adhere to PEP-8 standards using `ruff format`.
  - Resolved over 100 minor linting violations (e.g., unused variable assignments, duplicated dictionary keys).
  - Explicitly configured the linter to gracefully ignore `E402` (Module level import not at top of file) to preserve the intentional late-loading of heavy dependencies like PennyLane for faster CLI startup times.

### Fixed
- **Exception Handling Hardening**: 
  - Audited the codebase to replace unsafe, bare `except:` blocks with explicit `except Exception:` handling inside statistical scoring pipelines (e.g., log-loss and ROC AUC calculators) in `samyoga_svm.py` and `samyoga_shadow.py`.
- **Formatting Artifacts**: 
  - Cleansed the entire repository of CRLF vs LF line-ending conflicts and collapsed excessive vertical whitespace blocks for optimal readability.

---

## [1.0.0] - Initial Release

### Added
- **Core Orchestrator**: 
  - Shipped `darshan.py` acting as an interactive CLI engine and REPL featuring Rich-powered terminal UI formatting, colorized outputs, and interactive menus.
  - *Reference: See [CLI Command Reference](docs/Guides/cli_and_commands.md) and [UI & Themes](docs/Guides/ui_and_themes.md).*
- **Parampara Models (Classical)**: 
  - Implemented highly optimized classical baseline algorithms including Support Vector Machines (SVM), HistGradientBoosting, and Calibrated Random Forests.
  - *Reference: See [Parampara Architecture Guide](docs/Guides/parampara_architecture.md).*
- **Ananta Models (Pure Quantum)**: 
  - Developed Variational Quantum Classifiers (VQC) using Angle Embeddings and Strongly Entangling Layers for pure quantum machine learning execution.
  - *Reference: See [Ananta Architecture Guide](docs/Guides/ananta_architecture.md).*
- **Samyoga Models (Hybrid Quantum-Classical)**: 
  - Engineered advanced neural-quantum hybrid architectures by combining quantum feature extraction circuits with classical machine learning classifiers.
  - *Reference: See [Samyoga Architecture Guide](docs/Guides/samyoga_architecture.md).*
- **Evaluation & Preprocessing Pipeline**: 
  - Created end-to-end dataset preprocessing methodologies (PCA, scaling) accessible via `data/loader.py`.
  - Built a rigorous fair-track cross-validation benchmarking system to ensure non-biased evaluation across models.
  - Implemented automated statistical significance testing and LaTeX table generation for research publications.
  - *Reference: See [Datasets & Preprocessing](docs/Guides/datasets_and_preprocessing.md) and [Fair-Track Methodology](docs/Guides/fair_track_methodology.md).*
- **Visualization Suite**: 
  - Shipped a comprehensive Matplotlib-based plotting engine (`ui/graphs.py`) for rendering convergence tracking charts, ROC curves, and benchmark comparison visualizations directly to the `results/figures/` directory.
