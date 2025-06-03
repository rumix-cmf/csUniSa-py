# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Semantic Versioning](https://semver.org/).

---

## [1.2.1] - AB, AM, BDF

### Added

- Adams-Bashforth and Adams-Moulton methods, up to four steps
- BDF methods, up to four steps

### Changed

- Tester parameters tweaked
- Tester `print_result` output format to .2e

## [1.1.0] - Stability region plots

### Added

- `scanning` and `boundary_locus` methods for plotting the absolute stability region of linear multistep methods.

## [1.0.0] - Classes

### Added

- Introduced classes `LinearMultistepMethod` and `InitialValueProblem`

### Changed

- Project structure and code refactored accordingly
- Unit test framework refactored to accomodate classes and their methods

## [0.2.0] - New methods and utilities

### Added

- Midpoint rule (`midpoint`)
- Trapezoid rule (`trapezoid`)

### Changed

- Suppressed runtime warnings in all tests
- Removed project structure from `README.md`
- Utilities are now in a single module `utils.py`
- `print_result` always outputs max error, now accordingly coloured

## [0.1.0] - Initial Python Port

### Added
- Euler method (`euler`)
- `run_tests.py` to execute all tests
- Shared ODE problem definitions in `ivp_cases.py`
- `plot_utils.py` for reusable visualisation
  - Plots exact vs. numerical with stars and lines
  - Standardised title: "Method, Problem, Step"
  - Returns figure object for editing
- `reference_solvers.py` to generate high-accuracy `.npz` reference data
- Project configuration via `pyproject.toml`
- README and CHANGELOG documentation
