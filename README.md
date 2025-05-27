# csunisa

📘 **csunisa** is a Python package for educational exploration of numerical methods, especially initial value problems (IVPs) and ordinary differential equations (ODEs). It is the Python port of a [MATLAB library](https://github.com/rumix-cmf/csUniSa/tree/main) originally developed for coursework at the University of Salerno.

---

## 📂 Project Structure

```
csunisa/
├── csunisa/
│   ├── __init__.py
│   ├── odes.py
│   ├── plot_utils.py
│   └── reference_solvers.py  # High-accuracy reference generator
├── tests/
│   ├── test_euler.py
│   ├── ivp_cases.py
│   └── reference/
├── run_tests.py
├── pyproject.toml
├── README.md
└── CHANGELOG.md
```

---

## 🚀 Features

- ⚙️ Numerical solvers
- 📦 Modular design: solvers, plotting, IVP definitions
- ✅ Pretty test output with coloured pass/fail icons
- 📈 Reusable plotting with automatic labels and titles
- 🧪 Standalone test scripts — no test framework needed
- 📐 Reference data generation via `reference_solvers.py`

---

## 📦 Requirements

- Python 3.8 or later
- Dependencies: `numpy`, `matplotlib`, `scipy`

Install in editable/development mode:

```bash
pip install -e .
```

---

## 🧪 Running Tests

To run all solver tests:

```bash
python run_tests.py
```

---

## 🛠️ Generating Reference Solutions

Use `make_reference.py` to generate high-accuracy `.npz` files for IVP comparisons:

```bash
python make_reference.py
```

You can reuse the logic in `reference_solvers.generate_reference(...)` to build your own test references.

---

## 📜 Example: Euler + Plot

```python
from csunisa.odes import euler
from csunisa.plot_utils import plot_solution
import numpy as np

f = lambda t, y: -y
t, y = euler(f, (0, 5), np.array([1.0]), 0.1)

plot_solution(
    t, y,
    y_exact_fn=lambda t: np.exp(-t),
    method_name="Euler",
    problem_name="Decay",
    step_size=0.1,
    show=True
)
```

---

## 🧑‍💻 Author

Francesco R. — Mathematics student at the University of Salerno.
