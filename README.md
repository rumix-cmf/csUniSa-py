# csUniSa-py

📘 **csunisa** is a Python package for educational exploration of numerical methods, especially initial value problems (IVPs) and ordinary differential equations (ODEs). It is the Python port of a [MATLAB library](https://github.com/rumix-cmf/csUniSa/tree/main) originally developed for coursework at the University of Salerno.


## 📦 Requirements

- Python 3.8 or later
- Dependencies: `numpy`, `matplotlib`, `scipy`

---

## 🧪 Running Tests

To run all solver tests:

```bash
python run_tests.py
```

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
