import csv
from pathlib import Path
from sympy import sympify
import numpy as np

_registry = {}

DATA_PATH = Path(__file__).parent / "data" / "lmm.csv"


def _load_registry(path=DATA_PATH):
    global _registry
    with open(Path(path), newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            name = row['name']
            k = int(row['k'])
            alpha = [float(sympify(a)) for a in row["alpha"].split(',')]
            beta = [float(sympify(b)) for b in row["beta"].split(',')]
            _registry[name] = {
                "k": k,
                "alpha": np.array(alpha),
                "beta": np.array(beta)
            }


_load_registry()


def get_method(name):
    """Return method data as a dictionary with keys: 'k', 'alpha', 'beta'."""
    return _registry[name]


def list_methods():
    """Return a list of all available method names."""
    return list(_registry.keys())
