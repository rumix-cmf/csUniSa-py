import csv
from pathlib import Path
from sympy import sympify
import numpy as np

_registry = {}
DATA_DIR = Path(__file__).parent / "data" / "rk"


def _load_registry():
    global _registry
    for path in DATA_DIR.glob("*.csv"):
        name = path.stem
        with open(path, newline='') as csvfile:
            reader = csv.reader(csvfile)

            meta = {}
            rows = []

            for row in reader:
                if not row:
                    continue
                if row[0].startswith('#'):
                    parts = row[0][1:].split(':', maxsplit=1)
                    if len(parts) == 2:
                        key, value = parts
                        meta[key.strip()] = value.strip()
                else:
                    rows.append(row)

            # Count stage rows (non-empty first column)
            s = 0
            for row in rows:
                if not row[0].strip():  # empty first cell ⇒ it's a b-row
                    break
                s += 1

            c = []
            A = []
            for row in rows[:s]:
                c.append(float(sympify(row[0])))
                A.append([float(sympify(aij)) for aij in row[1:s+1]])

            b_row = rows[s][1:s+1]
            b = [float(sympify(bi)) for bi in b_row]
            b_embedded = None
            if len(rows) > s + 1:
                b_embedded = [float(sympify(bi)) for bi in rows[s + 1][1:s+1]]

            _registry[name] = {
                "s": s,
                "A": np.array(A),
                "b": np.array(b),
                "c": np.array(c),
                "order": int(meta.get("order", 0)),  # fallback to 0
                "order_embedded": int(meta.get("order_embedded", 0)),
                "b_embedded": np.array(b_embedded) if b_embedded else None,
                "name": meta.get("name", name)
            }


_load_registry()


def get_method(name):
    """Return method data as a dictionary with keys: 's', 'A', 'b', 'c'."""
    return _registry[name]


def list_methods():
    """Return a list of all available method names."""
    return list(_registry.keys())
