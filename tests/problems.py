import os
import numpy as np
from tests.reference_solvers import generate_reference
from csunisa.ivp import InitialValueProblem


def get_reference_path(name):
    here = os.path.dirname(__file__)
    return os.path.join(here, "reference", f"{name}.npz")


def decay():
    def f(t, y):
        return -y

    t_span = (0, 1)
    y0 = np.array([1])

    def y_exact(t):
        return np.exp(-t)

    return InitialValueProblem(f, t_span, y0, "decay", y_exact)


def brusselator(A=1, B=3):
    def f(t, y):
        return np.array([A + y[0]**2 * y[1] - (B + 1) * y[0],
                         B * y[0] - y[0]**2 * y[1]])

    t_span = (0, 20)
    y0 = np.array([1.5, 3])

    ref_path = get_reference_path("brusselator_ref")

    if os.path.exists(ref_path):
        data = np.load(ref_path)
        t_ref, y_ref = data["t"], data["y"]
    else:
        t_ref, y_ref = generate_reference(f, t_span, y0, save_path=ref_path)

    ivp = InitialValueProblem(f, t_span, y0, f"Brusselator (A={A}, B={B})")
    ivp.t_ref = t_ref
    ivp.y_ref = y_ref

    return ivp


problems = {
    "decay": decay,
    "brusselator": brusselator
}
