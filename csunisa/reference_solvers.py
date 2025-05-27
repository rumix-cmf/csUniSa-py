"""
reference_solvers.py

Provides utilities to generate high-accuracy reference solutions for ODEs
using SciPy's solve_ivp for use in numerical method validation.
"""

import numpy as np
from scipy.integrate import solve_ivp


def generate_reference(f, t_span, y0, h=0.001, method="RK45", rtol=1e-10,
                       atol=1e-12, save_path=None):
    """
    Generate a high-accuracy reference solution using solve_ivp.

    Parameters
    ----------
    f : callable
        The ODE system function f(t, y).
    t_span : tuple
        Time interval as (t0, tf).
    y0 : list or ndarray
        Initial condition.
    h : float
        Step size for t_eval.
    method : str
        Solver method for solve_ivp (default: RK45).
    rtol : float
        Relative tolerance.
    atol : float
        Absolute tolerance.
    save_path : str, optional
        Path to save the solution in .npz format.

    Returns
    -------
    t : ndarray
        Time values.
    y : ndarray
        Solution values, shape (N, D)
    """
    t_eval = np.arange(t_span[0], t_span[1] + h, h)
    sol = solve_ivp(f, t_span, y0, method=method,
                    t_eval=t_eval, rtol=rtol, atol=atol)

    t = sol.t
    y = sol.y.T

    if save_path:
        np.savez(save_path, t=t, y=y)

    return t, y
