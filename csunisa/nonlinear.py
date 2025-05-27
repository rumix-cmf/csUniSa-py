"""
nonlinear.py

This module contains numerical methods for solving nonlinear equations
"""
import numpy as np


def fixed_point_iteration(g, x0, tol=1e-6, max_iter=100):
    """
    This function uses the fixed-point iteration method to find the root of the
    nonlinear equation x= g(x). The contraction g is applied until the solution
    converges within a specific tolerance or the maximum number of iterations
    is reached.

    Parameters
    ----------
    g : callable
        A contraction g(x).
    x0 : ndarray
        Initial guess.
    tol : float, optional
        Tolerance. The default is 1e-6.
    max_iter : int, optional
        Maximum iterations. The default is 100.

    Returns
    -------
    x_new : ndarray
        Fixed point of g, if found.

    iterations : int
        Number of iterations.
    """
    x = x0
    iterations = 0

    while iterations < max_iter:
        x_new = g(x)
        difference = np.abs(x_new - x)
        if np.all(difference < tol):
            return x_new, iterations
        x = x_new
        iterations = iterations + 1

    raise AssertionError("Max iterations reached. Convergence may not be "
                         "achieved.\n")
