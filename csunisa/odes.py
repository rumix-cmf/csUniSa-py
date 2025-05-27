"""
odes.py

This module contains numerical methods for solving ordinary differential
equations (ODEs), with a focus on initial value problems (IVPs). Each solver is
implemented as a standalone function that follows a consistent interface.

Currently implemented methods:
- Euler method (explicit)

Planned methods:
- Midpoint method
- Trapezoid method

Each solver takes as input:
- A callable representing the ODE system: f(t, y)
- A time interval (t0, tf)
- An initial condition y0
- A step size h

Returns:
- t : 1D numpy array of time points
- y : 2D numpy array of solution values at each time point (shape: N x D)

Example usage:
    from csunisa.odes import euler
    t, y = euler(f, (0, 5), np.array([1.0]), 0.1)
"""

import numpy as np


def euler(f, t_span, y0, h):
    """
    Solve an ODE using the explicit Euler method.

    Parameters
    ----------
    f : callable
        The ODE system function f(t, y).
    t_span : tuple of float
        A tuple (t0, tf) for the time interval.
    y0 : ndarray
        Initial condition array.
    h : float
        Step size.

    Returns
    -------
    t : ndarray
        Time points.
    y : ndarray
        Array of solution values at each time step.
    """
    t0, tf = t_span
    t = np.arange(t0, tf + h, h)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0

    for i in range(1, len(t)):
        y[i] = y[i-1] + h * f(t[i-1], y[i-1])

    return t, y
