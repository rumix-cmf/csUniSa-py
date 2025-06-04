import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


class InitialValueProblem:
    """
    An initial value problem (IVP) of the form:

    y(t) = f(t, y), t ∈ [t0, tf]
    y(t0) = y0

    Parameters
    ----------
    f : callable
        The function that determines the ODE.
    t_span : tuple of float
        A tuple (t0, tf) for the time interval.
    y0 : ndarray
        Initial condition array.
    name : str, optional
        Name or label for the problem.
    y_exact_fn : callable, optional
        The problem's exact solution y(x), if available. Alteratively, see the
        following attributes.

    Attributes
    ----------
    f : callable
        The function f(t, y) that determines the ODE.
    t_span : tuple of float
        A tuple (t0, tf) for the time interval.
    y0 : ndarray (M, 1)
        Initial condition array.
    name : str, optional
        Name or label for the problem.
    y_exact_fn : callable, optional
        The problem's exact solution y(x), if available. Alteratively, see the
        following attributes.
    t_ref, y_ref :  ndarray
        Time values and solution values of a reference solution.
    """

    def __init__(self, f, t_span, y0, name=None, y_exact_fn=None):
        self.f = f
        self.t_span = t_span
        self.y0 = y0
        self.name = name or "unnamed IVP"
        self.y_exact_fn = y_exact_fn

    def compute_starting_values(self, k, h):
        """
        Compute the additional starting values required by a multistep method.
        Requires an exact solution or a reference solution.

        Parameters
        ----------
        k : int
            Number of steps.
        h : float
            Step size.

        Returns
        -------
        ingition: ndarray (k-1, len(y0))
            Starting values y₁,...,y_k.
        y_exact : ndarray (len(t), len(y0))
            Useful byproduct.
        """
        t0, tf = self.t_span
        t = np.arange(t0, tf + h, h)
        starting = np.zeros((k-1, len(self.y0)))

        if self.y_exact_fn is not None:
            y_exact = self.y_exact_fn(t)
        elif self.t_ref is not None and self.y_ref is not None:
            y_exact = interp1d(self.t_ref, self.y_ref, axis=0,
                               fill_value="extrapolate")(t)
        else:
            raise ValueError("No exact/reference solution available.")
        if len(y_exact.shape) == 1:
            y_exact = y_exact.reshape((len(y_exact), 1))

        for i in range(1, k):
            starting[i-1] = y_exact[i]

        return starting, y_exact

    def plot_solution(self):
        """
        Plot exact or reference solution.

        Returns
        -------
        fig, ax
        """
        if self.y_exact_fn is not None:
            t0, tf = self.t_span
            t = np.linspace(t0, tf, 100)
            y = self.y_exact_fn(t)
        elif self.t_ref is not None and self.y_ref is not None:
            t, y = self.t_ref, self.y_ref
        else:
            raise ValueError("No exact nor reference solution available.")
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        fig, ax = plt.subplots()
        for i in range(y.shape[1]):
            ax.plot(t, y[:, i], label=f"y{i+1} (exact)", linestyle="--",
                    c='grey')

        return fig, ax