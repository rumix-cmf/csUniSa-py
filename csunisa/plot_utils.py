import matplotlib.pyplot as plt


def plot_result(t, y, method_name, ivp, h=None):
    """
    Plot the numerical solution (t, y) on top of the exact solution.

    Parameters
    ----------
    t : ndarray
        Time values.
    y : ndarray
        Numerical solution.
    method_name : str
        Name of the method used.
    ivp : InitialValueProblem
        Test problem
    h : float, optional
        Step size, to be displayed in title.

    Returns
    -------
    fig, ax
    """
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    fig, ax = ivp.plot_solution()
    for i in range(y.shape[1]):
        ax.plot(t, y[:, i], marker="o", mfc="none", linewidth=0.5,
                label=f"y{i+1} (numerical)")

    if h is not None:
        h = f", h={h}"
    else:
        h = ""
    title = f"{method_name}, {ivp.name}" + h
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("y(t)")
    ax.grid()
    ax.legend()
    fig.tight_layout()

    return fig, ax


def plot_error_plte(t, error, plte, method_name, ivp):
    """
    Plot the estimated plte and the actual error.

    Parameters
    ----------
    t : ndarray
        Time values.
    error : ndarray
        Error's norm.
    plte : ndarray
        Norm of estimated plte.
    method_name : str
        Name of the method used.
    ivp : InitialValueProblem
        Test problem
    """
    fig, ax = plt.subplots()
    ax.semilogy(t, plte, label='estimate', linestyle='--', c='grey')
    ax.semilogy(t, error, label='actual')
    title = f"{method_name}, {ivp.name}"
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("error")
    ax.grid()
    ax.legend()
    fig.tight_layout()

    return fig, ax
