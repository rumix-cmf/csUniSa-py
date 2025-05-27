def print_result(test_name, passed, error=None):
    """
    Print results for unit tests.
    """
    # ANSI colours
    RESET = "[0m"
    GREEN = "[32m"
    RED = "[31m"

    if passed:
        print(f"{test_name:<30} {GREEN}✅ PASS{RESET}")
        print(f"{GREEN}Max error: {error:.4f}{RESET}")
    else:
        print(f"{test_name:<30} {RED}❌ FAIL{RESET}")
        print(f"{RED}Max error: {error:.4f}{RESET}")


def plot_solution(t, y, method_name, problem_name, step_size, y_exact=None,
                  save_path=None, show=False):
    """
    Plot numerical and exact solutions (if given), returning the figure object.

    Parameters
    ----------
    t : ndarray
        Time values.
    y : ndarray
        Numerical solution (N, D) or (N,).
    method_name : str
        Name of the method used (e.g. "Euler").
    problem_name : str
        Descriptive name of the problem.
    step_size : float
        Step size used.
    y_exact_fn : ndarray, optional
        Exact solution (N, D) or (N,).
    save_path : str, optional
        If set, saves the plot to this path.
    show : bool, default False
        Whether to immediately show the plot window.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object for further editing.
    """
    import os
    import matplotlib.pyplot as plt

    if y_exact is not None:
        if y_exact.ndim == 1:
            y_exact = y_exact.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    fig, ax = plt.subplots()
    for i in range(y.shape[1]):
        if y_exact is not None:
            ax.plot(t, y_exact[:, i], label=f"y{i+1} (exact)", linestyle="-")
        ax.plot(t, y[:, i], "*", label=f"y{i+1} (numerical)")

    title = f"{method_name}, {problem_name}, h = {step_size}"
    ax.set_title(title)
    ax.set_xlabel("Time t")
    ax.set_ylabel("y(t)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
    if show:
        fig.show()

    return fig
