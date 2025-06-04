import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from csunisa.plot_utils import plot_result
from unit_tests.test_utils import print_result
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def test_lmm(lmm, ivp, h, test_tol=1e-3, tol=1e-6, max_iter=100, plot=False):
    """
    Test a linear multistep method on an IVP.

    Parameters
    ----------
    lmm : LinearMultistepMethod
        Method to be tested.
    ivp : InitialValueProblem
        IVP to test the problem on.
    h : float
        Step size.
    test_tol : float, default 1e-3
        Tolerance to determine success.
    tol : float, default 1e-6
        Tolerance for implicit methods.
    max_iter : int, default 100
        Maximum iterations for implicit methods.
    plot : bool, default False
        Whether to show plots.
    """
    starting = ivp.compute_starting_values(lmm.k, h)
    t, y = lmm.solve(ivp, h, starting=starting, tol=tol, max_iter=max_iter)
    y_exact = ivp.compute_exact_solution(t)
    if y_exact.ndim == 1:
        y_exact = y_exact.reshape(-1, 1)

    err_abs = []
    err_rel = []
    for i in range(len(t)):
        err = la.norm(y[i, :] - y_exact[i, :])
        ref = la.norm(y_exact[i, :])
        err_abs.append(err)
        err_rel.append(err / ref if ref > 0 else np.inf)

    err_abs = np.array(err_abs)
    err_rel = np.array(err_rel)

    rel_max = np.max(err_rel)

    # Classification
    if np.isnan(rel_max) or np.isinf(rel_max):
        status = "FAIL"
    elif rel_max < test_tol:
        status = "PASS"
    elif rel_max < 1e3 * test_tol:
        status = "WARN"
    else:
        status = "FAIL"

    print_result(f"{ivp.name}, h={h}", status, np.max(err_abs))

    if plot:
        plot_result(t, y, lmm.name, ivp, h)
        plt.figure()
        plt.semilogy(t, err_abs)
        plt.title(f'{lmm.name}, {ivp.name}, h={h}')
        plt.xlabel('t')
        plt.ylabel('error')
        plt.grid()
        plt.show()

def test_pc(pc, ivp, mu, h=0.05, tol=1e-3, test_tol=1e-3, plot=False):
    """
    Test a predictor-corrector method on an IVP.

    Parameters
    ----------
    pc : PredictorCorrector
        Method to be tested.
    ivp : InitialValueProblem
        IVP to test the problem on.
    mu : int
        As in P(EC)^μ E.
    h : float, default 0.05
        Initial step size.
    tol : float, default 1e-6
        Tolerance to accept or reject a step.
    test_tol : float, default 1e-3
        Tolerance to determine success.
    plot : bool, default False
        Whether to show plots.
    """
    starting = ivp.compute_starting_values(pc.predictor.k, h)
    t, y = pc.solve(ivp, mu, h, starting, tol)
    y_exact = ivp.compute_exact_solution(t)
    if y_exact.ndim == 1:
        y_exact = y_exact.reshape(-1, 1)

    err_abs = []
    for i in range(len(t)):
        err = la.norm(y[i, :] - y_exact[i, :])
        err_abs.append(err)

    err_abs = np.array(err_abs)
    err_max = np.max(err_abs)

    # Classification
    if np.isnan(err_max) or np.isinf(err_max):
        status = "FAIL"
    elif err_max < test_tol:
        status = "PASS"
    elif err_max < 1e3 * test_tol:
        status = "WARN"
    else:
        status = "FAIL"

    print_result(f"{ivp.name}", status, err_max)

    if plot:
        plot_result(t, y, pc.name, ivp)
        plt.figure()
        plt.semilogy(t, err_abs)
        plt.title(f'{pc.name}, {ivp.name}')
        plt.xlabel('t')
        plt.ylabel('error')
        plt.grid()
        plt.show()
