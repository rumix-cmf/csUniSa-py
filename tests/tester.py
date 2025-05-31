import numpy as np
from csunisa.plot_utils import plot_solution
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def print_result(test_name, passed, error=None):
    """
    Print results for unit tests.
    """
    # ANSI colours
    RESET = "[0m"
    GREEN = "[32m"
    RED = "[31m"

    if passed:
        print(f"{test_name:<40} {GREEN}✅ PASS{RESET}")
        print(f"{GREEN}Max error: {error:.4f}{RESET}")
    else:
        print(f"{test_name:<40} {RED}❌ FAIL{RESET}")
        print(f"{RED}Max error: {error:.4f}{RESET}")


def test_lmm(lmm, ivp, h, test_tol=0.1, tol=1e-6, max_iter=100):
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
    test_tol : float, default 0.1
        Tolerance to determine success.
    tol : float, optional, default 1e-6
        Tolerance for implicit methods.
    max_iter : int, optional, default 100
        Maximum iterations for implicit methods.
    """
    ignition, y_exact = ivp.compute_ignition_values(lmm.k, h)
    t, y = lmm.solve(ivp, h, ignition=ignition, tol=tol, max_iter=max_iter)
    # if ivp.y_exact_fn is not None:
    #     y_exact = ivp.y_exact_fn(t)
    # else:
    #     y_exact = interp1d(ivp.t_ref, ivp.y_ref, axis=0, kind='linear',
    #                        fill_value='extrapolate')(t)
    error = np.abs(y - y_exact)
    passed = np.all(error < test_tol)
    print_result(f"{ivp.name}, h={h}", passed, error.max())
    plot_solution(t, y, lmm.name, ivp.name, h, y_exact)
