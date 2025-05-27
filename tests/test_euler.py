import numpy as np
from csunisa.odes import euler
import ivp_cases as ivp
from csunisa.plot_utils import plot_solution
from scipy.interpolate import interp1d

# ANSI colours
RESET = "[0m"
GREEN = "[32m"
RED = "[31m"


def print_result(test_name, passed, error=None):
    if passed:
        print(f"{test_name:<30} {GREEN}✅ PASS{RESET}")
    else:
        print(f"{test_name:<30} {RED}❌ FAIL{RESET}")
        if error is not None:
            print(f"Max error: {error:.4f}")
            # raise AssertionError(f"{test_name} failed.")


# def test_euler_decay():
#     def f(t, y):
#         return -y

#     t_span = (0, 1)
#     y0 = np.array([1])
#     h = 0.1

#     t, y = euler(f, t_span, y0, h)
#     y_exact = np.exp(-t)
#     error = np.abs(y[:, 0] - y_exact)
#     passed = np.all(error < 0.05)
#     print_result("Decay", passed, error.max())a


def test_euler_decay(h):
    f, t_span, y0, y_exact_fn = ivp.decay()

    t, y = euler(f, t_span, y0, h)
    y_exact = y_exact_fn(t)
    error = np.abs(y[:, 0] - y_exact)
    passed = np.all(error < 0.05)
    print_result(f"decay, h={h}", passed, error.max())
    plot_solution(t, y, "Euler", "decay", h, y_exact)


def test_euler_brusselator(h):
    f, t_span, y0, t_ref, y_ref = ivp.brusselator()

    t, y = euler(f, t_span, y0, h)
    y_exact = interp1d(t_ref, y_ref, axis=0, kind="linear",
                       fill_value="extrapolate")(t)
    error = np.abs(y - y_exact)
    passed = np.all(error < 0.05)
    print_result(f"Brusselator, h={h}", passed, error.max())
    plot_solution(t, y, "Euler", "Brusselator", h, y_exact)


if __name__ == "__main__":
    print(f"{'Test Case':<30} {'Result'}")
    print("-" * 42)
    test_euler_decay(0.1)
    test_euler_brusselator(0.1)
    test_euler_brusselator(0.001)
