import numpy as np
from csunisa.odes import trapezoid
import ivp_cases as ivp
from csunisa.utils import print_result, plot_solution
from scipy.interpolate import interp1d
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def test_trapezoid_decay(h):
    f, t_span, y0, y_exact_fn = ivp.decay()

    t, y = trapezoid(f, t_span, y0, h)
    y_exact = y_exact_fn(t)
    error = np.abs(y[:, 0] - y_exact)
    passed = np.all(error < 0.05)
    print_result(f"decay, h={h}", passed, error.max())
    plot_solution(t, y, "trapezoid", "decay", h, y_exact)


def test_trapezoid_brusselator(h):
    f, t_span, y0, t_ref, y_ref = ivp.brusselator()

    t, y = trapezoid(f, t_span, y0, h)
    y_exact = interp1d(t_ref, y_ref, axis=0, kind="linear",
                       fill_value="extrapolate")(t)
    error = np.abs(y - y_exact)
    passed = np.all(error < 0.05)
    print_result(f"Brusselator, h={h}", passed, error.max())
    plot_solution(t, y, "trapezoid", "Brusselator", h, y_exact)


if __name__ == "__main__":
    print(f"{'Test Case':<30} {'Result'}")
    print("-" * 40)
    test_trapezoid_decay(0.1)
    test_trapezoid_brusselator(0.1)
    test_trapezoid_brusselator(0.05)
