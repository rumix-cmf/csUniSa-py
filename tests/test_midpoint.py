import numpy as np
from csunisa.odes import midpoint
import ivp_cases as ivp
from csunisa.utils import print_result, plot_solution
from scipy.interpolate import interp1d
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def test_midpoint_decay(h):
    f, t_span, y0, y_exact_fn = ivp.decay()
    y1 = y_exact_fn(h)

    t, y = midpoint(f, t_span, y0, y1, h)
    y_exact = y_exact_fn(t)
    error = np.abs(y[:, 0] - y_exact)
    passed = np.all(error < 0.05)
    print_result(f"decay, h={h}", passed, error.max())
    plot_solution(t, y, "midpoint", "decay", h, y_exact)


def test_midpoint_brusselator(h):
    f, t_span, y0, t_ref, y_ref = ivp.brusselator()

    t = np.arange(t_span[0], t_span[1] + h, h)
    y_exact = interp1d(t_ref, y_ref, axis=0, kind="linear",
                       fill_value="extrapolate")(t)
    y1 = y_exact[1]
    t, y = midpoint(f, t_span, y0, y1, h)
    error = np.abs(y - y_exact)
    passed = np.all(error < 0.05)
    print_result(f"Brusselator, h={h}", passed, error.max())
    plot_solution(t, y, "midpoint", "Brusselator", h, y_exact)


if __name__ == "__main__":
    print(f"{'Test Case':<30} {'Result'}")
    print("-" * 40)
    test_midpoint_decay(0.1)
    test_midpoint_brusselator(0.1)
    test_midpoint_brusselator(0.001)
