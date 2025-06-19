import numpy as np
from numpy import linalg as la


class RungeKutta:
    """
    A Runge-Kutta (RK) method for numerically solving initial value problems.

    Parameters
    ----------
    A : ndarray (s, s)
        Coefficients A = (a_ij) from the Butcher table.
    b : array_like
        Coefficients b = (b_i) from the Butcher table.
    c : ndarray, optional
        Coefficients c = (c_i) from the Butcher table, default assumed row-sum
        of the A coefficients.
    order : int, optional
        Consistency order p.
    order_embedded : int, optional
        Consistency order p* of the second, higher order method, if any.
    b_embedded : array_like, optional
        Coefficients b* of the second, higher order method, if any.
    name : str, optional
        Label for identification.

    Attributes
    ----------
    s : int
        Number of stages.
    A : ndarray (s, s)
        Coefficients A = (a_ij) from the Butcher table.
    b : ndarray
        Coefficients b = (b_i) from the Butcher table.
    c : ndarray
        Coefficients c = (c_i) from the Butcher table, default assumed row-sum
        of the A coefficients.
    order : int, optional
        Consistency order p.
    b_embedded : array_like, optional
        Coefficients b* of the second, higher order method, if any.
    name : str, optional
        Label for identification.
    """

    def __init__(self, A, b, c=[], order=None, order_embedded=None,
                 b_embedded=[], name=''):
        self.s = len(A)
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        if c is not None:
            self.c = c
        else:
            self.c = np.ndarray((self.s, 1))
            for i in range(self.s):
                self.c[i] = np.sum(self.A[i, :])
        self.order = order
        self.order_embedded = order_embedded
        self.b_embedded = b_embedded
        self.name = name or 'unnamed RK'

    def solve_fs(self, ivp, h):
        """
        WARNING: ONLY SUPPORTS EXPLICIT RK

        Apply the method to an IVP, with fixed step.

        Parameters
        ----------
        ivp : InitialValueProblem
        h : float
            Step size.

        Returns
        -------
        t : ndarray
            Time points.
        y : ndarray (len(t), len(y0))
            Array of solution values at each time step.
        """
        t0, tf = ivp.t_span
        t = np.arange(t0, tf + h, h)
        y = np.zeros((len(t), len(ivp.y0)))
        y[0] = ivp.y0
        k = np.zeros((self.s, len(ivp.y0)))

        for m in range(1, len(t)):
            k[0] = ivp.f(t[m-1], y[m-1])
            for i in range(1, self.s):
                k[i] = ivp.f(t[m-1] + self.c[i] * h, y[m-1] + h * sum(
                    self.A[i, j] * k[j] for j in range(i)
                ))
            y[m] = y[m-1] + h * sum(self.b[j] * k[j] for j in range(self.s))

        return t, y

    def solve_vs(self, ivp, h=0.05, tol=1e-3):
        """
        WARNING: ONLY SUPPORTS EXPLICIT RK

        Apply the method to an IVP, with adaptive step.

        Parameters
        ----------
        ivp : InitialValueProblem
        h : float, default 0.05
            Initial step size.
        tol : float, default 1e-3
            Tolerance to accept or reject a step.
        """
        t0, tf = ivp.tspan
        t = [t0]
        y = np.zeros((1, len(ivp.y0)))
        y[0] = ivp.y0
        ystar = y
        k = np.zeros((self.s, len(ivp.y0)))

        # Main loop
        while t[-1] < tf:
            # Make sure the current step does not exceed timespan
            if t[-1] + h > tf:
                h = tf - t[-1]

            # Apply both methods
            k[0] = ivp.f(t[-1], y[-1])
            for i in range(1, self.s):
                # VEDI DOMANDE RICEVIMENTO
                k[i] = ivp.f(t[-1] + self.c[i] * h, y[-1] + h * sum(
                    self.A[i, j] * k[j] for j in range(i)
                ))
            y_new = y[-1] + h * sum(
                self.b[j] * k[j] for j in range(self.s)
            )
            ystar_new = y[-1] + h * sum(
                self.b_embedded[j] + k[j] for j in range(self.s)
            )

            # Accept current iteration, or try again with new step
            plte = la.norm(ystar_new - y_new)
            if plte < tol:
                t.append(t[-1] + h)
                y = np.vstack((y, y_new))
                facma = 1.5
            else:
                facma = 1

            h *= min(facma, max(
                0.05, 0.9 * (tol / plte)**(1 / (self.order + 1))
            ))

        return np.array(t), y