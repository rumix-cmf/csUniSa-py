import numpy as np
from numpy import linalg as la


class PredictorCorrector:
    """
    A predictor-corrector (PC) method for numerically solving inital value
    problems.

    Parameters
    ----------
    predictor : LinearMultistepMethod
        An explicit LMM.
    corrector : LinearMultistepMethod
        An implicit LMM.
    name : str, optional
        Label for identification.

    Attributes
    ----------
    predictor : LinearMultistepMethod
        An explicit LMM.
    corrector : LinearMultistepMethod
        An implicit LMM.
    k : int
        Number of steps, which must be equal for P and C.
    w : float
        Constant W to compute Milne's PLTE estimate.
    name : str, optional
        Label for identification.
    """

    def __init__(self, predictor, corrector, name=""):
        self.predictor = predictor
        self.corrector = corrector
        if predictor.k != corrector.k:
            raise ValueError(f"Predictor steps kp={predictor.k} differ from "
                             f"corrector steps kc={corrector.k}")
        self.k = predictor.k
        self.w = corrector.error_constant / (predictor.error_constant
                                             - corrector.error_constant)
        self.name = name or predictor.name + "+" + corrector.name

    def solve(self, ivp, mu, h=0.05, starting=[], tol=1e-3):
        """
        Apply the method to an IVP. Uses Milne's estimate to compute the step
        size at each iteration. It is necessary that p* ≥ p, or p* + μ > p.

        Parameters
        ----------
        ivp : InitialValueProblem
        mu : int
            As in P(EC)^μ E.
        h : float, default 0.05
            Initial step size.
        starting : ndarray (k-1, len(y0)), optional
            Starting values for multistep methods.
        tol : float, default 1e-3
            Tolerance to accept or reject a step.

        Returns
        -------
        t : ndarray
            Time points.
        y : ndarray (len(t), len(y0))
            Array of solution values at each time step.
        """

        t0, tf = ivp.t_span
        t = [t0]
        y = np.zeros((self.k, len(ivp.y0)))
        y[0] = ivp.y0

        # Additional starting values
        if len(starting) == 0:
            if self.k > 1:
                raise ValueError(f"{self.k}-step method needs {self.k-1} "
                                 "additional starting values (got 0)")
        elif starting.shape != (self.k-1, len(ivp.y0)):
            raise ValueError(f"{self.k}-step method needs {self.k-1} "
                             "additional starting values "
                             f"(got {starting.shape[0]})")
        for i in range(1, self.k):
            y[i] = starting[i-1]
            t.append(t[-1] + h)

        # Main loop
        while t[-1] < tf:
            # Make sure current step does not exceed timespan
            if t[-1] + h > tf:
                h = tf - t[-1]

            # P
            y_pred = sum(
                - self.predictor.alpha[j] * y[j-self.k]
                + h * self.predictor.beta[j] * ivp.f(t[j-self.k], y[j-self.k])
                for j in range(0, self.k)
            )
            y0 = y_pred  # Needed for Milne's estimate

            # (EC)^μ
            for i in range(1, mu + 1):
                f_pred = ivp.f(t[-1] + h, y_pred)
                y_corr = sum(
                    - self.corrector.alpha[j] * y[j-self.k]
                    + h * self.corrector.beta[j] * ivp.f(t[j-self.k],
                                                         y[j-self.k])
                    for j in range(0, self.k)
                ) + h * self.corrector.beta[self.k] * f_pred
                y_pred = y_corr

            # Accept current iteration, or try again with new step
            plte = la.norm(self.w * (y_corr - y0))
            if plte < tol:
                t.append(t[-1] + h)
                y = np.vstack((y, y_corr))
                facma = 1.5
            else:
                facma = 1

            h *= min(facma, max(
                0.05, 0.9 * (tol / plte)**(1 / (self.corrector.order + 1))
            ))

        return np.array(t), y
