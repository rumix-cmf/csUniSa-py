import numpy as np
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt
from csunisa.nonlinear import fixed_point_iteration


class LinearMultistepMethod:
    """
    A linear multistep method (LMM) for numerically solving initial value
    problems.

    Parameters
    ----------
    alpha : array_like of float
        Coefficients of the y terms (α₀ to α_k).
    beta : array_like of float
        Coefficients of the f terms (β₀ to β_k).
    name : str, optional
        Name or label for the method.

    Attributes
    ----------
    alpha : ndarray
        Normalised array of α coefficients.
    beta : ndarray
        Array of β coefficients.
    k : int
        Number of steps (derived from the length of alpha).
    name : str
        Label for identification.
    """

    def __init__(self, alpha, beta, name=""):
        self.alpha = np.array(alpha, dtype=float)
        self.beta = np.array(beta, dtype=float)
        self.k = len(alpha) - 1
        self.name = name or "unnamed LMM"

        # Normalise if αₖ ≠ 1
        if self.alpha[-1] != 1:
            self.alpha /= self.alpha[0]
            self.beta /= self.alpha[0]

    def characteristic_polynomials(self):
        """
        Return the ρ(z) and σ(z) polynomials as NumPy Polynomial objects.
        """
        return Polynomial(self.alpha, symbol="z"), Polynomial(self.beta,
                                                              symbol="z")

    def stability_polynomial(self, hbar):
        """
        Return π(z; h) = ρ(z) - h * σ(z) as a Polynomial.

        Parameters
        ----------
        hbar : float
        """
        rho, sigma = self.characteristic_polynomials()
        return rho - hbar * sigma


    def solve(self, ivp, h, ignition=[], tol=1e-6, max_iter=100):
        """
        Apply the method to an IVP.

        Parameters
        ----------
        ivp : InitialValueProblem
        h : float
            Step size.
        ignition: ndarray (k-1, len(y0)), optional
            Ignition values for multistep methods.
        tol : float, default 1e-6
            Tolerance for implicit methods.
        max_iter : int, default 100
            Maximum iterations for implicit methods.

        Returns
        -------
        t : ndarray (len(t), 1)
            Time points.
        y : ndarray (len(t), len(y0))
            Array of solution values at each time step.
        """
        f = ivp.f
        t0, tf = ivp.t_span
        t = np.arange(t0, tf + h, h)
        y = np.zeros((len(t), len(ivp.y0)))
        y[0] = ivp.y0

        # Additional ignition values
        if ignition.shape != (self.k-1, len(ivp.y0)):
            raise ValueError(f"{self.k}-step method needs {self.k-1} "
                             "additional ignition values "
                             f"(got {ignition.shape[0]})")
        for i in range(1, self.k):
            y[i] = ignition[i-1]

        # Main loop for explicit methods
        if self.beta[-1] == 0:
            for i in range(self.k, len(t)):
                y[i] = sum(
                    -self.alpha[j] * y[i-self.k+j]
                    + h * self.beta[j] * f(t[i-self.k+j], y[i-self.k+j])
                    for j in range(0, self.k)
                )
        # Main loop for implicit methods
        else:
            for i in range(self.k, len(t)):
                def g(z):
                    return sum(
                        -self.alpha[j] * y[i-self.k+j]
                        + h * self.beta[j] * f(t[i-self.k+j], y[i-self.k+j])
                        for j in range(0, self.k)
                    ) + h * self.beta[self.k] * f(t[i], z)

                y[i], _ = fixed_point_iteration(g, y[i-1], tol, max_iter)

        return t, y

    def scanning(self, xs=[-4, 4], ys=[-4, 4], num=100):
        """
        Plot the method's absolute stability region by applying the scanning
        technique to the square xs × ys.
        """

        plt.xlim(xs)
        plt.ylim(ys)
        xs = np.linspace(xs[0], xs[1], num)
        ys = np.linspace(ys[0], ys[1], num)

        for x in xs:
            for y in ys:
                hbar = x + y*1j
                pi = self.stability_polynomial(hbar)
                if np.all(np.abs(pi.roots()) < 1 - 1e-10):
                    plt.plot(x, y, "ks")
        title = f"{self.name}, num={num}"
        plt.title(title)
        plt.grid()
        plt.show()

    def boundary_locus(self, xs=[-4, 4], ys=[-4, 4], num=50):
        """
        Plot the boundary of the method's absolute stability region by applying
        the boundary locus technique.
        """
        plt.xlim(xs)
        plt.ylim(ys)
        theta = np.linspace(0, 2*np.pi, num)
        rho, sigma = self.characteristic_polynomials()

        for t in theta:
            hbar = rho(np.exp(t*1j)) / sigma(np.exp(t*1j))
            plt.plot(np.real(hbar), np.imag(hbar), "ks")
        title = f"{self.name}, num={num}"
        plt.title(title)
        plt.grid()
        plt.show()
