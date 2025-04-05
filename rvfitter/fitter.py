from typing import List

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, least_squares

from rvfitter import Model, Planet


class Fitter:
    """
    Fits the RV data using our multi-planet model.

    Uses the least squares optimization routine from scipy.optimize.

    Attributes:
        __t (NDArray[np.longdouble]): Array of observation times.
        __rv_obs (NDArray[np.longdouble]): Array of observed RV data.
        __rv_err (NDArray[np.longdouble]): Array of measurement errors.
        __fit_result (OptimizeResult): The result of the least_squares minimization.
        __model (Model): The fitted multi-planet RV model.
        __result (NDArray[np.longdouble]): The result RV data.
        __uncertainties (NDArray[np.longdouble]): The result uncertainties.
    """
    __rv_obs: NDArray[np.longdouble]
    __rv_err: NDArray[np.longdouble]
    __t: NDArray[np.longdouble]
    __fit_result: OptimizeResult
    __model: Model
    __result: NDArray[np.longdouble]
    __uncertainties: NDArray[np.longdouble]

    def __init__(self, t: NDArray[np.longdouble],
                 rv_obs: NDArray[np.longdouble],
                 rv_err: NDArray[np.longdouble] = None):
        self.__t = t
        self.__rv_obs = rv_obs
        self.__rv_err = rv_err if rv_err is not None else (
            np.zeros(rv_obs.shape[0], dtype=np.longdouble))

    def __residuals(self, theta: NDArray) -> NDArray:
        """
        Computes residuals between model and observed RV data.

        Parameter theta is an array containing the model parameters:
          For each planet: period, t_peri, k, e, omega [5 parameters per planet]
          Plus one additional parameter for the systemic velocity offset v0.

        Returns:
            res (NDArray[np.longdouble]): The residuals between the model and
              observed RV data.
        """
        num_planets: int = (len(theta) - 1) // 5
        v0: np.longdouble = np.longdouble(theta[-1])
        model_rv: NDArray[np.longdouble] = (
                np.zeros(self.__t.shape[0], dtype=np.longdouble) + v0)
        for i in range(num_planets):
            indices: slice = slice(i * 5, i * 5 + 5)
            p: np.longdouble
            t_peri: np.longdouble
            k: np.longdouble
            e: np.longdouble
            omega: np.longdouble
            p, t_peri, k, e, omega = theta[indices]
            planet: Planet = Planet(p, t_peri, k, e, omega)
            model_rv += planet.compute_radial_velocities(self.__t)
        res: NDArray[np.longdouble] = (
                (self.__rv_obs - model_rv) / self.__rv_err)
        return res

    def fit(self, initial_guess: NDArray) -> None:
        """
        Fit the model parameters to the data by minimizing the residuals.

        Parameters:
            initial_guess (NDArray): Initial guess for the parameters.
                The ordering is
                [p1, t_peri1, k1, e1, omega1,
                p2, t_peri2, k2, e2, omega2,
                ...,
                v0]

        Returns:
            result (OptimizeResult): The result of the least_squares minimization.
        """
        self.__fit_result = least_squares(self.__residuals, initial_guess)
        self.__result = self.__fit_result.x
        self.__compute_uncertainties()
        num_planets: int = (len(self.__result) - 1) // 5
        v0: np.longdouble = self.__result[-1]
        planets: List[Planet] = []
        for i in range(num_planets):
            indices: slice = slice(i * 5, i * 5 + 5)
            p: np.longdouble
            t_peri: np.longdouble
            k: np.longdouble
            e: np.longdouble
            omega: np.longdouble
            p, t_peri, k, e, omega = self.__result[indices]
            planets.append(Planet(p, t_peri, k, e, omega))
        self.__model = Model(planets, v0)

    def __compute_uncertainties(self) -> None:
        """
        Compute the parameter uncertainties from the covariance matrix.

        Uses the Jacobian from the optimization result along with the residuals
        to estimate the reduced chi-squared and then the covariance matrix.
        """
        J: NDArray[np.longdouble] = self.__fit_result.jac
        dof: np.long = np.long(self.__rv_obs.shape[0] - self.__result.shape[0])
        s_sq: np.longdouble = np.sum(self.__residuals(self.__result) ** 2) / dof
        covariance_matrix: NDArray[np.longdouble] = (
                np.linalg.inv(J.T.dot(J)) * s_sq)
        self.__uncertainties = np.sqrt(np.diag(covariance_matrix))

    @property
    def t(self) -> NDArray[np.longdouble]:
        return self.__t

    @property
    def rv_obs(self) -> NDArray[np.longdouble]:
        return self.__rv_obs

    @property
    def rv_err(self) -> NDArray[np.longdouble]:
        return self.__rv_err

    @property
    def fit_result(self) -> OptimizeResult:
        return self.__fit_result

    @property
    def model(self) -> Model:
        return self.__model

    @property
    def result(self) -> NDArray[np.longdouble]:
        return self.__result

    @property
    def uncertainties(self) -> NDArray[np.longdouble]:
        return self.__uncertainties
