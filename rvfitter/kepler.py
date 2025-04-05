import numpy as np
from numpy.typing import NDArray


class Kepler:
    """
    Attributes:
        __M (NDArray[np.longdouble]): Mean anomaly in radians.
        __e (np.longdouble): Orbital eccentricity (0 <= e < 1).
        __E (NDArray[np.longdouble]): Eccentric anomaly in radians.
        __f (NDArray[np.longdouble]): True anomaly in radians.
        __beta (np.longdouble)
    """
    __M: NDArray[np.longdouble]
    __e: np.longdouble
    __E: NDArray[np.longdouble]
    __f: NDArray[np.longdouble]
    __beta: np.longdouble

    def __init__(self, M: NDArray[np.longdouble], e: np.longdouble) -> None:
        self.__M = M
        self.__e = e
        self.__solve_kepler()
        self.__true_anomaly()

    def __solve_kepler(self, tol: np.longdouble = 1e-8, max_iter: int = 100) \
            -> None:
        """
        Solves Kepler's equation for the eccentric anomaly E given mean
        anomaly M and eccentricity e using Newton-Raphson iteration.

        Parameters:
            tol (np.longdouble): Tolerance for convergence.
            max_iter (int): Maximum number of iterations.
        """
        self.__E = self.__M.copy()
        for _ in range(max_iter):
            fn: NDArray[np.longdouble] = (
                    self.__E - self.__e * np.sin(self.__E) - self.__M)
            fnprime: NDArray[np.longdouble] = 1 - self.__e * np.cos(self.__E)
            delta: NDArray[np.longdouble] = -fn / fnprime
            self.__E += delta
            if np.all(np.abs(delta) < tol):
                break

    def __true_anomaly(self) -> None:
        """
        Converts eccentric anomaly E to true anomaly f.
        """
        self.__f = 2 * np.arctan(np.sqrt((1 + self.__e) / (1 - self.__e))
                                  * np.tan(self.__E / 2))
        # self.__beta = self.__e / (1 + np.sqrt(1 - np.square(self.__e)))
        # self.__f = self.__E + 2 * np.arctan2(self.__beta * np.sin(self.__E),
        #                                      1 - self.__beta * np.cos(self.__E))

    @property
    def M(self) -> NDArray[np.longdouble]:
        return self.__M

    @property
    def e(self) -> np.longdouble:
        return self.__e

    @property
    def E(self) -> NDArray[np.longdouble]:
        return self.__E

    @property
    def f(self) -> NDArray[np.longdouble]:
        return self.__f

    @property
    def beta(self) -> np.longdouble:
        return self.__beta
