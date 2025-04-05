import numpy as np
from numpy.typing import NDArray

from rvfitter import Kepler


class Planet:
    """
    Attributes:
        __period (np.longdouble): Orbital period (in days).
        __t_peri (np.longdouble): Time of periastron passage.
        __k (np.longdouble): RV semi-amplitude (in m/s).
        __e (np.longdouble): Eccentricity of orbit.
        __omega (np.longdouble): Argument of periastron (in radians).
    """
    __period: np.longdouble
    __t_peri: np.longdouble
    __k: np.longdouble
    __e: np.longdouble
    __omega: np.longdouble

    def __init__(self, period: np.longdouble, t_peri: np.longdouble,
                 k: np.longdouble,
                 e: np.longdouble, omega: np.longdouble) -> None:
        self.__period = period
        self.__t_peri = t_peri
        self.__k = k
        self.__e = e
        self.__omega = omega

    def compute_radial_velocities(self, t: NDArray[np.longdouble]) \
            -> NDArray[np.longdouble]:
        """
        Compute the Keplerian radial velocity at times t.

        Parameters:
            t (NDArray[np.longdouble]): Times at which to compute the RV.

        Returns:
            rv (NDArray[np.longdouble]): Radial velocity at each time t.
        """
        M: NDArray[np.longdouble] = (
                2 * np.pi * (t - self.__t_peri) / self.__period)
        kepler: Kepler = Kepler(M, self.__e)
        rv: NDArray[np.longdouble] = (
                self.__k * np.cos(kepler.f + self.__omega) + self.__e *
                np.cos(self.__omega))
        return rv

    @property
    def period(self) -> np.longdouble:
        return self.__period

    @property
    def t_peri(self) -> np.longdouble:
        return self.__t_peri

    @property
    def k(self) -> np.longdouble:
        return self.__k

    @property
    def e(self) -> np.longdouble:
        return self.__e

    @property
    def omega(self) -> np.longdouble:
        return self.__omega
