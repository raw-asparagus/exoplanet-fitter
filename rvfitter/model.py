from typing import List

import numpy as np
from numpy.typing import NDArray

from rvfitter import Planet


class Model:
    """
    Combines the RV signals of multiple planets and an optional systemic offset.

    Attributes:
        __planets (List[Planet]): A list of Planet objects.
        __v0 (np.longdouble)): Systemic velocity offset.
    """
    __planets: List[Planet]
    __v0: np.longdouble

    def __init__(self, planets: List[Planet] = None, v0: np.longdouble = None) \
            -> None:
        self.__planets = planets if planets is not None else []
        self.__v0 = v0 if v0 is not None else np.longdouble(0.0)

    def add_planet(self, planet: Planet) -> None:
        """
        Adds a Planet to the model.

        Parameters:
            planet (Planet): Planet to add.
        """
        self.__planets.append(planet)

    def evaluate(self, t: NDArray[np.longdouble]) -> NDArray[
        np.longdouble]:
        """
        Evaluate the combined radial velocity model at times t.

        Parameters:
            t (NDArray): Array of observation times.

        Returns:
            rv (NDArray): Radial velocity at each time t.
        """
        total_rv: NDArray[np.longdouble] = (
                np.zeros(t.shape[0], dtype=np.longdouble) + self.__v0)
        for planet in self.__planets:
            total_rv += planet.compute_radial_velocities(t)
        return total_rv

    @property
    def planets(self) -> List[Planet]:
        return self.__planets

    @property
    def v0(self) -> np.longdouble:
        return self.__v0
