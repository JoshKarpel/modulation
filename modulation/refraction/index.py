import logging
from typing import Optional, Union

import abc

import simulacra as si
import simulacra.units as u

import numpy as np

logger = logging.getLogger(__name__)

FloatOrArray = Union[float, np.ndarray]


class IndexOfRefraction(abc.ABC):
    """An interface for an index of refraction."""

    def __init__(self, name: Optional[str] = None):
        """
        Parameters
        ----------
        name
            An optional human-readable name for the index of refraction (i.e., the name of the material).
        """
        self.name = name

    @abc.abstractmethod
    def __call__(self, wavelength: FloatOrArray, temperature=None):
        """Calculate the index of refraction at the given wavelength."""
        raise NotImplementedError

    def info(self) -> si.Info:
        info = si.Info(header=self.__class__.__name__)
        if self.name is not None:
            info.add_field("Name", self.name)
        return info


class ConstantIndex(IndexOfRefraction):
    """A wavelength-independent index of refraction."""

    def __init__(self, n=1, **kwargs):
        self.n = n

        super().__init__(**kwargs)

    def __eq__(self, other):
        return self.n == other.n

    def __call__(self, wavelength, temperature=None):
        return self.n * np.ones_like(wavelength)

    def __str__(self):
        return f"{self.name or self.__class__.__name__}(n = {self.n})"

    @property
    def tex(self):
        return fr"{self.name or self.__class__.__name__}($n = {self.n}$)"


class SellmeierIndex(IndexOfRefraction):
    """An index of refraction that is calculated using the Sellmeier equation."""

    def __init__(
        self, B, C, thermo_optic_coefficient=None, base_temperature=None, **kwargs
    ):
        self.B = B
        self.C = C

        self.base_temperature = base_temperature
        self.thermo_optic_coefficient = thermo_optic_coefficient

        super().__init__(**kwargs)

    @classmethod
    def from_name(cls, name: str) -> "SellmeierIndex":
        """

        Construct a :class:`SellmeierIndex` from a set of known materials.
        See the ``SELLMEIER_COEFFICIENTS`` constant in this module.

        Parameters
        ----------
        name
            The name of the material.
            See the ``SELLMEIER_COEFFICIENTS`` constant in this module for available names.

        Returns
        -------
        index
        """
        B, C = SELLMEIER_COEFFICIENTS[name]
        thermo_optic_coefficient = THERMO_OPTIC_COEFFICIENTS.get(name, None)
        base_temperature = TEMPERATURES.get(name, None)

        return cls(
            B,
            C,
            thermo_optic_coefficient=thermo_optic_coefficient,
            base_temperature=base_temperature,
            name=name,
        )

    def __eq__(self, other):
        return all(
            (
                np.array_equal(self.B, other.B),
                np.array_equal(self.C, other.C),
                self.base_temperature == other.base_temperature,
                self.thermo_optic_coefficient == other.thermo_optic_coefficient,
            )
        )

    def __hash__(self):
        return hash((tuple(self.B), tuple(self.C)))

    def __call__(self, wavelength: FloatOrArray, temperature=None):
        wavelength_sq = wavelength ** 2
        mod = sum(
            b * wavelength_sq / (wavelength_sq - c) for b, c in zip(self.B, self.C)
        )
        n = np.sqrt(1 + mod)

        if temperature is not None:
            n += self.thermo_optic_coefficient * (temperature - self.base_temperature)

        return n

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.name or ""}, B = {self.B}, C = {self.C})'
        )

    def __str__(self):
        return self.name or self.__class__.__name__

    @property
    def tex(self):
        return str(self)

    def info(self) -> si.Info:
        info = super().info()

        info.add_field("B", self.B)
        info.add_field("C", self.C)
        info.add_field("Base Temperature", f"{self.base_temperature} K")
        info.add_field("Thermo-Optic Coefficient", self.thermo_optic_coefficient)

        return info


silica = (
    np.array([0.696_166_3, 0.407_942_6, 0.897_479_4]),
    (np.array([0.068_404_3, 0.116_241_4, 9.896_161]) ** 2) * (u.um ** 2),
)

SELLMEIER_COEFFICIENTS = {
    "FS": silica,  # i.e., fused silica
    "SiO2": silica,
    "silica": silica,
    "BK7": (
        np.array([1.039_612_12, 0.231_792_344, 1.010_469_45]),
        np.array([6.000_698_67e-3, 2.001_791_44e-2, 103.560_653]) * (u.um ** 2),
    ),
    "MgF2-o": (
        np.array([0.487_551_08, 0.398_750_31, 2.312_035_3]),
        (np.array([0.043_384_08, 0.094_614_42, 23.793_604]) ** 2) * (u.um ** 2),
    ),
    "MgF2-e": (
        np.array([0.413_440_23, 0.504_974_99, 2.490_486_2]),
        (np.array([0.036_842_62, 0.090_761_62, 23.771_995]) ** 2) * (u.um ** 2),
    ),
    "CaF2": (
        np.array([0.443_749_998, 0.444_930_066, 0.150_133_991, 8.853_199_46]),
        np.array([0.001_780_278_54, 0.007_885_360_61, 0.012_411_949_1, 2752.28175])
        * (u.um ** 2),
    ),
}

silica_thermo_optic = 1.090e-5
THERMO_OPTIC_COEFFICIENTS = {
    "FS": silica_thermo_optic,  # i.e., fused silica
    "SiO2": silica_thermo_optic,
    "silica": silica_thermo_optic,
}

silica_temperature = u.zero_celsius + (20 * u.K)  # 20 Celsius
TEMPERATURES = {
    "FS": silica_temperature,  # i.e., fused silica
    "SiO2": silica_temperature,
    "silica": silica_temperature,
}
