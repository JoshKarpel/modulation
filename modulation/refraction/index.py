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
    def __call__(self, wavelength: FloatOrArray):
        """Calculate the index of refraction at the given wavelength."""
        raise NotImplementedError

    def info(self) -> si.Info:
        info = si.Info(header = self.__class__.__name__)
        if self.name is not None:
            info.add_field('Name', self.name)
        return info


class ConstantIndex(IndexOfRefraction):
    """A wavelength-independent index of refraction."""

    def __init__(self, n = 1, **kwargs):
        self.n = n

        super().__init__(**kwargs)

    def __call__(self, wavelength):
        return self.n * np.ones_like(wavelength)

    def __str__(self):
        return f'{self.name or self.__class__.__name__}(n = {self.n})'

    @property
    def tex(self):
        return fr'{self.name or self.__class__.__name__}($n = {self.n}$)'


class SellmeierIndex(IndexOfRefraction):
    """An index of refraction that is calculated using the Sellmeier equation."""

    def __init__(self, B, C, **kwargs):
        self.B = B
        self.C = C

        super().__init__(**kwargs)

    @classmethod
    def from_name(cls, name: str) -> 'SellmeierIndex':
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
        return cls(B, C, name = name)

    def __call__(self, wavelength: FloatOrArray):
        wavelength_sq = wavelength ** 2
        mod = sum(b * wavelength_sq / (wavelength_sq - c) for b, c in zip(self.B, self.C))
        return np.sqrt(1 + mod)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.name or ""}, B = {self.B}, C = {self.C})'

    def __str__(self):
        return self.name or self.__class__.__name__

    @property
    def tex(self):
        return str(self)


SELLMEIER_COEFFICIENTS = {
    'BK7': (
        np.array([1.03961212, 0.231792344, 1.01046945]),
        np.array([6.00069867e-3, 2.00179144e-2, 103.560653]) * (u.um ** 2),
    ),
    'FS': (  # fused silica
        np.array([0.696166300, 0.407942600, 0.897479400]),
        (np.array([0.0684043, 0.1162414, 9.896161]) ** 2) * (u.um ** 2),
    ),
    'MgF2-o': (
        np.array([0.48755108, 0.39875031, 2.3120353]),
        (np.array([0.04338408, 0.09461442, 23.793604]) ** 2) * (u.um ** 2),
    ),
    'MgF2-e': (
        np.array([0.41344023, 0.50497499, 2.4904862]),
        (np.array([0.03684262, 0.09076162, 23.771995]) ** 2) * (u.um ** 2),
    ),
    'CaF2': (
        np.array([0.443749998, 0.444930066, 0.150133991, 8.85319946]),
        np.array([0.00178027854, 0.00788536061, 0.0124119491, 2752.28175]) * (u.um ** 2),
    ),
    'SiO2': (
        np.array([0.6961663, 0.4079426, 0.8974794]),
        (np.array([0.0684043, 0.1162414, 9.896161]) ** 2) * (u.um ** 2),
    )
}