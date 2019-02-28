import logging

from enum import IntEnum
import functools

import numpy as np

import simulacra as si
import simulacra.units as u

from .threej import threej

logger = logging.getLogger(__name__)


class CartesianComponent(IntEnum):
    """An enumeration for the components of a vector in Cartesian coordinates."""

    X = 0
    Y = 1
    Z = 2


class SphericalComponent(IntEnum):
    """An enumeration for the components of a vector in spherical coordinates."""

    R = 0
    THETA = 1
    PHI = 2


class VectorSphericalHarmonicType(si.utils.StrEnum):
    """An enumeration for the possible types of a vector spherical harmonic."""

    RADIAL = "radial"
    GRADIENT = "gradient"
    CROSS = "cross"


class RecurrentSphericalHarmonic(si.math.SphericalHarmonic):
    @functools.lru_cache(maxsize=None)
    def a(self, l, m):
        l_sq = l ** 2
        return np.sqrt(((4 * l_sq) - 1) / (l_sq - (m ** 2)))

    @functools.lru_cache(maxsize=None)
    def b(self, l, m):
        lm_sq = (l - 1) ** 2
        return -np.sqrt((lm_sq - (m ** 2)) / ((4 * lm_sq) - 1))

    def __call__(self, theta, phi=0):
        if isinstance(phi, float):
            phi = phi * np.ones_like(theta)
        P = np.ones_like(theta) / np.sqrt(u.twopi)  # P(l = 0, m = 0)

        sin_theta = np.sin(theta)
        abs_m = abs(self.m)
        for curr_m in range(1, abs_m + 1):
            P *= -np.sqrt(1 + (0.5 / curr_m)) * sin_theta

        # # recurse up to l = l, m = m (if l = m, we're done)
        if abs_m != self.l:
            cos_theta = np.cos(theta)
            P, prev_P = np.sqrt((2 * abs_m) + 3) * cos_theta * P, P

            # if l = m + 1, we're done
            for curr_l in range(abs_m + 2, self.l + 1):
                P, prev_P = (
                    self.a(curr_l, abs_m)
                    * ((cos_theta * P) + (self.b(curr_l, abs_m) * prev_P)),
                    P,
                )

        # turn normalized legendre polynomial into normalized spherical harmonic
        Y = P * np.exp(1j * self.m * phi) / np.sqrt(2)
        if self.m < 0 and self.m % 2 != 0:  # account for negative m
            Y *= -1

        return Y


@functools.lru_cache(maxsize=None)
def four_sph_harm_integral(*spherical_harmonics: si.math.SphericalHarmonic):
    """All spherical harmonics unconjugated"""
    if len(spherical_harmonics) != 4:
        raise Exception("must have 4 spherical harmonics")

    l1, l2, l3, l = (sh.l for sh in spherical_harmonics)
    m1, m2, m3, m = (sh.m for sh in spherical_harmonics)

    if m1 + m2 + m3 + m != 0:
        return 0

    sign = 1 if (m + m3) % 2 == 0 else -1
    prefactor = np.sqrt(
        ((2 * l1) + 1) * ((2 * l2) + 1) * ((2 * l3) + 1) * ((2 * l) + 1)
    ) / (4 * u.pi)

    lp_min = max(abs(l1 - l2), abs(l3 - l), abs(m1 + m2))
    lp_max = min(l1 + l2, l3 + l)

    acc = sum(
        ((2 * lp) + 1)
        * (
            threej(lp, l1, l2, -(m1 + m2), m1, m2)
            * threej(lp, l1, l2, 0, 0, 0)
            * threej(lp, l3, l, m1 + m2, m3, -(m1 + m2 + m3))
            * threej(lp, l3, l, 0, 0, 0)
        )
        for lp in range(lp_min, lp_max + 1)
    )

    return sign * prefactor * acc


class VectorSphericalHarmonic:
    """A class that represents a vector spherical harmonic."""

    __slots__ = ("type", "l", "m")

    def __init__(self, *, type: VectorSphericalHarmonicType, l: int = 0, m: int = 0):
        self.type = type
        self.l = l
        self.m = m

    @property
    def _lm(self):
        return self.l, self.m

    def __le__(self, other):
        return self._lm <= other._lm

    def __ge__(self, other):
        return self._lm >= other._lm

    def __lt__(self, other):
        return self._lm < other._lm

    def __gt__(self, other):
        return self._lm > other._lm

    def __eq__(self, other):
        return self._lm == other._lm and self.type == other.type

    def __call__(self, theta, phi) -> np.array:
        """Always build arguments using meshgrid!"""
        vf = np.zeros(
            (*theta.shape, 3), dtype=np.complex128
        )  # (positions, vector component)

        sph_harm_lm = RecurrentSphericalHarmonic(self.l, self.m)
        if self.type is VectorSphericalHarmonicType.RADIAL:
            vf[..., SphericalComponent.R] = sph_harm_lm(theta, phi)
        else:
            if self.l == 0:
                return vf
            sph_harm = self.m * sph_harm_lm(theta, phi)
            dy_dtheta = sph_harm / np.tan(theta)
            dy_dphi = 1j * sph_harm / np.sin(theta)
            try:
                sph_harm_lmp = RecurrentSphericalHarmonic(self.l, self.m + 1)
                dy_dtheta += (
                    np.sqrt((self.l - self.m) * (self.l + self.m + 1))
                    * np.exp(-1j * phi)
                    * sph_harm_lmp(theta, phi)
                )
            except si.exceptions.SimulacraException:
                pass

            norm = 1 / np.sqrt(self.l * (self.l + 1))
            dy_dtheta *= norm
            dy_dphi *= norm

            if self.type is VectorSphericalHarmonicType.GRADIENT:
                vf[..., SphericalComponent.THETA] = dy_dtheta
                vf[..., SphericalComponent.PHI] = dy_dphi
            elif self.type is VectorSphericalHarmonicType.CROSS:
                vf[..., SphericalComponent.THETA] = dy_dphi
                vf[..., SphericalComponent.PHI] = -dy_dtheta

        return vf

    def info(self) -> si.Info:
        info = si.Info(header="Vector Spherical Harmonic")

        info.add_field("Type", self.type)
        info.add_field("l", self.l)
        info.add_field("m", self.l)

        return info


def inner_product_of_vsh(a, b):
    """Point-wise inner product of two vector spherical harmonics."""
    return np.einsum("ijk,ijk->ij", a, b)
