import logging

import numpy as np
import scipy.special as spc

import simulacra as si
from simulacra import units as u

from . import vsh

from modulation.raman import Mode

logger = logging.getLogger(__name__)


def find_mode_with_closest_wavelength(modes, wavelength):
    return sorted(modes, key=lambda m: abs(m.wavelength - wavelength))[0]


class MicrosphereModePolarization(si.utils.StrEnum):
    """An enumeration for the types of electromagnetic modes in a spherical microresonator."""

    TRANSVERSE_ELECTRIC = "TE"
    TRANSVERSE_MAGNETIC = "TM"


class MicrosphereMode(Mode):
    def __init__(
        self,
        wavelength: float,
        l: int,
        m: int,
        radial_mode_number: int,
        microsphere,
        polarization: MicrosphereModePolarization = MicrosphereModePolarization.TRANSVERSE_ELECTRIC,
    ):
        self.l = l
        self.m = m

        self.wavelength = wavelength
        self.radial_mode_number = radial_mode_number
        self.microsphere = microsphere
        self.polarization = polarization

    @classmethod
    def from_mode_location(cls, mode_location, m: int):
        return cls(
            wavelength=mode_location.wavelength,
            l=mode_location.l,
            m=m,
            radial_mode_number=mode_location.radial_mode_number,
            microsphere=mode_location.microsphere,
            polarization=mode_location.polarization,
        )

    def __eq__(self, other):
        return all(
            (
                self.l == other.l,
                self.m == other.m,
                self.wavelength == other.wavelength,
                self.radial_mode_number == other.radial_mode_number,
                self.microsphere == other.microsphere,
                self.polarization == other.polarization,
            )
        )

    def __hash__(self):
        return hash(
            (
                self.__class__,
                self.l,
                self.m,
                self.wavelength,
                self.radial_mode_number,
                self.microsphere,
                self.polarization,
            )
        )

    def __str__(self):
        return f"{self.__class__.__name__}(λ = {self.wavelength / u.nm:.3f} nm, f = {self.frequency / u.THz:.3f} THz, l = {self.l}, m = {self.m}, radial_mode_number = {self.radial_mode_number}, polarization = {self.polarization})"

    @property
    def tex(self):
        return fr"u^{{{self.polarization}, \ell={self.l}, m = {self.m}, n = {self.radial_mode_number}}}_{{\lambda={self.wavelength / u.nm:.3f} \, \mathrm{{nm}}, f={self.frequency / u.THz:.3f} \, \mathrm{{THz}}}}"

    @property
    def omega(self):
        # this is correct because stored wavelength is for free space
        return u.twopi * u.c / self.wavelength

    @property
    def frequency(self):
        return self.omega / u.twopi

    @property
    def index_of_refraction(self):
        """The index of refraction at the mode's free-space wavelength."""
        return self.microsphere.index_of_refraction(self.wavelength)

    @property
    def k_inside(self):
        return self.k_outside * self.index_of_refraction

    @property
    def k_outside(self):
        return u.twopi / self.wavelength

    @property
    def mode_volume_inside_resonator(self):
        pre = (self.microsphere.radius ** 3) / 2

        kr = self.k_inside * self.microsphere.radius
        bessels = (spc.spherical_jn(self.l, kr) ** 2) - (
            spc.spherical_jn(self.l - 1, kr) * spc.spherical_jn(self.l + 1, kr)
        )

        result = pre * bessels

        return result

    @property
    def mode_volume_outside_resonator(self):
        L = 10 * self.microsphere.radius
        kl = self.k_outside * L
        kr = self.k_outside * self.microsphere.radius

        l_term = ((L ** 3) / 2) * (
            (spc.spherical_yn(self.l, kl) ** 2)
            - (spc.spherical_yn(self.l - 1, kl) * spc.spherical_yn(self.l + 1, kl))
        )
        r_term = ((self.microsphere.radius ** 3) / 2) * (
            (spc.spherical_yn(self.l, kr) ** 2)
            - (spc.spherical_yn(self.l - 1, kr) * spc.spherical_yn(self.l + 1, kr))
        )

        result = (self.inside_outside_amplitude_ratio ** 2) * (l_term - r_term)

        return result

    @property
    def mode_volume(self):
        return self.mode_volume_inside_resonator + self.mode_volume_outside_resonator

    @property
    def inside_outside_amplitude_ratio(self):
        """outside = inside * inside_outside_amplitude_ratio"""
        bessel_ratio = spc.spherical_jn(
            self.l, self.k_inside * self.microsphere.radius
        ) / spc.spherical_yn(self.l, self.k_outside * self.microsphere.radius)
        if self.polarization is MicrosphereModePolarization.TRANSVERSE_ELECTRIC:
            return bessel_ratio
        elif self.polarization is MicrosphereModePolarization.TRANSVERSE_MAGNETIC:
            return (self.k_inside / self.k_outside) * bessel_ratio

    def evaluate_electric_field_mode_shape_inside(self, r, theta, phi):
        if self.polarization is MicrosphereModePolarization.TRANSVERSE_ELECTRIC:
            sph = vsh.VectorSphericalHarmonic(
                type=vsh.VectorSphericalHarmonicType.CROSS, l=self.l, m=self.m
            )(theta, phi)

            rad_in = spc.spherical_jn(self.l, self.k_inside * r)

            return rad_in[..., np.newaxis] * sph

        elif self.polarization is MicrosphereModePolarization.TRANSVERSE_MAGNETIC:
            grad_sph = vsh.VectorSphericalHarmonic(
                type=vsh.VectorSphericalHarmonicType.GRADIENT, l=self.l, m=self.m
            )(theta, phi)

            grad_rad_in = spc.spherical_jn(self.l, self.k_inside * r) / (
                self.k_inside * r
            )
            grad_rad_in += (
                spc.spherical_jn(self.l, self.k_inside * r, derivative=True)
                / self.k_inside
            )

            radial_sph = np.sqrt(self.l * (self.l + 1)) * vsh.VectorSphericalHarmonic(
                type=vsh.VectorSphericalHarmonicType.RADIAL, l=self.l, m=self.m
            )(theta, phi)

            radial_rad_in = spc.spherical_jn(self.l, self.k_inside * r) / (
                self.k_inside * r
            )

            return (grad_rad_in[..., np.newaxis] * grad_sph) + (
                radial_rad_in[..., np.newaxis] * radial_sph
            )

    def evaluate_electric_field_mode_shape_outside(self, r, theta, phi):
        if self.polarization is MicrosphereModePolarization.TRANSVERSE_ELECTRIC:
            return (
                self.inside_outside_amplitude_ratio
                * spc.spherical_yn(self.l, self.k_outside * r)[..., np.newaxis]
                * vsh.VectorSphericalHarmonic(
                    type=vsh.VectorSphericalHarmonicType.CROSS, l=self.l, m=self.m
                )(theta, phi)
            )

        elif self.polarization is MicrosphereModePolarization.TRANSVERSE_MAGNETIC:
            grad_sph = vsh.VectorSphericalHarmonic(
                type=vsh.VectorSphericalHarmonicType.GRADIENT, l=self.l, m=self.m
            )(theta, phi)

            grad_rad_in = spc.spherical_jn(self.l, self.k_inside * r) / (
                self.k_inside * r
            )
            grad_rad_in += (
                spc.spherical_jn(self.l, self.k_inside * r, derivative=True)
                / self.k_inside
            )

            grad_rad_out = spc.spherical_yn(self.l, self.k_outside * r) / (
                self.k_inside * r
            )
            grad_rad_out += (
                spc.spherical_yn(self.l, self.k_outside * r, derivative=True)
                / self.k_inside
            )

            radial_sph = np.sqrt(self.l * (self.l + 1)) * vsh.VectorSphericalHarmonic(
                type=vsh.VectorSphericalHarmonicType.RADIAL, l=self.l, m=self.m
            )(theta, phi)

            radial_rad_out = spc.spherical_yn(self.l, self.k_outside * r) / (
                self.k_inside * r
            )

            return self.inside_outside_amplitude_ratio * (
                (grad_rad_out[..., np.newaxis] * grad_sph)
                + (radial_rad_out[..., np.newaxis] * radial_sph)
            )

    def __call__(self, r, theta, phi):
        return np.where(
            r[..., np.newaxis] <= self.microsphere.radius,
            self.evaluate_electric_field_mode_shape_inside(r, theta, phi),
            self.evaluate_electric_field_mode_shape_inside(r, theta, phi),
        )
