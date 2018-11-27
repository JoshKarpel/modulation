import logging

import numpy as np
import simulacra as si
from scipy import special as spc
from simulacra import units as u

from . import vsh

from modulation.refraction import index
from modulation.raman import Mode

logger = logging.getLogger(__name__)


class MicrosphereModePolarization(si.utils.StrEnum):
    """An enumeration for the types of electromagnetic modes in a spherical microresonator."""

    TRANSVERSE_ELECTRIC = 'TE'
    TRANSVERSE_MAGNETIC = 'TM'

    def __str__(self):
        return self.value


class MicrosphereMode(Mode):
    def __init__(
        self,
        wavelength: float,
        l: int,
        m: int,
        radial_mode_number: int,
        index_of_refraction: index.IndexOfRefraction,
        microsphere_radius: float,
        amplitude: float = 1 * u.V / u.m,
        polarization: MicrosphereModePolarization = MicrosphereModePolarization.TRANSVERSE_ELECTRIC,
    ):
        self.l = l
        self.m = m

        self.wavelength = wavelength
        self.radial_mode_number = radial_mode_number
        self._index_of_refraction = index_of_refraction
        self.microsphere_radius = microsphere_radius

        self.amplitude = amplitude

        self.polarization = polarization

    @classmethod
    def from_mode_location(
        cls,
        mode_location,
        m: int,
        index_of_refraction: index.IndexOfRefraction,
        microsphere_radius: float,
        amplitude: float = 1 * u.V / u.m,
    ):
        return cls(
            l = mode_location.l,
            m = m,
            wavelength = mode_location.wavelength,
            radial_mode_number = mode_location.radial_mode_number,
            polarization = mode_location.polarization,
            index_of_refraction = index_of_refraction,
            microsphere_radius = microsphere_radius,
            amplitude = amplitude,
        )

    def __repr__(self):
        return f'{self.__class__.__name__}(λ = {u.uround(self.wavelength, u.nm)} nm, ℓ = {self.l}, m = {self.m}, radial_mode_number = {self.radial_mode_number}, polarization = {self.polarization})'

    @property
    def tex(self):
        return fr'u^{{{self.polarization}, \ell={self.l}, m = {self.m}, n = {self.radial_mode_number}}}_{{\lambda={self.wavelength / u.nm:.3f} \, \mathrm{{nm}}, f={self.frequency / u.THz:.3f} \, \mathrm{{THz}}}}'

    @property
    def omega(self):
        # this is correct because stored wavelength is for free space
        return u.twopi * u.c / self.wavelength

    @property
    def frequency(self):
        return self.omega / u.twopi

    @property
    def k_inside(self):
        return self.k_outside * self.index_of_refraction

    @property
    def k_outside(self):
        return u.twopi / self.wavelength

    @property
    def mode_volume_within_R(self):
        pre = (self.microsphere_radius ** 3) / 2

        kr = self.k_inside * self.microsphere_radius
        bessels = (spc.spherical_jn(self.l, kr) ** 2) - (spc.spherical_jn(self.l - 1, kr) * spc.spherical_jn(self.l + 1, kr))

        result = pre * bessels

        return result

    @property
    def mode_volume_outside_R(self):
        L = 10 * self.microsphere_radius
        kl = self.k_outside * L
        kr = self.k_outside * self.microsphere_radius

        l_term = ((L ** 3) / 2) * ((spc.spherical_yn(self.l, kl) ** 2) - (spc.spherical_yn(self.l - 1, kl) * spc.spherical_yn(self.l + 1, kl)))
        r_term = ((self.microsphere_radius ** 3) / 2) * ((spc.spherical_yn(self.l, kr) ** 2) - (spc.spherical_yn(self.l - 1, kr) * spc.spherical_yn(self.l + 1, kr)))

        result = (self.inside_outside_amplitude_ratio ** 2) * (l_term - r_term)

        return result

    @property
    def mode_volume(self):
        return self.mode_volume_within_R + self.mode_volume_outside_R

    @property
    def inside_outside_amplitude_ratio(self):
        """outside = inside * inside_outside_amplitude_ratio"""
        bessel_ratio = spc.spherical_jn(self.l, self.k_inside * self.microsphere_radius) / spc.spherical_yn(self.l, self.k_outside * self.microsphere_radius)
        if self.polarization is MicrosphereModePolarization.TRANSVERSE_ELECTRIC:
            return bessel_ratio
        elif self.polarization is MicrosphereModePolarization.TRANSVERSE_MAGNETIC:
            return (self.k_inside / self.k_outside) * bessel_ratio

    @property
    def index_of_refraction(self):
        """The index of refraction at the mode's free-space wavelength."""
        return self._index_of_refraction(self.wavelength)

    def evaluate_electric_field_mode_shape_inside(self, r, theta, phi):
        if self.polarization is MicrosphereModePolarization.TRANSVERSE_ELECTRIC:
            sph = vsh.VectorSphericalHarmonic(
                type = vsh.VectorSphericalHarmonicType.CROSS,
                l = self.l,
                m = self.m
            )(theta, phi)

            rad_in = spc.spherical_jn(self.l, self.k_inside * r)

            inside = rad_in[..., np.newaxis] * sph

            return inside

        elif self.polarization is MicrosphereModePolarization.TRANSVERSE_MAGNETIC:
            grad_sph = vsh.VectorSphericalHarmonic(
                type = vsh.VectorSphericalHarmonicType.GRADIENT,
                l = self.l,
                m = self.m,
            )(theta, phi)

            grad_rad_in = spc.spherical_jn(self.l, self.k_inside * r) / (self.k_inside * r)
            grad_rad_in += spc.spherical_jn(self.l, self.k_inside * r, derivative = True) / self.k_inside

            radial_sph = np.sqrt(self.l * (self.l + 1)) * vsh.VectorSphericalHarmonic(
                type = vsh.VectorSphericalHarmonicType.RADIAL,
                l = self.l,
                m = self.m,
            )(theta, phi)

            radial_rad_in = spc.spherical_jn(self.l, self.k_inside * r) / (self.k_inside * r)

            inside = (grad_rad_in[..., np.newaxis] * grad_sph) + (radial_rad_in[..., np.newaxis] * radial_sph)

            return inside

    def evaluate_electric_field_mode_shape_outside(self, r, theta, phi):
        if self.polarization is MicrosphereModePolarization.TRANSVERSE_ELECTRIC:
            sph = vsh.VectorSphericalHarmonic(
                type = vsh.VectorSphericalHarmonicType.CROSS,
                l = self.l,
                m = self.m
            )(theta, phi)

            rad_out = spc.spherical_yn(self.l, self.k_outside * r)

            outside = self.inside_outside_amplitude_ratio * rad_out[..., np.newaxis] * sph

            return outside

        elif self.polarization is MicrosphereModePolarization.TRANSVERSE_MAGNETIC:
            grad_sph = vsh.VectorSphericalHarmonic(
                type = vsh.VectorSphericalHarmonicType.GRADIENT,
                l = self.l,
                m = self.m,
            )(theta, phi)

            grad_rad_in = spc.spherical_jn(self.l, self.k_inside * r) / (self.k_inside * r)
            grad_rad_in += spc.spherical_jn(self.l, self.k_inside * r, derivative = True) / self.k_inside

            grad_rad_out = spc.spherical_yn(self.l, self.k_outside * r) / (self.k_inside * r)
            grad_rad_out += spc.spherical_yn(self.l, self.k_outside * r, derivative = True) / self.k_inside

            radial_sph = np.sqrt(self.l * (self.l + 1)) * vsh.VectorSphericalHarmonic(
                type = vsh.VectorSphericalHarmonicType.RADIAL,
                l = self.l,
                m = self.m,
            )(theta, phi)

            radial_rad_out = spc.spherical_yn(self.l, self.k_outside * r) / (self.k_inside * r)

            outside = self.inside_outside_amplitude_ratio * ((grad_rad_out[..., np.newaxis] * grad_sph) + (radial_rad_out[..., np.newaxis] * radial_sph))

            return outside

    def __call__(self, r, theta, phi):
        inside = self.evaluate_electric_field_mode_shape_inside(r, theta, phi)
        outside = self.evaluate_electric_field_mode_shape_inside(r, theta, phi)
        return np.where(
            r[..., np.newaxis] <= self.microsphere_radius,
            inside,
            outside,
        )
