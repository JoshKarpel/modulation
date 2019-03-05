import logging
from typing import Tuple

import functools
import operator

import numpy as np
from scipy import integrate as integ

import simulacra as si
from simulacra import units as u

from . import find
from ...raman import Mode, ModeVolumeIntegrator

logger = logging.getLogger(__name__)


class MicrosphereVolumeIntegrator(ModeVolumeIntegrator):
    def __init__(
        self,
        *,
        microsphere: find.Microsphere,
        pad: float = 0.8,
        threshold: float = 1e-15,
    ):
        self.radius = microsphere.radius
        self.pad = pad
        self.threshold = threshold

        self.phi = np.zeros(1)

    def mode_volume_integral(self, modes: Tuple[Mode, ...]) -> float:
        raise NotImplementedError

    def integrand(self, modes, r, theta, phi):
        return functools.reduce(
            operator.mul, (self.mode_magnitude_inside(m, r, theta, phi) for m in modes)
        )

    def mode_magnitude_inside(self, mode, r, theta, phi):
        mode_shape_vector_field = mode.evaluate_electric_field_mode_shape_inside(
            r, theta, phi
        )
        return np.sqrt(np.sum(np.abs(mode_shape_vector_field) ** 2, axis=-1))

    def jacobian(self, r, theta):
        return (r ** 2) * np.sin(theta)

    def estimate_r_lower_bound(self, modes):
        r = np.linspace(0, self.radius, 1000)[1:]
        theta = np.ones(1) * u.pi / 2

        r_mesh, theta_mesh, phi_mesh = np.meshgrid(r, theta, self.phi, indexing="ij")

        integrand = self.integrand(modes, r_mesh, theta_mesh, phi_mesh)
        integrand_max = np.max(integrand)

        index_of_threshold_r = np.argmax(integrand > (integrand_max * self.threshold))
        threshold_r = r[index_of_threshold_r]

        inner_limit = max(0, self.pad * threshold_r)

        return inner_limit

    def estimate_theta_bound(self, modes):
        r = np.ones(1) * 0.95 * self.radius
        theta = np.linspace(0, u.pi, 1000)[1:-1]  # avoid poles

        r_mesh, theta_mesh, phi_mesh = np.meshgrid(r, theta, self.phi, indexing="ij")

        integrand = self.integrand(modes, r_mesh, theta_mesh, phi_mesh)
        integrand_max = np.max(integrand)

        index_of_threshold_theta = np.argmax(
            integrand > (integrand_max * self.threshold)
        )
        threshold_theta = theta[index_of_threshold_theta]

        limit = max(0, self.pad * threshold_theta)

        return limit


class RiemannSumMicrosphereVolumeIntegrator(MicrosphereVolumeIntegrator):
    def __init__(self, r_points=500, theta_points=200, **kwargs):
        super().__init__(**kwargs)

        self.r_points = r_points
        self.theta_points = theta_points

    def mode_volume_integral(self, modes: Tuple[Mode, ...]) -> float:
        r_lower_bound = self.estimate_r_lower_bound(modes)
        theta_bound = self.estimate_theta_bound(modes)

        r = np.linspace(r_lower_bound, self.radius, self.r_points + 1)[
            1:
        ]  # in case lower bound is zero
        theta = np.linspace(theta_bound, u.pi - theta_bound, self.theta_points + 2)[
            1:-1
        ]  # in case bound is zero

        dr = np.abs(r[1] - r[0])
        dtheta = np.abs(theta[1] - theta[0])

        r_mesh, theta_mesh, phi_mesh = np.meshgrid(
            r, theta, self.phi, indexing="ij", sparse=True
        )

        integrand = self.integrand(modes, r_mesh, theta_mesh, phi_mesh)

        jac = self.jacobian(r_mesh, theta_mesh)

        result = u.twopi * np.sum(integrand * jac) * dr * dtheta

        return result


class FlexibleGridSimpsonMicrosphereVolumeIntegrator(
    RiemannSumMicrosphereVolumeIntegrator
):
    def __init__(self, r_points=501, theta_points=201, **kwargs):
        super().__init__(r_points=r_points, theta_points=theta_points, **kwargs)

    def mode_volume_integral(self, modes: Tuple[Mode, ...]) -> float:
        r_lower_bound = self.estimate_r_lower_bound(modes)
        theta_bound = self.estimate_theta_bound(modes)

        r = np.linspace(r_lower_bound, self.radius, self.r_points + 1)[
            1:
        ]  # in case lower bound is zero
        theta = np.linspace(theta_bound, u.pi - theta_bound, self.theta_points + 2)[
            1:-1
        ]  # in case bound is zero

        dr = np.abs(r[1] - r[0])
        dtheta = np.abs(theta[1] - theta[0])

        r_mesh, theta_mesh, phi_mesh = np.meshgrid(r, theta, self.phi, indexing="ij")

        integrand = self.integrand(modes, r_mesh, theta_mesh, phi_mesh)

        jac = self.jacobian(r_mesh, theta_mesh)

        integral = integ.simps(y=integ.simps(y=integrand * jac, axis=0), axis=0)[0]

        result = integral * u.twopi * dr * dtheta

        return result


class RombergMicrosphereVolumeIntegrator(RiemannSumMicrosphereVolumeIntegrator):
    def __init__(self, k_r=9, k_theta=8, **kwargs):
        super().__init__(
            r_points=(2 ** k_r) + 1, theta_points=(2 ** k_theta) + 1, **kwargs
        )

    def mode_volume_integral(self, modes: Tuple[Mode, ...]) -> float:
        r_lower_bound = self.estimate_r_lower_bound(modes)
        theta_bound = self.estimate_theta_bound(modes)

        r = np.linspace(r_lower_bound, self.radius, self.r_points + 1)[
            1:
        ]  # in case lower bound is zero
        theta = np.linspace(theta_bound, u.pi - theta_bound, self.theta_points + 2)[
            1:-1
        ]  # in case bound is zero

        dr = np.abs(r[1] - r[0])
        dtheta = np.abs(theta[1] - theta[0])

        r_mesh, theta_mesh, phi_mesh = np.meshgrid(r, theta, self.phi, indexing="ij")

        integrand = self.integrand(modes, r_mesh, theta_mesh, phi_mesh)

        jac = self.jacobian(r_mesh, theta_mesh)

        integral = integ.romb(y=integ.romb(y=integrand * jac, axis=0), axis=0)[0]

        result = integral * u.twopi * dr * dtheta

        return result


class FixedGridSimpsonMicrosphereVolumeIntegrator(
    FlexibleGridSimpsonMicrosphereVolumeIntegrator
):
    """
    A Simpson volume integrator that operates on a fixed grid.
    Because the grid is fixed, it and the mode magnitudes evaluated on it can be cached.
    """

    def __init__(self, r_points=2001, theta_points=501, **kwargs):
        super().__init__(r_points=r_points, theta_points=theta_points, **kwargs)

    @property
    def r_theta(self):
        r = np.linspace(0.5 * self.radius, self.radius, self.r_points)
        theta = np.linspace(0, u.pi, self.theta_points + 2)[1:-1]
        return r, theta

    @si.utils.cached_property
    def r_theta_phi_meshes(self):
        r, theta = self.r_theta
        r_mesh, theta_mesh, phi_mesh = np.meshgrid(
            r, theta, self.phi, indexing="ij", sparse=True
        )
        return r_mesh, theta_mesh, phi_mesh

    @si.utils.cached_property
    def fixed_jacobian(self):
        r, theta = self.r_theta
        dr = np.abs(r[1] - r[0])
        dtheta = np.abs(theta[1] - theta[0])
        r_mesh, theta_mesh, _ = self.r_theta_phi_meshes
        return self.jacobian(r_mesh, theta_mesh) * (u.twopi * dr * dtheta)

    def mode_volume_integral(self, modes: Tuple[Mode, ...]) -> float:
        integral = integ.simps(
            y=integ.simps(y=self.integrand(modes) * self.fixed_jacobian, axis=0), axis=0
        )[0]

        result = integral

        return result

    def integrand(self, modes):
        return functools.reduce(
            operator.mul, (self.mode_magnitude_inside(m) for m in modes)
        )

    @functools.lru_cache(maxsize=None)
    def mode_magnitude_inside(self, mode):
        mode_shape_vector_field = mode.evaluate_electric_field_mode_shape_inside(
            self.r_mesh, self.theta_mesh, self.phi_mesh
        )
        return np.sqrt(np.sum(np.abs(mode_shape_vector_field) ** 2, axis=-1))
