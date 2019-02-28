import numpy as np

import simulacra.units as u

from ... import refraction


def coupling_quality_factor_for_tapered_fiber(
    microsphere_index_of_refraction: refraction.IndexOfRefraction,
    fiber_index_of_refraction: refraction.IndexOfRefraction,
    microsphere_radius: float,
    fiber_taper_radius: float,
    wavelength: float,
    separation: float,
    l: int,
    m: int,
) -> float:
    # from Gorodetsky1999

    n_sphere = microsphere_index_of_refraction(wavelength)
    n_fiber = fiber_index_of_refraction(wavelength)

    gamma = (u.twopi / wavelength) * np.sqrt((n_sphere ** 2) - 1)

    q = 16 * np.sqrt(2) * (u.pi ** 5)
    q *= (
        (n_sphere ** 4)
        * n_fiber
        * (((n_sphere ** 2) - 1) ** 2)
        / (9 * ((n_fiber ** 2) - 1))
    )
    q *= (microsphere_radius ** 1.5) * (fiber_taper_radius ** 3) / (wavelength ** 4.5)
    q *= np.exp(
        (2 * gamma * separation) + (((l - m) ** 2) / (gamma * microsphere_radius))
    )

    return q
