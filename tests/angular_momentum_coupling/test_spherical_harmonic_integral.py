import pytest

import simulacra as si
import simulacra.units as u

import numpy as np

from whisper import vsh


def integrand(spherical_harmonics, theta, phi):
    acc = 1
    for sph_harm in spherical_harmonics:
        acc *= sph_harm(theta, phi)

    return acc * np.sin(theta)


def dblquad(*spherical_harmonics):
    result = si.math.complex_nquad(
        lambda theta, phi: integrand(spherical_harmonics, theta, phi),
        ranges = (
            (0, u.pi),
            (0, u.twopi),
        ),
        opts = dict(
            limit = 5000,
        ),
    )

    return np.real(result[0])


L_MAX = 2


@pytest.mark.parametrize(
    'sph_harms',
    (
        (
            vsh.RecurrentSphericalHarmonic(l1, m1),
            vsh.RecurrentSphericalHarmonic(l2, m2),
            vsh.RecurrentSphericalHarmonic(l3, m3),
            vsh.RecurrentSphericalHarmonic(l, m),
        )
        for l1 in range(0, L_MAX + 1)
        for l2 in range(0, L_MAX + 1)
        for l3 in range(0, L_MAX + 1)
        for l in range(0, L_MAX + 1)
        for m1 in range(-l1, l1 + 1)
        for m2 in range(-l2, l2 + 1)
        for m3 in range(-l3, l3 + 1)
        for m in range(-l, l + 1)
    )
)
def test_sph_harm_integral_via_sum_against_dblquad_all_valid(sph_harms):
    from_sum = vsh.four_sph_harm_integral(*sph_harms)
    from_dbl = dblquad(*sph_harms)

    assert np.isclose(from_sum, from_dbl, atol = 1e-12, rtol = 1e-12)
