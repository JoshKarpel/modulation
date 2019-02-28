import pytest

import numpy as np

import simulacra as si
import simulacra.units as u

import whisper.vsh as vsh

theta = np.linspace(0, u.pi, 100)[1:-1]
phi = np.linspace(0, u.twopi, 100)[1:]
theta_mesh, phi_mesh = np.meshgrid(theta, phi, indexing="ij")


@pytest.mark.parametrize("l, m", [(l, m) for l in range(21) for m in range(-l, l + 1)])
def test_recurrence_sph_harm_agrees_with_direct_sph_harm_for_small_lm(l, m):
    direct = si.math.SphericalHarmonic(l, m)(theta_mesh, phi_mesh)
    recurr = vsh.RecurrentSphericalHarmonic(l, m)(theta_mesh, phi_mesh)

    assert np.allclose(direct, recurr, rtol=1e-12, atol=1e-12)
