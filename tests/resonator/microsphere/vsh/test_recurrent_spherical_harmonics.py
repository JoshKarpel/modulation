import pytest

import numpy as np

import simulacra as si

from modulation.resonators.microspheres import vsh


@pytest.mark.parametrize("l, m", [(l, m) for l in range(21) for m in range(-l, l + 1)])
def test_recurrence_sph_harm_agrees_with_direct_sph_harm_for_small_lm(
    theta_and_phi_meshes, l, m
):
    theta_mesh, phi_mesh = theta_and_phi_meshes
    direct = si.math.SphericalHarmonic(l, m)(theta_mesh, phi_mesh)
    recurr = vsh.RecurrentSphericalHarmonic(l, m)(theta_mesh, phi_mesh)

    assert np.allclose(direct, recurr, rtol=1e-12, atol=1e-12)
