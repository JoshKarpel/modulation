import pytest

import numpy as np


from modulation.resonators.microspheres import vsh


@pytest.mark.parametrize("l, m", [(l, m) for l in range(21) for m in range(-l, l + 1)])
def test_radial_vsh_has_only_radial_component(theta_and_phi_meshes, l, m):
    theta_mesh, phi_mesh = theta_and_phi_meshes

    z = vsh.VectorSphericalHarmonic(
        type=vsh.VectorSphericalHarmonicType.RADIAL, l=l, m=m
    )

    assert not np.allclose(z(theta_mesh, phi_mesh)[..., vsh.SphericalComponent.R], 0)
    assert np.allclose(z(theta_mesh, phi_mesh)[..., vsh.SphericalComponent.THETA], 0)
    assert np.allclose(z(theta_mesh, phi_mesh)[..., vsh.SphericalComponent.PHI], 0)


@pytest.mark.parametrize(
    "type",
    [vsh.VectorSphericalHarmonicType.GRADIENT, vsh.VectorSphericalHarmonicType.CROSS],
)
@pytest.mark.parametrize("l, m", [(l, m) for l in range(21) for m in range(-l, l + 1)])
def test_grad_and_cross_vsh_have_only_theta_and_phi_component(
    theta_and_phi_meshes, type, l, m
):
    theta_mesh, phi_mesh = theta_and_phi_meshes

    z = vsh.VectorSphericalHarmonic(type=type, l=l, m=m)

    assert np.allclose(z(theta_mesh, phi_mesh)[..., vsh.SphericalComponent.R], 0)

    # note: when m = 0, the theta or phi components may be zero anyway, so skip them
    assert m == 0 or not np.allclose(
        z(theta_mesh, phi_mesh)[..., vsh.SphericalComponent.THETA], 0
    )
    assert m == 0 or not np.allclose(
        z(theta_mesh, phi_mesh)[..., vsh.SphericalComponent.PHI], 0
    )
