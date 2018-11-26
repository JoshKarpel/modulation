import pytest

import itertools

import numpy as np

import simulacra.units as u

import whisper.vsh as vsh

L_MAX = 3
L_AND_M = [(l, m) for l in range(L_MAX) for m in range(-l, l + 1)]

theta = np.linspace(0, u.pi, 100)[1:-1]
phi = np.linspace(0, u.twopi, 100)[1:]
theta_mesh, phi_mesh = np.meshgrid(theta, phi, indexing = 'ij')


@pytest.mark.parametrize(
    'type_a, type_b',
    itertools.combinations(list(vsh.VectorSphericalHarmonicType), 2)
)
@pytest.mark.parametrize(
    'l, m',
    [
        *[(l, m) for l in range(21) for m in range(-l, l + 1)],
        *[(l, m) for l in range(498, 503) for m in range(-l, l + 1)],
    ],
)
def test_pointwise_orthogonality_for_same_lm(type_a, type_b, l, m):
    vsh_a = vsh.VectorSphericalHarmonic(
        type = type_a,
        l = l,
        m = m,
    )
    vsh_b = vsh.VectorSphericalHarmonic(
        type = type_b,
        l = l,
        m = m,
    )

    # this is a point-wise inner product
    ip = vsh.inner_product_of_vsh(
        vsh_a(theta_mesh, phi_mesh),
        vsh_b(theta_mesh, phi_mesh),
    )

    assert np.allclose(ip, 0)


@pytest.mark.parametrize(
    'type_b',
    [vsh.VectorSphericalHarmonicType.GRADIENT, vsh.VectorSphericalHarmonicType.CROSS]
)
@pytest.mark.parametrize(
    'l_a, m_a',
    [
        *[(l, m) for l in range(5) for m in range(-l, l + 1)],
        *[(l, m) for l in range(500, 501) for m in range(-l, -l + 4)],
        *[(l, m) for l in range(500, 501) for m in range(l - 4, l)],
    ],
)
@pytest.mark.parametrize(
    'l_b, m_b',
    [
        *[(l, m) for l in range(3) for m in range(-l, l + 1)],
        *[(l, m) for l in range(500, 501) for m in range(-l, -l + 4)],
        *[(l, m) for l in range(500, 501) for m in range(l - 4, l)],
    ],
)
def test_pointwise_orthogonality_for_different_lm_for_radial_to_grad_and_cross(type_b, l_a, m_a, l_b, m_b):
    vsh_a = vsh.VectorSphericalHarmonic(
        type = vsh.VectorSphericalHarmonicType.RADIAL,
        l = l_a,
        m = m_a,
    )
    vsh_b = vsh.VectorSphericalHarmonic(
        type = type_b,
        l = l_b,
        m = m_b,
    )

    ip = vsh.inner_product_of_vsh(
        vsh_a(theta_mesh, phi_mesh),
        vsh_b(theta_mesh, phi_mesh),
    )

    assert np.allclose(ip, 0)


dense_theta = np.linspace(0, u.pi, 500)[1:-1]
dense_phi = np.linspace(0, u.twopi, 500)[1:]
dense_theta_mesh, dense_phi_mesh = np.meshgrid(dense_theta, dense_phi, indexing = 'ij')
d_theta = np.abs(dense_theta[1] - dense_theta[0])
d_phi = np.abs(dense_phi[1] - dense_phi[0])
sin_theta_mesh = np.sin(dense_theta_mesh)
jacobian = sin_theta_mesh * d_theta * d_phi


@pytest.mark.parametrize(
    'type_a, type_b',
    itertools.combinations_with_replacement(list(vsh.VectorSphericalHarmonicType), r = 2),
)
@pytest.mark.parametrize(
    'l_a, m_a',
    [(l, m) for l in range(3) for m in range(-l, l + 1)],
)
@pytest.mark.parametrize(
    'l_b, m_b',
    [(l, m) for l in range(3) for m in range(-l, l + 1)],
)
def test_integral_orthonormalization_small_l_and_m(type_a, type_b, l_a, m_a, l_b, m_b):
    vsh_a = vsh.VectorSphericalHarmonic(
        type = type_a,
        l = l_a,
        m = m_a,
    )
    vsh_b = vsh.VectorSphericalHarmonic(
        type = type_b,
        l = l_b,
        m = m_b,
    )

    a = vsh_a(dense_theta_mesh, dense_phi_mesh)
    b = vsh_b(dense_theta_mesh, dense_phi_mesh)

    result = np.sum(vsh.inner_product_of_vsh(np.conj(a), b) * jacobian)

    # if it's supposed to be zero, check result +1 == 1 for better stability
    # need to short-circuit the l = 0 cases for grad and cross, which are 0 when l = 0
    if any((l_a == 0 and type_a in (vsh.VectorSphericalHarmonicType.GRADIENT, vsh.VectorSphericalHarmonicType.CROSS),
            l_b == 0 and type_b in (vsh.VectorSphericalHarmonicType.GRADIENT, vsh.VectorSphericalHarmonicType.CROSS))):
        assert np.allclose(result + 1, 1)
    elif type_a is type_b and l_a == l_b and m_a == m_b:
        assert np.allclose(result, 1)
    else:
        assert np.allclose(result + 1, 1)


dense_theta = np.linspace(0, u.pi, 1000)[1:-1]
dense_phi = np.linspace(0, u.twopi, 1000)[1:]
dense_theta_mesh, dense_phi_mesh = np.meshgrid(dense_theta, dense_phi, indexing = 'ij')
d_theta = np.abs(dense_theta[1] - dense_theta[0])
d_phi = np.abs(dense_phi[1] - dense_phi[0])
sin_theta_mesh = np.sin(dense_theta_mesh)
jacobian = sin_theta_mesh * d_theta * d_phi


@pytest.mark.parametrize(
    'type_a, type_b',
    itertools.combinations_with_replacement(list(vsh.VectorSphericalHarmonicType), r = 2),
)
@pytest.mark.parametrize(
    'l_a, m_a',
    [(l, m) for l in range(3) for m in range(-l, l + 1)],
)
@pytest.mark.parametrize(
    'l_b, m_b',
    [(l, m) for l in range(3) for m in range(-l, l + 1)],
)
def test_integral_orthonormalization_small_l_and_m(type_a, type_b, l_a, m_a, l_b, m_b):
    vsh_a = vsh.VectorSphericalHarmonic(
        type = type_a,
        l = l_a,
        m = m_a,
    )
    vsh_b = vsh.VectorSphericalHarmonic(
        type = type_b,
        l = l_b,
        m = m_b,
    )

    a = vsh_a(dense_theta_mesh, dense_phi_mesh)
    b = vsh_b(dense_theta_mesh, dense_phi_mesh)

    result = np.sum(vsh.inner_product_of_vsh(np.conj(a), b) * jacobian)

    # if it's supposed to be zero, check result +1 == 1 for better stability
    # need to short-circuit the l = 0 cases for grad and cross, which are 0 when l = 0
    if any((l_a == 0 and type_a in (vsh.VectorSphericalHarmonicType.GRADIENT, vsh.VectorSphericalHarmonicType.CROSS),
            l_b == 0 and type_b in (vsh.VectorSphericalHarmonicType.GRADIENT, vsh.VectorSphericalHarmonicType.CROSS))):
        assert np.allclose(result + 1, 1)
    elif type_a is type_b and l_a == l_b and m_a == m_b:
        assert np.allclose(result, 1)
    else:
        assert np.allclose(result + 1, 1)
