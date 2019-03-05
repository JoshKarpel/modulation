import pytest

import numpy as np

import simulacra.units as u


@pytest.fixture(scope="session")
def theta():
    return np.linspace(0, u.pi, 100)[1:-1]


@pytest.fixture(scope="session")
def phi():
    return np.linspace(0, u.twopi, 100)[1:]


@pytest.fixture(scope="session")
def dense_theta():
    return np.linspace(0, u.pi, 1000)[1:-1]


@pytest.fixture(scope="session")
def dense_phi():
    return np.linspace(0, u.twopi, 1000)[1:]


@pytest.fixture(scope="session")
def theta_and_phi_meshes(theta, phi):
    theta_mesh, phi_mesh = np.meshgrid(theta, phi, indexing="ij")

    return theta_mesh, phi_mesh


@pytest.fixture(scope="session")
def dense_theta_and_phi_meshes(dense_theta, dense_phi):
    dense_theta_mesh, dense_phi_mesh = np.meshgrid(
        dense_theta, dense_phi, indexing="ij"
    )

    return dense_theta_mesh, dense_phi_mesh


@pytest.fixture(scope="session")
def dense_theta_phi_jacobian(dense_theta, dense_phi, dense_theta_and_phi_meshes):
    dense_theta_mesh, dense_phi_mesh = dense_theta_and_phi_meshes
    d_theta = np.abs(dense_theta[1] - dense_theta[0])
    d_phi = np.abs(dense_phi[1] - dense_phi[0])
    sin_theta_mesh = np.sin(dense_theta_mesh)
    jacobian = sin_theta_mesh * d_theta * d_phi
    return jacobian
