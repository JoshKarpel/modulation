import numpy as np

import simulacra.units as u

theta = np.linspace(0, u.pi, 100)[1:-1]
phi = np.linspace(0, u.twopi, 100)[1:]

theta_mesh, phi_mesh = np.meshgrid(theta, phi, indexing = 'ij')
