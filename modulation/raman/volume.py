import abc
import functools
import operator

import numpy as np


class ModeVolumeIntegrator(abc.ABC):
    def mode_volume_integral_inside(self, modes):
        raise NotImplementedError

    def integrand(self, modes, r, theta, phi):
        return functools.reduce(operator.mul, (self.mode_magnitude_inside(m, r, theta, phi) for m in modes))

    def mode_magnitude_inside(self, mode, r, theta, phi):
        mode_shape_vector_field = mode.evaluate_electric_field_mode_shape_inside(r, theta, phi)
        return np.sqrt(np.sum(np.abs(mode_shape_vector_field) ** 2, axis = -1))
