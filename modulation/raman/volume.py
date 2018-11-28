import logging
from typing import Tuple

import abc

import simulacra as si

from .mode import Mode

logger = logging.getLogger(__name__)


class ModeVolumeIntegrator(abc.ABC):
    """An interface for calculating volume integrals over products of mode magnitudes."""

    @abc.abstractmethod
    def mode_volume_integral(self, modes: Tuple[Mode, ...]) -> float:
        """
        Perform a volume integral over the product of the vector magnitudes of the electric fields of the given modes.
        The integral should only be over the "Raman-active" area of the resonator (i.e., not the outside).
        """
        raise NotImplementedError

    def info(self) -> si.Info:
        info = si.Info(header = f'Mode Volume Integrator: {self.__class__.__name__}')
        return info
