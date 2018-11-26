import logging

import abc

import simulacra as si
import simulacra.units as u

from .. import fmt

logger = logging.getLogger(__name__)


class Mode(abc.ABC):
    """
    An interface for a single electromagnetic mode of a resonator.
    """

    @property
    @abc.abstractmethod
    def omega(self) -> float:
        raise NotImplementedError

    @property
    def frequency(self):
        return self.omega / u.twopi

    @property
    @abc.abstractmethod
    def index_of_refraction(self) -> float:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def mode_volume_inside_resonator(self) -> float:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def mode_volume_outside_resonator(self) -> float:
        raise NotImplementedError

    def __str__(self):
        return f'{self.__class__.__name__}'

    def info(self) -> si.Info:
        info = si.Info(header = str(self))

        info.add_field('Frequency', fmt.quantity(self.frequency, fmt.FREQUENCY_UNITS))
        info.add_field('Index of Refraction', self.index_of_refraction)

        return info
