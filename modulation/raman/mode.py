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
        """The angular frequency of the mode."""
        raise NotImplementedError

    @property
    def frequency(self):
        """The cyclic frequency of the mode."""
        return self.omega / u.twopi

    @property
    def photon_energy(self):
        return u.hbar * self.omega

    @property
    @abc.abstractmethod
    def index_of_refraction(self) -> float:
        """The first-order index of refraction that the mode sees in the resonator."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def mode_volume_inside_resonator(self) -> float:
        """The mode volume of the part of the mode inside the resonator."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def mode_volume_outside_resonator(self) -> float:
        """The mode volume of the part of the mode outside the resonator."""
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        return self.__class__.__name__

    @property
    @abc.abstractmethod
    def tex(self):
        raise NotImplementedError

    def info(self) -> si.Info:
        info = si.Info(header = self.__class__.__name__)

        info.add_field('Frequency', fmt.quantity(self.frequency, fmt.FREQUENCY_UNITS))
        info.add_field('Index of Refraction', self.index_of_refraction)
        info.add_field('Mode Volume Inside Resonator', fmt.quantity(self.mode_volume_inside_resonator, fmt.VOLUME_UNITS))
        info.add_field('Mode Volume Outside Resonator', fmt.quantity(self.mode_volume_outside_resonator, fmt.VOLUME_UNITS))

        return info
