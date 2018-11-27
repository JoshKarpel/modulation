import logging

import simulacra.units as u

from ...raman import Mode

logger = logging.getLogger(__name__)


class MockMode(Mode):
    """A fake mode with arbitrarily-settable internal parameters."""

    def __init__(
        self,
        *,
        label: str,
        omega: float,
        index_of_refraction: float,
        mode_volume_inside_resonator: float,
        mode_volume_outside_resonator: float = 0,
    ):
        self.label = label
        self._omega = omega
        self._index_of_refraction = index_of_refraction
        self._mode_volume_inside_resonator = mode_volume_inside_resonator
        self._mode_volume_outside_resonator = mode_volume_outside_resonator

    @property
    def omega(self):
        return self._omega

    @property
    def index_of_refraction(self):
        return self._index_of_refraction

    @property
    def mode_volume_inside_resonator(self) -> float:
        return self._mode_volume_inside_resonator

    @property
    def mode_volume_outside_resonator(self) -> float:
        return self._mode_volume_outside_resonator

    @property
    def tex(self):
        return self.label

    def __repr__(self):
        return f'{self.__class__.__name__}(omega = {self.omega}, index_of_refraction = {self.index_of_refraction}, mode_volume_inside_resonator = {self.mode_volume_inside_resonator}, mode_volume_outside_resonator = {self.mode_volume_outside_resonator})'

    def __str__(self):
        return f'{self.__class__.__name__}(frequency = {self.frequency / u.THz:.6f} THz, index_of_refraction = {self.index_of_refraction}, mode_volume_inside_resonator = {self.mode_volume_inside_resonator / (u.um ** 3):.3f} µm^3, mode_volume_outside_resonator = {self.mode_volume_outside_resonator / (u.um ** 3):.3f} µm^3)'
