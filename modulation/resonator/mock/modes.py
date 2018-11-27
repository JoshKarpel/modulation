import logging

from ...raman import Mode

logger = logging.getLogger(__name__)


class MockMode(Mode):
    """A fake mode with arbitrarily-settable internal parameters."""

    def __init__(
        self,
        *,
        omega: float,
        index_of_refraction: float,
        mode_volume_inside_resonator: float,
        mode_volume_outside_resonator: float = 0,
    ):
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
