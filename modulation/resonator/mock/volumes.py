import logging
from typing import Tuple

import simulacra as si

from ... import fmt
from ...raman import Mode, ModeVolumeIntegrator

logger = logging.getLogger(__name__)


class MockVolumeIntegrator(ModeVolumeIntegrator):
    def __init__(self, *, volume_integral_result: float):
        self.volume_integral_result = volume_integral_result

    def mode_volume_integral(self, modes: Tuple[Mode, ...]) -> float:
        return self.volume_integral_result

    def info(self) -> si.Info:
        info = super().info()
        info.add_field('Volume Integral Result', fmt.quantity(self.volume_integral_result, fmt.VOLUME_UNITS))

        return info
