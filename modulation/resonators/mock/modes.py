import logging

import simulacra as si
import simulacra.units as u

from ...raman import Mode
from ... import fmt

logger = logging.getLogger(__name__)


class MockMode(Mode):
    def __init__(
        self,
        *,
        omega: float,
        index_of_refraction: float,
    ):
        self._omega = omega
        self._index_of_refraction = index_of_refraction

    @property
    def omega(self):
        return self._omega

    @property
    def index_of_refraction(self):
        return self._index_of_refraction
