import logging

import abc

logger = logging.getLogger(__name__)


class Mode(abc.ABC):
    """
    An interface for a single electromagnetic mode of a resonator.
    """
    pass
