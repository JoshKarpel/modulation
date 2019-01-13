import collections

import numpy as np

from . import sims


class freezeable_property():
    def __init__(self, func):
        self.func = func
        self.name = func.__name__

    def __get__(self, instance, cls = None):
        if instance is None:
            return self

        return self.func(instance)


class Lookback:
    """
    A helper class for a RamanSimulation that "looks backward" from the end of the simulation and collects summary data on the mode amplitudes.
    """

    STATS = [
        'mean',
        'max',
        'min',
        'std',
        'ptp',
    ]

    def __init__(
        self,
        lookback_time: float = 0,
    ):
        """
        Parameters
        ----------
        lookback_time
            The amount of simulation time to store mode amplitudes for.
            ``lookback_time = 0`` will store exactly one entry (the current time).
        """
        self.lookback_time = lookback_time

        self._entries = collections.deque()

    def add(self, time: float, mode_amplitudes: np.ndarray):
        """Add a new mode amplitude to the Lookback."""
        self._entries.append((time, mode_amplitudes.copy()))

        earliest_time = time - self.lookback_time
        while earliest_time > self._entries[0][0]:
            self._entries.popleft()

    @property
    def _stacked(self) -> np.ndarray:
        """
        Concatenate all of the stored mode amplitude arrays into a two-dimensional array.
        The first index corresponds to the times and the second to the mode.
        Summary statistics should operate on the first axis.
        """
        return np.vstack(e[1] for e in self._entries)

    @property
    def _stacked_magnitudes(self):
        return np.abs(self._stacked)

    @freezeable_property
    def mean(self) -> np.ndarray:
        return np.mean(self._stacked_magnitudes, axis = 0)

    @freezeable_property
    def max(self) -> np.ndarray:
        return np.max(self._stacked_magnitudes, axis = 0)

    @freezeable_property
    def min(self) -> np.ndarray:
        return np.min(self._stacked_magnitudes, axis = 0)

    @freezeable_property
    def std(self) -> np.ndarray:
        return np.std(self._stacked_magnitudes, axis = 0)

    @freezeable_property
    def ptp(self):
        return np.ptp(self._stacked_magnitudes, axis = 0)

    def freeze(self):
        for stat in self.STATS:
            self.__dict__[stat] = getattr(self, stat)

        self._entries = None
