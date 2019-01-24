import collections

import numpy as np

from . import exceptions

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
        if self._entries is None:
            raise exceptions.LookbackIsFrozen('this lookback is frozen; no more measurements can be added to it')

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
        """The average magnitude of each mode over the lookback time."""
        return np.mean(self._stacked_magnitudes, axis = 0)

    @freezeable_property
    def max(self) -> np.ndarray:
        """The maximum magnitude of each mode over the lookback time."""
        return np.max(self._stacked_magnitudes, axis = 0)

    @freezeable_property
    def min(self) -> np.ndarray:
        """The minimum magnitude of each mode over the lookback time."""
        return np.min(self._stacked_magnitudes, axis = 0)

    @freezeable_property
    def std(self) -> np.ndarray:
        """The standard deviation of the magnitude of each mode over the lookback time."""
        return np.std(self._stacked_magnitudes, axis = 0)

    def freeze(self):
        """Lock the statistical measurements to their current values and clear internal storage."""
        for stat in self.STATS:
            self.__dict__[stat] = getattr(self, stat)

        self._entries = None
