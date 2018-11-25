import abc
from typing import Optional

import numpy as np

import simulacra as si
import simulacra.units as u


class Pump(abc.ABC):
    @abc.abstractmethod
    def power(self, time):
        """Return the pump power at the given time."""
        raise NotImplementedError

    def __repr__(self):
        return f'{self.__class__.__name__}'

    def __str__(self):
        return f'{self.__class__.__name__}'

    def info(self) -> si.Info:
        info = si.Info(header = self.__class__.__name__)

        return info


class RectangularPump(Pump):
    """
    A pump with a rectangular profile in time.
    Either endpoint may be ``None``, in which case the rectangle extends to infinity in that direction.
    """

    __slots__ = ('_power', 'start_time', 'end_time')

    def __init__(
        self,
        power: float,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ):
        """
        Parameters
        ----------
        power
            The pump power.
        start_time
            The time that the pump turns on.
            If ``None``, the pump is (effectively) turned on at the moment the simulation begins.
        end_time
            The time that the pump turns off.
            If ``None``, the pump never turns off once it's on.
        """
        self._power = power

        if start_time is None:
            start_time = -np.inf
        self.start_time = start_time

        if end_time is None:
            end_time = np.inf
        self.end_time = end_time

    def power(self, time):
        """Return the pump power at the given time."""
        return np.where(
            np.greater_equal(time, self.start_time) * np.less_equal(time, self.end_time),
            self._power,
            0,
        )

    def __repr__(self):
        return f'{self.__class__.__name__}(power = {self._power}, start_time = {self.start_time}, end_time = {self.end_time})'

    def __str__(self):
        return f'{self.__class__.__name__}(power = {u.uround(self._power, u.mW)} mW, start_time = {u.uround(self.start_time, u.nsec)} ns, end_time = {u.uround(self.end_time, u.nsec)} ns)'

    def info(self) -> si.Info:
        info = super().info()

        info.add_field('Power', f'{u.uround(self._power, u.uW)} µW | {u.uround(self._power, u.mW)} mW | {u.uround(self._power, u.W)} W')
        info.add_field('Start Time', f'{u.uround(self.start_time, u.nsec)} ns' if self.start_time != -np.inf else '-∞')
        info.add_field('End Time', f'{u.uround(self.end_time, u.nsec)} ns' if self.end_time != np.inf else '+∞')

        return info


class ConstantPump(Pump):
    """A pump that is always on."""

    __slots__ = ('_power',)

    def __init__(self, power: float):
        self._power = power

    def power(self, time):
        return self._power * np.ones_like(time)

    def __repr__(self):
        return f'{self.__class__.__name__}(power = {self._power})'

    def __str__(self):
        return f'{self.__class__.__name__}(power = {u.uround(self._power, u.mW)} mW)'

    def info(self) -> si.Info:
        info = super().info()

        info.add_field('Power', f'{u.uround(self._power, u.uW)} µW | {u.uround(self._power, u.mW)} mW | {u.uround(self._power, u.W)} W')

        return info
