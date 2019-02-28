import logging
from typing import Optional

import abc

import numpy as np

import simulacra as si
import simulacra.units as u

from .. import fmt

logger = logging.getLogger(__name__)


class MonochromaticPump(abc.ABC):
    @property
    @abc.abstractmethod
    def omega(self):
        """Return the frequency of the pump."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_power(self, time):
        """Return the pump power at the given time."""
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def __str__(self):
        return f"{self.__class__.__name__}"

    def info(self) -> si.Info:
        info = si.Info(header=self.__class__.__name__)

        return info


class RectangularMonochromaticPump(MonochromaticPump):
    """
    A pump with a rectangular profile in time.
    Either endpoint may be ``None``, in which case the rectangle extends to infinity in that direction.
    """

    __slots__ = ("frequency", "power", "start_time", "end_time")

    def __init__(
        self,
        frequency: float,
        power: float,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ):
        """
        Parameters
        ----------
        frequency
            The pump frequency.
        power
            The pump power.
        start_time
            The time that the pump turns on.
            If ``None``, the pump is (effectively) turned on at the moment the simulation begins.
        end_time
            The time that the pump turns off.
            If ``None``, the pump never turns off once it's on.
        """
        self.frequency = frequency
        self.power = power

        if start_time is None:
            start_time = -np.inf
        self.start_time = start_time

        if end_time is None:
            end_time = np.inf
        self.end_time = end_time

    @property
    def omega(self):
        return u.twopi * self.frequency

    def get_power(self, time):
        """Return the pump power at the given time."""
        return np.where(
            np.greater_equal(time, self.start_time)
            * np.less_equal(time, self.end_time),
            self.power,
            0,
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(frequency = {self.frequency}, power = {self.power}, start_time = {self.start_time}, end_time = {self.end_time})"

    def __str__(self):
        return f"{self.__class__.__name__}(frequency = {self.frequency / u.THz:.6f} THz, power = {u.uround(self.power, u.mW)} mW, start_time = {u.uround(self.start_time, u.nsec)} ns, end_time = {u.uround(self.end_time, u.nsec)} ns)"

    def info(self) -> si.Info:
        info = super().info()

        info.add_field("Frequency", fmt.quantity(self.frequency, fmt.FREQUENCY_UNITS))
        info.add_field("Power", fmt.quantity(self.power, fmt.POWER_UNITS))
        info.add_field(
            "Start Time",
            fmt.quantity(self.start_time, fmt.TIME_UNITS)
            if self.start_time != -np.inf
            else "-∞",
        )
        info.add_field(
            "End Time",
            fmt.quantity(self.end_time, fmt.TIME_UNITS)
            if self.end_time != np.inf
            else "+∞",
        )

        return info


class ConstantMonochromaticPump(RectangularMonochromaticPump):
    """A pump that is always on."""

    __slots__ = ("frequency", "power")

    def __init__(self, frequency: float, power: float):
        super().__init__(frequency=frequency, power=power)

    def get_power(self, time):
        """Return the pump power at the given time."""
        return self.power * np.ones_like(time)

    def __repr__(self):
        return f"{self.__class__.__name__}(frequency = {self.frequency}, power = {self.power})"

    def __str__(self):
        return f"{self.__class__.__name__}(frequency = {self.frequency / u.THz:.6f THz}, power = {u.uround(self.power, u.mW)} mW)"

    def info(self) -> si.Info:
        info = super().info()

        info.add_field("Frequency", fmt.quantity(self.frequency, fmt.FREQUENCY_UNITS))
        info.add_field("Power", fmt.quantity(self.power, fmt.POWER_UNITS))

        return info
