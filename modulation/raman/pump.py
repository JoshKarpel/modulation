import logging
from typing import Optional

import abc

import numpy as np

import simulacra as si
import simulacra.units as u

from .. import fmt

logger = logging.getLogger(__name__)


class MonochromaticPump(abc.ABC):
    def __init__(self, frequency: float):
        self.frequency = frequency

    @property
    def omega(self):
        """Return the angular frequency of the pump."""
        return u.twopi * self.frequency

    @property
    def wavelength(self):
        return u.c / self.frequency

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
        super().__init__(frequency=frequency)

        if power < 0:
            raise ValueError("pump power must be non-negative")
        self.power = power

        if start_time is None:
            start_time = -np.inf
        self.start_time = start_time

        if end_time is None:
            end_time = np.inf
        self.end_time = end_time

    @classmethod
    def from_wavelength(
        cls,
        wavelength: float,
        power: float,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ):
        frequency = u.c / wavelength
        return cls(frequency, power, start_time, end_time)

    @classmethod
    def from_omega(
        cls,
        omega: float,
        power: float,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ):
        frequency = omega / u.twopi
        return cls(frequency, power, start_time, end_time)

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
        return f"{self.__class__.__name__}(frequency = {self.frequency / u.THz:.9f} THz, power = {self.power/ u.mW:.3f} mW, start_time = {self.start_time/u.nsec:3f} ns, end_time = {self.end_time/ u.nsec:3f} ns)"

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


class ConstantMonochromaticPump(MonochromaticPump):
    """A pump that is always on."""

    __slots__ = ("frequency", "power")

    def __init__(self, frequency: float, power: float):
        super().__init__(frequency=frequency)

        if power < 0:
            raise ValueError("pump power must be non-negative")
        self.power = power

    @classmethod
    def from_wavelength(cls, wavelength: float, power: float):
        frequency = u.c / wavelength
        return cls(frequency, power)

    @classmethod
    def from_omega(cls, omega: float, power: float):
        frequency = omega / u.twopi
        return cls(frequency, power)

    def get_power(self, time):
        """Return the pump power at the given time."""
        return self.power * np.ones_like(time)

    def __repr__(self):
        return f"{self.__class__.__name__}(frequency = {self.frequency}, power = {self.power})"

    def __str__(self):
        return f"{self.__class__.__name__}(frequency = {self.frequency / u.THz:.9f} THz, power = {self.power/ u.mW:.3f} mW)"

    def info(self) -> si.Info:
        info = super().info()

        info.add_field("Frequency", fmt.quantity(self.frequency, fmt.FREQUENCY_UNITS))
        info.add_field("Power", fmt.quantity(self.power, fmt.POWER_UNITS))

        return info
