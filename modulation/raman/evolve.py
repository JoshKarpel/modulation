import logging

import abc
import functools

import numpy as np
import scipy.optimize as opt
import mpmath

import simulacra as si

logger = logging.getLogger(__name__)


class EvolutionAlgorithm(abc.ABC):
    """
    An interface for an algorithm that evolves the mode amplitudes forward in time.
    """

    @abc.abstractmethod
    def evolve(
        self, sim, mode_amplitudes: np.ndarray, time_initial: float, time_final: float
    ) -> np.ndarray:
        raise NotImplementedError

    def info(self) -> si.Info:
        info = si.Info(header=f"Evolution Algorithm: {self.__class__.__name__}")
        return info


class ForwardEuler(EvolutionAlgorithm):
    """
    Calculate the mode amplitudes at the next time step using the forward Euler algorithm.
    This method has very poor accuracy and stability - it is mainly implemented for completeness and for doing comparisons to other methods.
    """

    def evolve(
        self, sim, mode_amplitudes: np.ndarray, time_initial: float, time_final: float
    ) -> np.ndarray:
        deriv = sim.calculate_polarization_and_decay(mode_amplitudes, time_initial)
        dt = time_final - time_initial
        return mode_amplitudes + (deriv * dt)


def split(amps):
    return np.concatenate((np.real(amps), np.imag(amps)))


def combine(joined):
    real, imag = np.split(joined, 2)
    return real + (1j * imag)


class ImplicitEvolutionAlgorithm(EvolutionAlgorithm):
    def evolve(
        self, sim, mode_amplitudes: np.ndarray, time_initial: float, time_final: float
    ) -> np.ndarray:
        sol = opt.fsolve(
            lambda x: self.root_function(
                x,
                sim=sim,
                initial_mode_amplitudes=mode_amplitudes,
                time_initial=time_initial,
                time_final=time_final,
            ),
            x0=split(mode_amplitudes),
        )

        return combine(sol)

    @abc.abstractmethod
    def root_function(
        self,
        new_mode_amplitudes,
        sim,
        initial_mode_amplitudes,
        time_initial,
        time_final,
    ):
        raise NotImplementedError


class BackwardEuler(ImplicitEvolutionAlgorithm):
    def root_function(
        self,
        new_mode_amplitudes,
        sim,
        initial_mode_amplitudes,
        time_initial,
        time_final,
    ):
        result = (
            -combine(new_mode_amplitudes)
            + initial_mode_amplitudes
            + (
                (time_final - time_initial)
                * sim.calculate_polarization_and_decay(
                    combine(new_mode_amplitudes), time_final
                )
            )
        )
        return split(result)


class TrapezoidRule(ImplicitEvolutionAlgorithm):
    def root_function(
        self,
        new_mode_amplitudes,
        sim,
        initial_mode_amplitudes,
        time_initial,
        time_final,
    ):
        result = (
            -combine(new_mode_amplitudes)
            + initial_mode_amplitudes
            + (
                0.5
                * (time_final - time_initial)
                * (
                    sim.calculate_polarization_and_decay(
                        initial_mode_amplitudes, time_initial
                    )
                    + sim.calculate_polarization_and_decay(
                        combine(new_mode_amplitudes), time_final
                    )
                )
            )
        )
        return split(result)


class RungeKutta4(EvolutionAlgorithm):
    """
    Calculate the mode amplitudes at the next time step using the fourth-order Runge-Kutta algorithm.
    This algorithm has much better accuracy and stability than forward Euler.
    """

    def evolve(
        self, sim, mode_amplitudes: np.ndarray, time_initial: float, time_final: float
    ) -> np.ndarray:
        dt = time_final - time_initial
        time_half = time_initial + (dt / 2)

        k1 = dt * sim.calculate_polarization_and_decay(mode_amplitudes, time_initial)
        k2 = dt * sim.calculate_polarization_and_decay(
            mode_amplitudes + (k1 / 2), time_half
        )
        k3 = dt * sim.calculate_polarization_and_decay(
            mode_amplitudes + (k2 / 2), time_half
        )
        k4 = dt * sim.calculate_polarization_and_decay(mode_amplitudes + k3, time_final)

        return mode_amplitudes + ((k1 + (2 * k2) + (2 * k3) + k4) / 6)
