import abc

import numpy as np


class EvolutionAlgorithm(abc.ABC):
    @abc.abstractmethod
    def evolve(
        self,
        sim,
        mode_amplitudes: np.ndarray,
        time_initial: float,
        time_final: float,
    ) -> np.ndarray:
        raise NotImplementedError


class ForwardEuler(EvolutionAlgorithm):
    """
    Calculate the mode amplitudes at the next time step using the forward Euler algorithm.
    This method has very poor accuracy and stability - it is mainly implemented for completeness and for doing comparisons to other methods.
    """

    def evolve(
        self,
        sim,
        mode_amplitudes: np.ndarray,
        time_initial: float,
        time_final: float,
    ) -> np.ndarray:
        return mode_amplitudes + (sim.calculate_total_derivative(mode_amplitudes, time_initial) * (time_final - time_initial))


class RungeKutta4(EvolutionAlgorithm):
    """
    Calculate the mode amplitudes at the next time step using the fourth-order Runge-Kutta algorithm.
    This method has much better accuracy and stability than forward Euler, but still has trouble with the non-linear pump term.
    """

    def evolve(
        self,
        sim,
        mode_amplitudes: np.ndarray,
        time_initial: float,
        time_final: float,
    ) -> np.ndarray:
        dt = time_final - time_initial
        time_half = time_initial + (dt / 2)

        k1 = dt * sim.calculate_total_derivative(mode_amplitudes, time_initial)
        k2 = dt * sim.calculate_total_derivative(mode_amplitudes + (k1 / 2), time_half)
        k3 = dt * sim.calculate_total_derivative(mode_amplitudes + (k2 / 2), time_half)
        k4 = dt * sim.calculate_total_derivative(mode_amplitudes + k3, time_final)

        return mode_amplitudes + ((k1 + (2 * k2) + (2 * k3) + k4) / 6)
