from typing import Iterable, Dict, Union, Optional, Generator, Tuple, Callable, List
import logging

from pathlib import Path
import functools
import itertools
import datetime
import abc
import sys

import numpy as np

from tqdm import tqdm

import simulacra as si
import simulacra.units as u

from . import mode, pump, evolve, volume, material, plotter, lookback, exceptions
from .. import fmt
from .cy import four_wave_polarization

logger = logging.getLogger(__name__)


class RamanSimulation(si.Simulation):
    def __init__(self, spec):
        super().__init__(spec)

        self.lookback = spec.lookback

        self.latest_checkpoint_time = (
            datetime.datetime.utcnow() - 2 * self.spec.checkpoint_every
        )

        total_time = self.spec.time_final - self.spec.time_initial
        self.time_index = 0
        self.time_steps = int(total_time / self.spec.time_step) + 1
        self.real_time_step = total_time / (self.time_steps - 1)
        self.time_rng_state = np.random.RandomState().get_state()

        self.mode_omegas = np.array([m.omega for m in self.spec.modes])
        self.mode_amplitude_decay_rates = self.mode_omegas / (
            2 * self.spec.mode_total_quality_factors
        )
        self.mode_index_of_refraction = np.array(
            [m.index_of_refraction for m in self.spec.modes]
        )
        self.mode_epsilons = self.mode_index_of_refraction ** 2
        self.mode_amplitudes = self.spec.mode_initial_amplitudes.copy()
        self.mode_to_index = {mode: idx for idx, mode in enumerate(self.spec.modes)}

        if self.spec.store_mode_amplitudes_vs_time:
            self.mode_amplitudes_vs_time = np.empty(
                (self.time_steps, len(self.mode_amplitudes)), dtype=np.complex128
            )

        self.mode_volumes_inside_resonator = np.array(
            [m.mode_volume_inside_resonator for m in self.spec.modes]
        )
        self.mode_volumes_outside_resonator = np.array(
            [m.mode_volume_outside_resonator for m in self.spec.modes]
        )
        self.mode_volumes = (
            self.mode_volumes_inside_resonator + self.mode_volumes_outside_resonator
        )

        self.mode_background_magnitudes = np.sqrt(
            self.mode_photon_energy / self.mode_energy_prefactor
        )  # one photon per mode

        self.polarization_prefactor = (
            0.5j
            * self.spec.material.number_density
            * (self.mode_omegas / (u.epsilon_0 * self.mode_epsilons))
            * self.spec.material.raman_prefactor
        )
        self.pump_prefactor = np.sqrt(
            self.mode_omegas / self.spec.mode_coupling_quality_factors
        ) / np.sqrt(self.mode_energy_prefactor)

        if self.spec.cached_polarization_sum_factors is None:
            self.polarization_sum_factors = self._calculate_polarization_sum_factors()
        else:
            logger.debug(f"Using cached polarization sum factors for {self}")
            self.polarization_sum_factors = self.spec.cached_polarization_sum_factors
        self.spec.mode_volume_integrator.clear()

        self.plot = plotter.RamanSimulationPlotter(self)

    def time_at(self, time_index):
        return self.times[time_index]

    @property
    def current_time(self):
        return self.time_at(self.time_index)

    @si.utils.cached_property
    def times(self):
        times = np.linspace(
            self.spec.time_initial, self.spec.time_final, self.time_steps
        )

        rng = np.random.RandomState()
        rng.set_state(self.time_rng_state)
        scatter = rng.uniform(
            -self.real_time_step / 10, self.real_time_step / 10, size=self.time_steps
        )
        scatter[0] = 0
        scatter[-1] = 0

        return times + scatter

    @property
    def available_animation_frames(self) -> int:
        return self.time_steps

    @property
    def mode_photon_energy(self):
        """Return what the energy of each mode would be if it had a single photon in it."""
        return u.hbar * self.mode_omegas

    @property
    def mode_energy_prefactor(self):
        """The number that is multiplied by the square of the mode magnitude to get the energy in that mode."""
        inside = (
            0.5 * u.epsilon_0 * self.mode_epsilons * self.mode_volumes_inside_resonator
        )
        outside = 0.5 * u.epsilon_0 * self.mode_volumes_outside_resonator

        return inside + outside

    # @property
    # def mode_intensity_prefactor(self):
    #     """Multiply by E_q^2 and the mode shape^2 to get the intensity in space."""
    #     return 0.5 * u.c * u.epsilon_0 * np.sqrt(self.mode_epsilons)

    def mode_energies(self, mode_amplitudes):
        """Return the energy of each mode, based on the given ``mode_amplitudes``."""
        return self.mode_energy_prefactor * (np.abs(mode_amplitudes) ** 2)

    def mode_photon_numbers(self, mode_amplitudes):
        return self.mode_energies(mode_amplitudes) / self.mode_photon_energy

    def mode_output_powers(self, mode_amplitudes):
        """Return the output power of each mode, based on the given ``mode_amplitudes``."""
        return (
            self.mode_energies(mode_amplitudes)
            * self.mode_omegas
            / self.spec.mode_coupling_quality_factors
        )

    @property
    def mode_magnitudes_vs_time(self):
        return np.abs(self.mode_amplitudes_vs_time)

    def _two_photon_detuning(self, omega_x: float, omega_y: float) -> complex:
        return (omega_x - omega_y) - (
            self.spec.material.modulation_omega
            + (1j * self.spec.material.raman_linewidth)
        )

    def _double_inverse_detuning(self, omega_x: float, omega_y: float) -> complex:
        return np.conj(1 / self._two_photon_detuning(omega_x, omega_y)) + (
            1 / self._two_photon_detuning(omega_y, omega_x)
        )

    def modes_with_amplitudes(
        self
    ) -> Generator[Tuple[mode.Mode, np.complex128], None, None]:
        """Yields ``(mode, amplitude)`` pairs for all of the cavity modes in order."""
        yield from zip(self.spec.modes, self.mode_amplitudes)

    def calculate_polarization_and_decay(
        self, mode_amplitudes: np.ndarray, time: float
    ) -> np.ndarray:
        polarization = self.calculate_polarization(mode_amplitudes, time)
        decay = -self.mode_amplitude_decay_rates * mode_amplitudes

        return polarization + decay

    def evolve_pump(
        self, mode_amplitudes: np.ndarray, time_initial: float, time_final: float
    ) -> np.ndarray:
        change = np.zeros_like(mode_amplitudes, dtype=np.complex128)
        for pump in self.spec.pumps:
            dw = self.mode_omegas - pump.omega
            freq_part = np.where(
                dw != 0,
                (np.exp(1j * dw * time_final) - np.exp(1j * dw * time_initial)) / dw,
                time_final - time_initial,
            )

            change += (
                np.sqrt(pump.get_power((time_initial + time_final) / 2)) * freq_part
            )

        return mode_amplitudes - 0.5j * self.pump_prefactor * change

    @abc.abstractmethod
    def _calculate_polarization_sum_factors(self) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def calculate_polarization(
        self, mode_amplitudes: np.ndarray, time: float
    ) -> np.ndarray:
        raise NotImplementedError

    def generate_background_amplitudes(self) -> np.ndarray:
        """Generate a set of background **amplitudes** with randomized phases."""
        return self.mode_background_magnitudes * np.exp(
            1j * u.twopi * np.random.random(self.mode_amplitudes.shape)
        )

    def run(
        self,
        progress_bar: bool = False,
        checkpoint_callback: Callable[[Path], None] = None,
    ) -> None:
        self.status = si.Status.RUNNING

        try:
            for animator in self.spec.animators:
                animator.initialize(self)

            if progress_bar:
                pbar = tqdm(total=self.time_steps)

            while True:
                if self.spec.store_mode_amplitudes_vs_time:
                    self.mode_amplitudes_vs_time[
                        self.time_index, :
                    ] = self.mode_amplitudes

                if self.lookback is not None:
                    self.lookback.add(
                        self.time_at(self.time_index), self.mode_amplitudes
                    )

                for animator in self.spec.animators:
                    if (
                        self.time_index == 0
                        or self.time_index + 1 == self.time_steps
                        or self.time_index % animator.decimation == 0
                    ):
                        animator.send_frame_to_ffmpeg()

                if progress_bar:
                    pbar.update(1)
                if self.time_index + 1 == self.time_steps:
                    break

                self.time_index += 1

                start = self.time_at(self.time_index - 1)
                end = self.time_at(self.time_index)
                midpoint = (start + end) / 2

                after_first_pump = self.evolve_pump(
                    self.mode_amplitudes, time_initial=start, time_final=midpoint
                )
                after_nonlinear = self.spec.evolution_algorithm.evolve(
                    sim=self,
                    mode_amplitudes=after_first_pump,
                    time_initial=start,
                    time_final=end,
                )
                after_second_pump = self.evolve_pump(
                    after_nonlinear, time_initial=midpoint, time_final=end
                )
                self.mode_amplitudes = after_second_pump

                if self.spec.checkpoints:
                    now = datetime.datetime.utcnow()
                    if (now - self.latest_checkpoint_time) > self.spec.checkpoint_every:
                        self.do_checkpoint(now, checkpoint_callback)

            if progress_bar:
                pbar.close()
        finally:
            for animator in self.spec.animators:
                animator.cleanup()

            self.spec.animators = ()

        if self.spec.lookback is not None and self.spec.freeze_lookback:
            self.lookback.freeze()

        self.status = si.Status.FINISHED

    def do_checkpoint(
        self, now: datetime.datetime, callback: Callable[[Path], None]
    ) -> None:
        self.status = si.Status.PAUSED
        path = self.save(target_dir=self.spec.checkpoint_dir)
        callback(path)
        self.latest_checkpoint_time = now
        logger.info(
            f"{self} checkpointed at time step {self.time_index + 1} / {self.time_steps} ({self.percent_completed:.2f}%)"
        )
        self.status = si.Status.RUNNING

    @property
    def percent_completed(self) -> float:
        return 100 * (self.time_index + 1) / self.time_steps

    def info(self) -> si.Info:
        info = super().info()

        mem = si.Info(header="Memory Usage")

        try:
            psf_mem = self.polarization_sum_factors.nbytes
        except AttributeError:
            psf_mem = 0
        mem.add_field("Polarization Sum Factors", si.utils.bytes_to_str(psf_mem))

        info.add_info(mem)

        return info

    def save(
        self, target_dir: Optional[str] = None, file_extension: str = "sim", **kwargs
    ) -> str:
        del self.times
        return super().save(
            target_dir=target_dir, file_extension=file_extension, **kwargs
        )


class StimulatedRamanScatteringSimulation(RamanSimulation):
    def _calculate_polarization_sum_factors(self) -> np.ndarray:
        num_modes = len(self.spec.modes)

        logger.debug(f"Building two-mode inverse detuning array for {self}...")
        double_inverse_detunings = np.zeros((num_modes, num_modes), dtype=np.complex128)
        for (q, mode_q), (s, mode_s) in itertools.product(
            enumerate(self.spec.modes), repeat=2
        ):
            double_inverse_detunings[q, s] = self._double_inverse_detuning(
                mode_s.omega, mode_q.omega
            )

        logger.debug(f"Building four-mode volume ratio array for {self}...")
        mode_volume_ratios = np.zeros((num_modes, num_modes), dtype=np.complex128)
        mode_pairs = itertools.combinations_with_replacement(
            enumerate(self.spec.modes), r=2
        )
        if is_interactive_session():
            mode_pairs = tqdm(list(mode_pairs))
        for (q, mode_q), (s, mode_s) in mode_pairs:
            volume = self.spec.mode_volume_integrator.mode_volume_integral(
                (mode_q, mode_q, mode_s, mode_s)
            )

            mode_volume_ratios[q, s] = volume / self.mode_volumes[q]
            if q != s:
                mode_volume_ratios[s, q] = volume / self.mode_volumes[s]

        return np.einsum(
            "q,qs,qs->qs",
            self.polarization_prefactor,
            double_inverse_detunings,
            mode_volume_ratios,
        )

    def calculate_polarization(
        self, mode_amplitudes: np.ndarray, time: float
    ) -> np.ndarray:
        raman = np.einsum(
            "q,s,qs->q",
            mode_amplitudes,
            np.abs(mode_amplitudes) ** 2,
            self.polarization_sum_factors,
        )

        return raman


class RamanSidebandSimulation(StimulatedRamanScatteringSimulation):
    """A version of the SRS simulation that doesn't include the self-interaction, and only includes nearest-neighbour sideband interactions."""

    def _calculate_polarization_sum_factors(self) -> np.ndarray:
        num_modes = len(self.spec.modes)

        logger.debug(f"Building two-mode inverse detuning array for {self}...")
        double_inverse_detunings = np.empty((num_modes, num_modes), dtype=np.complex128)
        mode_pairs = itertools.product(enumerate(self.spec.modes), repeat=2)
        if is_interactive_session():
            mode_pairs = tqdm(list(mode_pairs))
        for (r, mode_r), (s, mode_s) in mode_pairs:
            double_inverse_detunings[r, s] = self._double_inverse_detuning(
                mode_s.omega, mode_r.omega
            )

        logger.debug(f"Building four-mode volume ratio array for {self}...")
        mode_volume_ratios = np.empty((num_modes, num_modes), dtype=np.complex128)
        mode_pairs = itertools.combinations_with_replacement(
            enumerate(self.spec.modes), r=2
        )
        if is_interactive_session():
            mode_pairs = list(tqdm(mode_pairs))
        for (r, mode_r), (s, mode_s) in mode_pairs:
            if r == s or not (r == s + 1 or r == s - 1):
                mode_volume_ratios[r, s] = 0
                mode_volume_ratios[s, r] = 0
                continue

            volume = self.spec.mode_volume_integrator.mode_volume_integral(
                (mode_r, mode_r, mode_s, mode_s)
            )

            mode_volume_ratios[r, s] = volume / self.mode_volumes[r]
            if r != s:
                mode_volume_ratios[s, r] = volume / self.mode_volumes[s]

        return np.einsum(
            "q,qs,qs->qs",
            self.polarization_prefactor,
            double_inverse_detunings,
            mode_volume_ratios,
        )


class FourWaveMixingSimulation(RamanSimulation):
    def _calculate_polarization_sum_factors(self) -> np.ndarray:
        num_modes = len(self.spec.modes)

        logger.debug(f"Building two-mode inverse detuning array for {self}...")
        double_inverse_detunings = np.zeros((num_modes, num_modes), dtype=np.complex128)
        pairs = itertools.product(enumerate(self.spec.modes), repeat=2)
        if is_interactive_session():
            pairs = tqdm(list(pairs))
        for (s, mode_s), (t, mode_t) in pairs:
            double_inverse_detunings[s, t] = self._double_inverse_detuning(
                mode_s.omega, mode_t.omega
            )

        logger.debug(f"Building four-mode volume ratio array for {self}...")
        mode_volume_ratios = np.zeros(
            (num_modes, num_modes, num_modes, num_modes), dtype=np.float64
        )
        four_modes_combinations = itertools.combinations_with_replacement(
            enumerate(self.spec.modes), r=4
        )
        if is_interactive_session():
            four_modes_combinations = tqdm(list(four_modes_combinations))
        for (
            (q, mode_q),
            (r, mode_r),
            (s, mode_s),
            (t, mode_t),
        ) in four_modes_combinations:
            modes = (mode_q, mode_r, mode_s, mode_t)
            volume = self.spec.mode_volume_integrator.mode_volume_integral(modes)

            for q_, r_, s_, t_ in itertools.permutations((q, r, s, t)):
                four_mode_detuning = np.abs(
                    self.spec.modes[r_].frequency
                    - self.spec.modes[s_].frequency
                    + self.spec.modes[t_].frequency
                    - self.spec.modes[q_].frequency
                )
                if four_mode_detuning <= self.spec.four_mode_detuning_cutoff:
                    mode_volume_ratios[q_, r_, s_, t_] = volume / self.mode_volumes[q]

        return np.einsum(
            "q,st,qrst->qrst",
            self.polarization_prefactor,
            double_inverse_detunings,
            mode_volume_ratios,
        )

    @functools.lru_cache(maxsize=4)
    def _calculate_phase_array(self, time: float) -> np.ndarray:
        """
        Caching helps some algorithms like RK4 which need to evaluate at the same time multiple times.

        Parameters
        ----------
        time

        Returns
        -------

        """
        return np.exp(-1j * time * self.mode_omegas)

    def calculate_polarization(
        self, mode_amplitudes: np.ndarray, time: float
    ) -> np.ndarray:
        phase = self._calculate_phase_array(time)
        fields = mode_amplitudes * phase
        return four_wave_polarization(fields, phase, self.polarization_sum_factors)


AUTO_CUTOFF = "AUTO_CUTOFF"


class RamanSpecification(si.Specification):
    simulation_type = RamanSimulation

    def __init__(
        self,
        name: str,
        *,
        modes: Iterable[mode.Mode],
        mode_initial_amplitudes: Optional[
            Dict[mode.Mode, Union[int, float, complex]]
        ] = None,
        mode_intrinsic_quality_factors: Dict[mode.Mode, Union[int, float]],
        mode_coupling_quality_factors: Dict[mode.Mode, Union[int, float]],
        pumps: List[pump.MonochromaticPump],
        time_initial: float = 0 * u.nsec,
        time_final: float = 100 * u.nsec,
        time_step: float = 1 * u.nsec,
        material: material.RamanMaterial = None,
        evolution_algorithm: evolve.EvolutionAlgorithm = evolve.RungeKutta4(),
        mode_volume_integrator: volume.ModeVolumeIntegrator = None,
        four_mode_detuning_cutoff=AUTO_CUTOFF,
        store_mode_amplitudes_vs_time: bool = False,
        lookback: Optional[lookback.Lookback] = None,
        freeze_lookback: bool = True,
        cached_polarization_sum_factors: Optional[np.ndarray] = None,
        checkpoints: bool = False,
        checkpoint_every: datetime.timedelta = datetime.timedelta(hours=1),
        checkpoint_dir: Optional[str] = None,
        animators=(),
        **kwargs,
    ):
        super().__init__(name, **kwargs)

        if mode_initial_amplitudes is None:
            mode_initial_amplitudes = {}

        self.modes = tuple(sorted(modes, key=lambda m: m.omega))
        self.mode_initial_amplitudes = np.array(
            [mode_initial_amplitudes.get(mode, 0) for mode in self.modes],
            dtype=np.complex128,
        )
        self.mode_intrinsic_quality_factors = np.array(
            [mode_intrinsic_quality_factors[mode] for mode in self.modes],
            dtype=np.float64,
        )
        self.mode_coupling_quality_factors = np.array(
            [mode_coupling_quality_factors[mode] for mode in self.modes],
            dtype=np.float64,
        )
        self.mode_total_quality_factors = (
            self.mode_intrinsic_quality_factors * self.mode_coupling_quality_factors
        ) / (self.mode_intrinsic_quality_factors + self.mode_coupling_quality_factors)
        self.pumps = pumps

        self.time_initial = time_initial
        self.time_final = time_final
        self.time_step = time_step

        if material is None:
            raise exceptions.MissingRamanMaterial("material cannot be None")
        self.material = material

        self.evolution_algorithm = evolution_algorithm

        if mode_volume_integrator is None:
            raise exceptions.MissingVolumeIntegrator(
                "mode_volume_integrator cannot be None"
            )
        self.mode_volume_integrator = mode_volume_integrator

        if four_mode_detuning_cutoff == AUTO_CUTOFF:
            four_mode_detuning_cutoff = 0.5 / time_step
        self.four_mode_detuning_cutoff = four_mode_detuning_cutoff

        self.store_mode_amplitudes_vs_time = store_mode_amplitudes_vs_time
        self.lookback = lookback
        self.freeze_lookback = freeze_lookback

        self.cached_polarization_sum_factors = cached_polarization_sum_factors

        self.checkpoints = checkpoints
        self.checkpoint_every = checkpoint_every
        self.checkpoint_dir = checkpoint_dir

        self.animators = animators

    def info(self) -> si.Info:
        info = super().info()

        info.add_info(self.material.info())

        info_pumps = si.Info(header="Pumps")
        for pump in self.pumps:
            info_pumps.add_info(pump.info())
        info.add_info(info_pumps)

        info_modes = si.Info(header="Modes")
        if len(self.modes) > 0:
            info_modes.add_field("Mode Type", self.modes[0].__class__.__name__)
        info_modes.add_field("Number of Modes", len(self.modes))
        info.add_info(info_modes)

        info.add_info(self.mode_volume_integrator.info())

        info_evolution = si.Info(header="Time Evolution")
        info_evolution.add_field(
            "Initial Time", fmt.quantity(self.time_initial, fmt.TIME_UNITS)
        )
        info_evolution.add_field(
            "Final Time", fmt.quantity(self.time_final, fmt.TIME_UNITS)
        )
        info_evolution.add_field(
            "Time Step", fmt.quantity(self.time_step, fmt.TIME_UNITS)
        )
        info_evolution.add_field(
            "Inverse Time Step", fmt.quantity(1 / self.time_step, fmt.FREQUENCY_UNITS)
        )
        info_evolution.add_field(
            "Four-Mode Detuning Cutoff",
            fmt.quantity(self.four_mode_detuning_cutoff, fmt.FREQUENCY_UNITS)
            if self.four_mode_detuning_cutoff is not AUTO_CUTOFF
            else "AUTO",
        )
        info_evolution.add_info(self.evolution_algorithm.info())
        info.add_info(info_evolution)

        info_checkpoint = si.Info(header="Checkpointing")
        if self.checkpoints:
            if self.checkpoint_dir is not None:
                working_in = self.checkpoint_dir
            else:
                working_in = "cwd"
            info_checkpoint.header += (
                f": every {self.checkpoint_every}, working in {working_in}"
            )
        else:
            info_checkpoint.header += ": disabled"
        info.add_info(info_checkpoint)

        if len(self.animators) > 0:
            info_animation = si.Info(header="Animation")
            for animator in self.animators:
                info_animation.add_info(animator.info())
            info.add_info(info_animation)

        return info

    def mode_info(self) -> si.Info:
        info = si.Info(header="Modes")

        for mode, q_intrinsic, q_coupling, q_total in zip(
            self.modes,
            self.mode_intrinsic_quality_factors,
            self.mode_coupling_quality_factors,
            self.mode_total_quality_factors,
        ):
            mode_info = mode.info()
            mode_info.add_field("Intrinsic Quality Factor", f"{q_intrinsic:.4g}")
            mode_info.add_field("Coupling Quality Factor", f"{q_coupling:.4g}")
            mode_info.add_field("Total Quality Factor", f"{q_total:.4g}")
            info.add_info(mode_info)

        return info


class StimulatedRamanScatteringSpecification(RamanSpecification):
    simulation_type = StimulatedRamanScatteringSimulation


class RamanSidebandSpecification(StimulatedRamanScatteringSpecification):
    simulation_type = RamanSidebandSimulation


class FourWaveMixingSpecification(RamanSpecification):
    simulation_type = FourWaveMixingSimulation


def is_interactive_session():
    import __main__ as main

    return any(
        (
            bool(getattr(sys, "ps1", sys.flags.interactive)),  # console sessions
            not hasattr(main, "__file__"),  # jupyter-like notebooks
        )
    )
