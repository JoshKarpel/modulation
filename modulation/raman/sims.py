import logging
from typing import Iterable, Dict, Union, Optional

import functools
import itertools
import datetime

import numpy as np

from tqdm import tqdm

import simulacra as si
import simulacra.units as u

from . import mode, pump, evolve, volume, material, plotter, exceptions
from .. import fmt
from .cy import four_wave_polarization

logger = logging.getLogger(__name__)


class RamanSimulation(si.Simulation):
    def __init__(self, spec):
        super().__init__(spec)

        self.latest_checkpoint_time = datetime.datetime.utcnow()

        self.time_index = 0
        self.time_steps = len(self.times)

        self.mode_omegas = np.array([m.omega for m in self.spec.modes])
        self.mode_amplitude_decay_rates = self.mode_omegas / (2 * self.spec.mode_total_quality_factors)  # wiki's definition
        self.mode_index_of_refraction = np.array([m.index_of_refraction for m in self.spec.modes])
        self.mode_epsilons = self.mode_index_of_refraction ** 2
        self.polarization_prefactor = 0.5j * self.spec.material.number_density * (self.mode_omegas / (u.epsilon_0 * self.mode_epsilons))
        self.mode_amplitudes = self.spec.mode_initial_amplitudes.copy()
        self.mode_to_index = {mode: idx for idx, mode in enumerate(self.spec.modes)}

        if self.spec.store_mode_amplitudes_vs_time:
            self.mode_amplitudes_vs_time = np.empty((len(self.times), len(self.mode_amplitudes)), dtype = np.complex128)

        self.mode_volumes_inside_resonator = np.array([m.mode_volume_inside_resonator for m in self.spec.modes])
        self.mode_volumes_outside_resonator = np.array([m.mode_volume_outside_resonator for m in self.spec.modes])
        self.mode_volumes = self.mode_volumes_inside_resonator + self.mode_volumes_outside_resonator

        self.mode_background_magnitudes = np.sqrt(self.mode_photon_energy / self.mode_energy_prefactor)  # one photon per mode

        self.pump_prefactor = np.sqrt(self.mode_omegas / self.spec.mode_coupling_quality_factors) / np.sqrt(self.mode_energy_prefactor)

        if self.spec.cached_polarization_sum_factors is None:
            self.polarization_sum_factors = self._calculate_polarization_sum_factors()
        else:
            logger.debug(f'Using cached polarization sum factors for {self}')
            self.polarization_sum_factors = self.spec.cached_polarization_sum_factors

        self.plot = plotter.RamanSimulationPlotter(self)

    @property
    def times(self):
        total_time = self.spec.time_final - self.spec.time_initial
        times = np.linspace(
            self.spec.time_initial,
            self.spec.time_final,
            int(total_time / self.spec.time_step) + 1,
        )

        return times

    @property
    def time(self):
        return self.times[self.time_index]

    @property
    def available_animation_frames(self):
        return self.time_steps

    def _calculate_polarization_sum_factors(self):
        raise NotImplementedError

    @property
    def mode_photon_energy(self):
        return u.hbar * self.mode_omegas

    @property
    def mode_energy_prefactor(self):
        electric_inside = 0.5 * u.epsilon_0 * self.mode_epsilons * self.mode_volumes_inside_resonator
        electric_outside = 0.5 * u.epsilon_0 * self.mode_volumes_outside_resonator

        return electric_inside + electric_outside

    # @property
    # def mode_intensity_prefactor(self):
    #     """Multiply by E_q^2 and the mode shape^2 to get the intensity in space."""
    #     return 0.5 * u.c * u.epsilon_0 * np.sqrt(self.mode_epsilons)

    def mode_energies(self, mode_amplitudes):
        return self.mode_energy_prefactor * (np.abs(mode_amplitudes) ** 2)

    def mode_output_powers(self, mode_amplitudes):
        return self.mode_energies(mode_amplitudes) * self.mode_omegas / self.spec.mode_coupling_quality_factors

    def _two_photon_detuning(self, omega_x, omega_y):
        return (omega_x - omega_y) - (self.spec.material.modulation_omega + (1j * self.spec.material.raman_linewidth))

    def _double_inverse_detuning(self, omega_x, omega_y):
        return np.conj(1 / self._two_photon_detuning(omega_x, omega_y)) + (1 / self._two_photon_detuning(omega_y, omega_x))

    def modes_with_amplitudes(self):
        yield from zip(self.spec.modes, self.mode_amplitudes)

    def calculate_total_derivative(self, mode_amplitudes, time):
        polarization = self.polarization_prefactor * self.calculate_polarization(mode_amplitudes, time)
        pump = self.calculate_pumps(time)
        decay = -self.mode_amplitude_decay_rates * mode_amplitudes

        return polarization + pump + decay

    @functools.lru_cache(maxsize = 4)
    def calculate_pumps(self, time):
        return self.pump_prefactor * np.sqrt([pump.power(time) for pump in self.spec.mode_pumps], dtype = np.float64)

    def calculate_polarization(self, mode_amplitudes, time):
        raise NotImplementedError

    def generate_background_amplitudes(self):
        return self.mode_background_magnitudes * np.exp(1j * u.twopi * np.random.random(self.mode_amplitudes.shape))

    def run(self, show_progress_bar: bool = False):
        self.status = si.Status.RUNNING

        try:
            for animator in self.spec.animators:
                animator.initialize(self)

            if show_progress_bar:
                pbar = tqdm(total = self.time_steps)

            times = self.times
            while True:
                # spontaneous raman
                self.mode_amplitudes[:] = np.where(
                    np.abs(self.mode_amplitudes) >= self.mode_background_magnitudes,
                    self.mode_amplitudes,
                    self.generate_background_amplitudes(),
                )

                if self.spec.store_mode_amplitudes_vs_time:
                    self.mode_amplitudes_vs_time[self.time_index, :] = self.mode_amplitudes

                for animator in self.spec.animators:
                    if self.time_index == 0 or self.time_index == self.time_steps or self.time_index % animator.decimation == 0:
                        animator.send_frame_to_ffmpeg()

                if show_progress_bar:
                    pbar.update(1)
                if self.time_index == self.time_steps - 1:
                    break

                self.time_index += 1

                # pump, decay, polarization
                self.mode_amplitudes[:] = self.spec.evolution_algorithm.evolve(
                    sim = self,
                    mode_amplitudes = self.mode_amplitudes,
                    time_initial = times[self.time_index - 1],
                    time_final = times[self.time_index],
                )

                if self.spec.checkpoints:
                    now = datetime.datetime.utcnow()
                    if (now - self.latest_checkpoint_time) > self.spec.checkpoint_every:
                        self.do_checkpoint(now)

            if show_progress_bar:
                pbar.close()
        finally:
            for animator in self.spec.animators:
                animator.cleanup()

            self.spec.animators = ()

        self.status = si.Status.FINISHED

    def do_checkpoint(self, now):
        self.status = si.Status.PAUSED
        self.save(target_dir = self.spec.checkpoint_dir)
        self.latest_checkpoint_time = now
        logger.info(f'{self} checkpointed at time index {self.time_index} / {self.time_steps - 1} ({self.percent_completed}%)')
        self.status = si.Status.RUNNING

    @property
    def percent_completed(self):
        return round(100 * self.time_index / (self.time_steps - 1), 2)


class StimulatedRamanScatteringSimulation(RamanSimulation):
    def _calculate_polarization_sum_factors(self):
        num_modes = len(self.spec.modes)

        logger.debug(f'Building two-mode inverse detuning array for {self}...')
        double_inverse_detunings = np.empty((num_modes, num_modes), dtype = np.complex128)
        for (r, mode_r), (s, mode_s) in itertools.product(enumerate(self.spec.modes), repeat = 2):
            double_inverse_detunings[r, s] = self._double_inverse_detuning(mode_s.omega, mode_r.omega)

        logger.debug(f'Building four-mode volume ratio array for {self}...')
        mode_volume_ratios = np.empty((num_modes, num_modes), dtype = np.complex128)
        mode_pairs = list(itertools.combinations_with_replacement(enumerate(self.spec.modes), r = 2))
        for (r, mode_r), (s, mode_s) in mode_pairs:
            volume = self.spec.mode_volume_integrator.mode_volume_integral((mode_r, mode_r, mode_s, mode_s))

            mode_volume_ratios[r, s] = volume / self.mode_volumes[r]
            if r != s:
                mode_volume_ratios[s, r] = volume / self.mode_volumes[s]

        return self.spec.material.raman_prefactor * double_inverse_detunings * mode_volume_ratios

    def calculate_polarization(self, mode_amplitudes, time):
        raman = np.einsum(
            'r,s,rs->r',
            mode_amplitudes,
            np.abs(mode_amplitudes) ** 2,
            self.polarization_sum_factors,
        )

        return raman


class FourWaveMixingSimulation(RamanSimulation):
    def _calculate_polarization_sum_factors(self):
        num_modes = len(self.spec.modes)

        logger.debug(f'Building two-mode inverse detuning array for {self}...')
        double_inverse_detunings = np.empty((num_modes, num_modes), dtype = np.complex128)
        all_mode_pairs = list(itertools.product(enumerate(self.spec.modes), repeat = 2))
        for (s, mode_s), (t, mode_t) in tqdm(all_mode_pairs):
            double_inverse_detunings[s, t] = self._double_inverse_detuning(mode_s.omega, mode_t.omega)

        logger.debug(f'Building four-mode volume ratio array for {self}...')
        mode_volume_ratios = np.empty((num_modes, num_modes, num_modes, num_modes), dtype = np.float64)
        four_modes_combinations = list(itertools.combinations_with_replacement(enumerate(self.spec.modes), r = 4))
        for (q, mode_q), (r, mode_r), (s, mode_s), (t, mode_t) in tqdm(four_modes_combinations):
            volume = self.spec.mode_volume_integrator.mode_volume_integral((mode_q, mode_r, mode_s, mode_t))

            for q_, r_, s_, t_ in itertools.permutations((q, r, s, t)):
                mode_volume_ratios[q_, r_, s_, t_] = volume / self.mode_volumes[q]

        logger.debug(f'Building four-mode frequency difference array for {self}...')
        omega_q, omega_r, omega_s, omega_t = np.meshgrid(
            self.mode_omegas,
            self.mode_omegas,
            self.mode_omegas,
            self.mode_omegas,
            indexing = 'ij',
            sparse = True,
        )
        self.frequency_differences = omega_r + omega_t - omega_s - omega_q

        return self.spec.material.raman_prefactor * np.einsum(
            'st,qrst->qrst',
            double_inverse_detunings,
            mode_volume_ratios,
        )

    @functools.lru_cache(maxsize = 4)
    def _calculate_phase_array(self, time):
        """
        Caching helps some algorithms like RK4 which need to evaluate at the same time multiple times.

        Parameters
        ----------
        time

        Returns
        -------

        """
        minus_omega_t = -time * self.mode_omegas
        return np.cos(minus_omega_t) + (1j * np.sin(minus_omega_t))

    def calculate_polarization(self, mode_amplitudes, time):
        phase = self._calculate_phase_array(time)
        fields = mode_amplitudes * phase
        return four_wave_polarization(
            fields,
            phase,
            self.polarization_sum_factors,
        )


class RamanSpecification(si.Specification):
    simulation_type = RamanSimulation

    def __init__(
        self,
        name,
        *,
        modes: Iterable[mode.Mode],
        mode_initial_amplitudes: Dict[mode.Mode, Union[int, float, complex]],
        mode_intrinsic_quality_factors: Dict[mode.Mode, Union[int, float]],
        mode_coupling_quality_factors: Dict[mode.Mode, Union[int, float]],
        mode_pumps: Dict[mode.Mode, pump.Pump],
        time_initial: float = 0 * u.nsec,
        time_final: float = 100 * u.nsec,
        time_step: float = 1 * u.nsec,
        material: material.RamanMaterial = None,
        evolution_algorithm: evolve.EvolutionAlgorithm = evolve.RungeKutta4(),
        mode_volume_integrator: volume.ModeVolumeIntegrator = None,
        store_mode_amplitudes_vs_time: bool = False,
        cached_polarization_sum_factors = None,
        checkpoints: bool = False,
        checkpoint_every: datetime.timedelta = datetime.timedelta(hours = 1),
        checkpoint_dir: Optional[str] = None,
        animators = (),
        **kwargs,
    ):
        super().__init__(name, **kwargs)

        self.modes = tuple(sorted(modes, key = lambda m: m.omega))
        self.mode_initial_amplitudes = np.array([mode_initial_amplitudes.get(mode, 0) for mode in self.modes], dtype = np.complex128)
        self.mode_intrinsic_quality_factors = np.array([mode_intrinsic_quality_factors[mode] for mode in self.modes], dtype = np.float64)
        self.mode_coupling_quality_factors = np.array([mode_coupling_quality_factors[mode] for mode in self.modes], dtype = np.float64)
        self.mode_total_quality_factors = (self.mode_intrinsic_quality_factors * self.mode_coupling_quality_factors) / (self.mode_intrinsic_quality_factors + self.mode_coupling_quality_factors)
        self.mode_pumps = [mode_pumps.get(mode, pump.ConstantPump(power = 0)) for mode in self.modes]

        self.time_initial = time_initial
        self.time_final = time_final
        self.time_step = time_step

        if material is None:
            raise exceptions.MissingRamanMaterial('material cannot be None')
        self.material = material
        self.evolution_algorithm = evolution_algorithm
        if mode_volume_integrator is None:
            raise exceptions.MissingVolumeIntegrator('mode_volume_integrator cannot be None')
        self.mode_volume_integrator = mode_volume_integrator

        self.store_mode_amplitudes_vs_time = store_mode_amplitudes_vs_time

        self.cached_polarization_sum_factors = cached_polarization_sum_factors

        self.checkpoints = checkpoints
        self.checkpoint_every = checkpoint_every
        self.checkpoint_dir = checkpoint_dir

        self.animators = animators

    def info(self) -> si.Info:
        info = super().info()

        info.add_info(self.material.info())

        info_modes = si.Info(header = 'Modes')
        if len(self.modes) > 0:
            info_modes.add_field('Mode Type', self.modes[0].__class__.__name__)
        info_modes.add_field('Number of Modes', len(self.modes))
        info.add_info(info_modes)

        info.add_info(self.mode_volume_integrator.info())

        info_evolution = si.Info(header = 'Time Evolution')
        info_evolution.add_field('Initial Time', fmt.quantity(self.time_initial, fmt.TIME_UNITS))
        info_evolution.add_field('Final Time', fmt.quantity(self.time_final, fmt.TIME_UNITS))
        info_evolution.add_field('Time Step', fmt.quantity(self.time_step, fmt.TIME_UNITS))
        info_evolution.add_field('Inverse Time Step', fmt.quantity(1 / self.time_step, fmt.FREQUENCY_UNITS))
        info_evolution.add_info(self.evolution_algorithm.info())
        info.add_info(info_evolution)

        info_checkpoint = si.Info(header = 'Checkpointing')
        if self.checkpoints:
            if self.checkpoint_dir is not None:
                working_in = self.checkpoint_dir
            else:
                working_in = 'cwd'
            info_checkpoint.header += f': every {self.checkpoint_every} time steps, working in {working_in}'
        else:
            info_checkpoint.header += ': disabled'
        info.add_info(info_checkpoint)

        if len(self.animators) > 0:
            info_animation = si.Info(header = 'Animation')
            for animator in self.animators:
                info_animation.add_info(animator.info())
            info.add_info(info_animation)

        return info

    def mode_info(self) -> si.Info:
        info = si.Info(header = 'Modes')

        for mode, q_intrinsic, q_coupling, q_total, pump in zip(
            self.modes,
            self.mode_intrinsic_quality_factors,
            self.mode_coupling_quality_factors,
            self.mode_total_quality_factors,
            self.mode_pumps,
        ):
            mode_info = mode.info()
            mode_info.add_field('Intrinsic Quality Factor', f'{q_intrinsic:.4g}')
            mode_info.add_field('Coupling Quality Factor', f'{q_coupling:.4g}')
            mode_info.add_field('Total Quality Factor', f'{q_total:.4g}')
            mode_info.add_info(pump.info())
            info.add_info(mode_info)

        return info


class StimulatedRamanScatteringSpecification(RamanSpecification):
    simulation_type = StimulatedRamanScatteringSimulation


class FourWaveMixingSpecification(RamanSpecification):
    simulation_type = FourWaveMixingSimulation
