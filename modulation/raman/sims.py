import functools
from typing import Iterable, Dict, Union, Optional
import logging

import itertools
import datetime
import collections

import numpy as np

from tqdm import tqdm

import simulacra as si
import simulacra.units as u

from . import mode, pumps, evolve, volume, exceptions
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
        self.mode_index_of_refraction = np.array([m.mode_index_of_refraction for m in self.spec.modes])
        self.mode_epsilons = self.mode_index_of_refraction ** 2
        self.polarization_prefactor = 0.5j * self.spec.number_density * (self.mode_omegas / (u.epsilon_0 * self.mode_epsilons))
        self.degenerate_modes_by_frequency = self._degenerate_modes(self.spec.modes)
        self.mode_amplitudes = self.spec.mode_initial_amplitudes.copy()
        self.mode_to_index = {mode: idx for idx, mode in enumerate(self.spec.modes)}

        if self.spec.store_mode_amplitudes_vs_time:
            self.mode_amplitudes_vs_time = np.empty((len(self.times), len(self.mode_amplitudes)), dtype = np.complex128)

        self.mode_volumes_within_R = np.array([m.mode_volume_within_R for m in self.spec.modes])
        self.mode_volumes_outside_R = np.array([m.mode_volume_outside_R for m in self.spec.modes])
        self.mode_volumes = self.mode_volumes_within_R + self.mode_volumes_outside_R

        self.mode_background_magnitudes = np.sqrt(self.mode_photon_energy / self.mode_energy_prefactor)  # one photon per mode

        self.pump_prefactor = np.sqrt(self.mode_omegas / self.spec.mode_coupling_quality_factors) / np.sqrt(self.mode_energy_prefactor)

        if self.spec.cached_polarization_sum_factors is None:
            self.polarization_sum_factors = self._calculate_polarization_sum_factors()
        else:
            logger.debug(f'Using cached polarization sum factors for {self}')
            self.polarization_sum_factors = self.spec.cached_polarization_sum_factors

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
        electric_inside = 0.5 * u.epsilon_0 * self.mode_epsilons * self.mode_volumes_within_R
        electric_outside = 0.5 * u.epsilon_0 * self.mode_volumes_outside_R

        return electric_inside + electric_outside

    # @property
    # def mode_intensity_prefactor(self):
    #     """Multiply by E_q^2 and the mode shape^2 to get the intensity in space."""
    #     return 0.5 * u.c * u.epsilon_0 * np.sqrt(self.mode_epsilons)

    def mode_energies(self, mode_amplitudes):
        # in principle, we should include the outer part
        # but it never has any significant amount of energy in it

        return self.mode_energy_prefactor * (np.abs(mode_amplitudes) ** 2)

    def mode_output_powers(self, mode_amplitudes):
        return self.mode_energies(mode_amplitudes) * self.mode_omegas / self.spec.mode_coupling_quality_factors

    def _degenerate_modes(self, modes):
        degenerate_modes = collections.defaultdict(list)

        for mode in modes:
            degenerate_modes[mode.omega].append(mode)

        return degenerate_modes

    def two_photon_detuning(self, omega_x, omega_y):
        return (omega_x - omega_y) - (self.spec.modulation_omega + (1j * self.spec.gamma_b))

    def double_inverse_detuning(self, omega_x, omega_y):
        return np.conj(1 / self.two_photon_detuning(omega_x, omega_y)) + (1 / self.two_photon_detuning(omega_y, omega_x))

    def modes_with_amplitudes(self):
        yield from zip(self.spec.modes, self.mode_amplitudes)

    def calculate_total_derivative(self, mode_amplitudes, time):
        polarization = self.polarization_prefactor * self.calculate_polarization(mode_amplitudes, time)
        pump = self.calculate_pumps(time)
        decay = -self.mode_amplitude_decay_rates * mode_amplitudes

        return polarization + pump + decay

    @functools.lru_cache(maxsize = 4)
    def calculate_pumps(self, time):
        return self.pump_prefactor * np.sqrt([pump.power(time) for pump in self.spec.mode_pump_rates], dtype = np.float64)

    def calculate_polarization(self, mode_amplitudes, time):
        raise NotImplementedError

    def generate_background_amplitudes(self):
        return self.mode_background_magnitudes * np.exp(1j * u.twopi * np.random.random(self.mode_amplitudes.shape))

    def run(self, progress_bar: bool = False):
        self.status = si.Status.RUNNING

        try:
            for animator in self.spec.animators:
                animator.initialize(self)

            if progress_bar:
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

                if progress_bar:
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

            if progress_bar:
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

    def plot_mode_magnitudes_vs_time(
        self,
        time_unit = 'nsec',
        magnitude_unit = 'V_per_m',
        y_log_axis = True,
        y_pad = .05,
        y_log_pad = 10,
        mode_filter = None,
        mode_kwargs = None,
        average_over = 1 * u.nsec,
        **kwargs,
    ):
        if mode_filter is None:
            mode_filter = lambda sim, q, mode: True

        if mode_kwargs is None:
            mode_kwargs = lambda sim, q, mode: {}

        mode_numbers = [q for mode, q in self.mode_to_index.items() if mode_filter(self, q, mode)]

        x = self.times
        y = [np.abs(self.mode_amplitudes_vs_time[:, q]) for q in mode_numbers]

        if average_over is not None:
            l = self.mode_amplitudes_vs_time[:, 0].size
            R = int(1 * u.nsec / self.spec.time_step)
            pad_size = int((np.ceil(l / R) * R) - l)

            x = np.append(x, np.zeros(pad_size) * np.NaN).reshape((-1, R)).mean(axis = 1)
            y = [np.append(yy, np.zeros(pad_size) * np.NaN).reshape((-1, R)).mean(axis = 1) for yy in y]

        si.vis.xy_plot(
            f'{self.name}__mode_magnitudes_vs_time',
            x,
            *y,
            line_labels = [
                fr'${self.spec.modes[q].tex}$'
                for q in mode_numbers
            ],
            line_kwargs = [mode_kwargs(self, q, self.spec.modes[q]) for q in mode_numbers],
            x_unit = time_unit,
            x_label = r'$t$',
            y_unit = magnitude_unit,
            y_label = r'$\left| \mathcal{E}_q(t) \right|$',
            y_log_axis = y_log_axis,
            y_pad = y_pad,
            y_log_pad = y_log_pad,
            legend_on_right = True,
            **kwargs,
        )

    def plot_mode_energies_vs_time(
        self,
        time_unit = 'nsec',
        energy_unit = 'pJ',
        y_log_axis = True,
        y_pad = .05,
        y_log_pad = 10,
        mode_filter = None,
        mode_kwargs = None,
        average_over = 1 * u.nsec,
        **kwargs,
    ):
        if mode_filter is None:
            mode_filter = lambda sim, q, mode: True

        if mode_kwargs is None:
            mode_kwargs = lambda sim, q, mode: {}

        mode_numbers = [q for mode, q in self.mode_to_index.items() if mode_filter(self, q, mode)]

        x = self.times
        y = [self.mode_energies(self.mode_amplitudes_vs_time)[:, q] for q in mode_numbers]

        if average_over is not None:
            l = self.mode_amplitudes_vs_time[:, 0].size
            R = int(1 * u.nsec / self.spec.time_step)
            pad_size = int((np.ceil(l / R) * R) - l)

            x = np.append(x, np.zeros(pad_size) * np.NaN).reshape((-1, R)).mean(axis = 1)
            y = [np.append(yy, np.zeros(pad_size) * np.NaN).reshape((-1, R)).mean(axis = 1) for yy in y]

        si.vis.xy_plot(
            f'{self.name}__mode_energies_vs_time',
            x,
            *y,
            line_labels = [
                fr'${self.spec.modes[q].tex}$'
                for q in mode_numbers
            ],
            line_kwargs = [mode_kwargs(self, q, self.spec.modes[q]) for q in mode_numbers],
            x_unit = time_unit,
            x_label = r'$t$',
            y_unit = energy_unit,
            y_label = r'$U_q(t)$',
            y_log_axis = y_log_axis,
            legend_on_right = True,
            y_pad = y_pad,
            y_log_pad = y_log_pad,
            **kwargs,
        )

    def plot_mode_output_powers_vs_time(
        self,
        time_unit = 'nsec',
        power_unit = 'mW',
        y_log_axis = True,
        y_pad = .05,
        y_log_pad = 10,
        mode_filter = None,
        mode_kwargs = None,
        average_over = 1 * u.nsec,
        **kwargs,
    ):
        if mode_filter is None:
            mode_filter = lambda sim, q, mode: True

        if mode_kwargs is None:
            mode_kwargs = lambda sim, q, mode: {}

        mode_numbers = [q for mode, q in self.mode_to_index.items() if mode_filter(self, q, mode)]

        x = self.times
        y = [self.mode_output_powers(self.mode_amplitudes_vs_time)[:, q] for q in mode_numbers]

        if average_over is not None:
            l = self.mode_amplitudes_vs_time[:, 0].size
            R = int(1 * u.nsec / self.spec.time_step)
            pad_size = int((np.ceil(l / R) * R) - l)

            x = np.append(x, np.zeros(pad_size) * np.NaN).reshape((-1, R)).mean(axis = 1)
            y = [np.append(yy, np.zeros(pad_size) * np.NaN).reshape((-1, R)).mean(axis = 1) for yy in y]

        si.vis.xy_plot(
            f'{self.name}__mode_output_powers_vs_time',
            x,
            *y,
            line_labels = [
                fr'${self.spec.modes[q].tex}$'
                for q in mode_numbers
            ],
            line_kwargs = [mode_kwargs(self, q, self.spec.modes[q]) for q in mode_numbers],
            x_unit = time_unit,
            x_label = r'$t$',
            y_unit = power_unit,
            y_label = r'$P^{\mathrm{out}}_q(t)$',
            y_log_axis = y_log_axis,
            legend_on_right = True,
            y_pad = y_pad,
            y_log_pad = y_log_pad,
            **kwargs,
        )

    def stackplot_mode_energies_vs_time(
        self,
        time_unit = 'nsec',
        energy_unit = 'pJ',
        y_log_axis = False,
        y_pad = .05,
        y_log_pad = 10,
        mode_filter = None,
        mode_kwargs = None,
        **kwargs,
    ):
        if mode_filter is None:
            mode_filter = lambda sim, q, mode: True

        if mode_kwargs is None:
            mode_kwargs = lambda sim, q, mode: {}

        mode_numbers = [q for mode, q in self.mode_to_index.items() if mode_filter(self, q, mode)]

        si.vis.xy_stackplot(
            f'{self.name}__mode_energies_vs_time__stacked',
            self.times,
            *[self.mode_energies(self.mode_amplitudes_vs_time)[:, q] for q in mode_numbers],
            line_labels = [
                fr'${self.spec.modes[q].tex}$'
                for q in mode_numbers
            ],
            line_kwargs = [mode_kwargs(self, q, self.spec.modes[q]) for q in mode_numbers],
            x_unit = time_unit,
            x_label = r'$t$',
            y_unit = energy_unit,
            y_label = r'$U_q(t)$',
            y_log_axis = y_log_axis,
            legend_on_right = True,
            y_pad = y_pad,
            y_log_pad = y_log_pad,
            **kwargs,
        )

    def plot_mode_photon_counts_vs_time(
        self,
        time_unit = 'nsec',
        y_log_axis = True,
        y_pad = .05,
        y_log_pad = 10,
        mode_filter = None,
        mode_kwargs = None,
        average_over = 1 * u.nsec,
        **kwargs,
    ):
        if mode_filter is None:
            mode_filter = lambda sim, q, mode: True

        if mode_kwargs is None:
            mode_kwargs = lambda sim, q, mode: {}

        mode_numbers = [q for mode, q in self.mode_to_index.items() if mode_filter(self, q, mode)]

        x = self.times
        y = [self.mode_energies(self.mode_amplitudes_vs_time)[:, q] / self.mode_photon_energy[q] for q in mode_numbers]

        if average_over is not None:
            l = self.mode_amplitudes_vs_time[:, 0].size
            R = int(1 * u.nsec / self.spec.time_step)
            pad_size = int((np.ceil(l / R) * R) - l)

            x = np.append(x, np.zeros(pad_size) * np.NaN).reshape((-1, R)).mean(axis = 1)
            y = [np.append(yy, np.zeros(pad_size) * np.NaN).reshape((-1, R)).mean(axis = 1) for yy in y]

        si.vis.xy_plot(
            f'{self.name}__mode_photon_counts_vs_time',
            x,
            *y,
            line_labels = [
                fr'${self.spec.modes[q].tex}$'
                for q in mode_numbers
            ],
            line_kwargs = [mode_kwargs(self, q, self.spec.modes[q]) for q in mode_numbers],
            x_unit = time_unit,
            x_label = r'$t$',
            y_label = r'$U_q(t) \, / \, \hbar \omega_q$',
            y_log_axis = y_log_axis,
            legend_on_right = True,
            y_pad = y_pad,
            y_log_pad = y_log_pad,
            **kwargs,
        )

    def stackplot_mode_photon_counts_vs_time(
        self,
        time_unit = 'nsec',
        y_pad = .05,
        y_log_pad = 10,
        mode_filter = None,
        mode_kwargs = None,
        **kwargs,
    ):
        if mode_filter is None:
            mode_filter = lambda sim, q, mode: True

        if mode_kwargs is None:
            mode_kwargs = lambda sim, q, mode: {}

        mode_numbers = [q for mode, q in self.mode_to_index.items() if mode_filter(self, q, mode)]

        si.vis.xy_stackplot(
            f'{self.name}__mode_photon_counts_vs_time__stacked',
            self.times,
            *[self.mode_energies(self.mode_amplitudes_vs_time)[:, q] / self.mode_photon_energy[q] for q in mode_numbers],
            line_labels = [
                fr'${self.spec.modes[q].tex}$'
                for q in mode_numbers
            ],
            line_kwargs = [mode_kwargs(self, q, self.spec.modes[q]) for q in mode_numbers],
            x_unit = time_unit,
            x_label = r'$t$',
            y_label = r'$U_q(t) \, / \, \hbar \omega_q$',
            legend_on_right = True,
            y_pad = y_pad,
            y_log_pad = y_log_pad,
            **kwargs,
        )


class StimulatedRamanScatteringSimulation(RamanSimulation):
    def _calculate_polarization_sum_factors(self):
        num_modes = len(self.spec.modes)

        logger.debug(f'Building two-mode inverse detuning array for {self}...')
        double_inverse_detunings = np.empty((num_modes, num_modes), dtype = np.complex128)
        for (r, mode_r), (s, mode_s) in itertools.product(enumerate(self.spec.modes), repeat = 2):
            double_inverse_detunings[r, s] = self.double_inverse_detuning(mode_s.omega, mode_r.omega)

        logger.debug(f'Building four-mode volume ratio array for {self}...')
        mode_volume_ratios = np.empty((num_modes, num_modes), dtype = np.complex128)
        mode_pairs = list(itertools.combinations_with_replacement(enumerate(self.spec.modes), r = 2))
        for (r, mode_r), (s, mode_s) in tqdm(mode_pairs):
            volume = self.spec.mode_volume_integrator.mode_volume_integral_inside((mode_r, mode_r, mode_s, mode_s), R = mode_r.microsphere_radius)

            mode_volume_ratios[r, s] = volume / self.mode_volumes[r]
            if r != s:
                mode_volume_ratios[s, r] = volume / self.mode_volumes[s]

        return self.spec.raman_prefactor * double_inverse_detunings * mode_volume_ratios

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
            double_inverse_detunings[s, t] = self.double_inverse_detuning(mode_s.omega, mode_t.omega)

        logger.debug(f'Building four-mode volume ratio array for {self}...')
        mode_volume_ratios = np.empty((num_modes, num_modes, num_modes, num_modes), dtype = np.float64)
        four_modes_combinations = list(itertools.combinations_with_replacement(enumerate(self.spec.modes), r = 4))
        for (q, mode_q), (r, mode_r), (s, mode_s), (t, mode_t) in tqdm(four_modes_combinations):
            volume = self.spec.mode_volume_integrator.mode_volume_integral_inside(
                (mode_q, mode_r, mode_s, mode_t),
                R = mode_r.microsphere_radius,
            )

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

        return self.spec.raman_prefactor * np.einsum(
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
        mode_pump_rates: Dict[mode.Mode, pumps.Pump],
        modulation_omega: float,
        C: complex,
        gamma_b: float,
        number_density: float,
        time_initial: float = 0 * u.nsec,
        time_final: float = 100 * u.nsec,
        time_step: float = 1 * u.nsec,
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
        self.mode_pump_rates = [mode_pump_rates.get(mode, pumps.ConstantPump(power = 0)) for mode in self.modes]

        self.modulation_omega = modulation_omega

        self.C = C
        self.raman_prefactor = (C ** 2) / (4 * (u.hbar ** 3))
        self.gamma_b = gamma_b

        self.number_density = number_density

        self.time_initial = time_initial
        self.time_final = time_final
        self.time_step = time_step

        self.evolution_algorithm = evolution_algorithm
        if mode_volume_integrator is None:
            raise exceptions.RamanException('evolution algorithm cannot be None')
        self.mode_volume_integrator = mode_volume_integrator

        self.store_mode_amplitudes_vs_time = store_mode_amplitudes_vs_time

        self.cached_polarization_sum_factors = cached_polarization_sum_factors

        self.checkpoints = checkpoints
        self.checkpoint_every = checkpoint_every
        self.checkpoint_dir = checkpoint_dir

        self.animators = animators


class StimulatedRamanScatteringSpecification(RamanSpecification):
    simulation_type = StimulatedRamanScatteringSimulation


class FourWaveMixingSpecification(RamanSpecification):
    simulation_type = FourWaveMixingSimulation
