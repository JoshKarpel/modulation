import numpy as np

import simulacra as si
import simulacra.units as u


class RamanSimulationPlotter:
    def __init__(self, sim):
        self.sim = sim

    def mode_magnitudes_vs_time(
        self,
        time_unit = 'nsec',
        magnitude_unit = 'V_per_m',
        y_log_axis = True,
        mode_filter = None,
        mode_kwargs = None,
        average_over = None,
        **kwargs,
    ):
        if mode_filter is None:
            mode_filter = lambda sim, q, mode: True

        if mode_kwargs is None:
            mode_kwargs = lambda sim, q, mode: {}

        mode_numbers = [q for mode, q in self.sim.mode_to_index.items() if mode_filter(self, q, mode)]

        x = self.sim.times
        y = [np.abs(self.sim.mode_amplitudes_vs_time[:, q]) for q in mode_numbers]

        if average_over is not None:
            l = self.sim.mode_amplitudes_vs_time[:, 0].size
            R = int(average_over / self.sim.spec.time_step)
            pad_size = int((np.ceil(l / R) * R) - l)

            x = np.append(x, np.zeros(pad_size) * np.NaN).reshape((-1, R)).mean(axis = 1)
            y = [np.append(yy, np.zeros(pad_size) * np.NaN).reshape((-1, R)).mean(axis = 1) for yy in y]

        si.vis.xy_plot(
            f'{self.sim.name}__mode_magnitudes_vs_time',
            x,
            *y,
            line_labels = [
                fr'${self.sim.spec.modes[q].tex}$'
                for q in mode_numbers
            ],
            line_kwargs = [mode_kwargs(self, q, self.sim.spec.modes[q]) for q in mode_numbers],
            x_unit = time_unit,
            x_label = r'$t$',
            y_unit = magnitude_unit,
            y_label = r'$\left| \mathcal{E}_q(t) \right|$',
            y_log_axis = y_log_axis,
            legend_on_right = True,
            **kwargs,
        )

    def mode_energies_vs_time(
        self,
        time_unit = 'nsec',
        energy_unit = 'pJ',
        y_log_axis = True,
        y_pad = .05,
        y_log_pad = 10,
        mode_filter = None,
        mode_kwargs = None,
        average_over = None,
        **kwargs,
    ):
        if mode_filter is None:
            mode_filter = lambda sim, q, mode: True

        if mode_kwargs is None:
            mode_kwargs = lambda sim, q, mode: {}

        mode_numbers = [q for mode, q in self.sim.mode_to_index.items() if mode_filter(self, q, mode)]

        x = self.sim.times
        y = [self.sim.mode_energies(self.sim.mode_amplitudes_vs_time)[:, q] for q in mode_numbers]

        if average_over is not None:
            l = self.sim.mode_amplitudes_vs_time[:, 0].size
            R = int(average_over / self.sim.spec.time_step)
            pad_size = int((np.ceil(l / R) * R) - l)

            x = np.append(x, np.zeros(pad_size) * np.NaN).reshape((-1, R)).mean(axis = 1)
            y = [np.append(yy, np.zeros(pad_size) * np.NaN).reshape((-1, R)).mean(axis = 1) for yy in y]

        si.vis.xy_plot(
            f'{self.sim.name}__mode_energies_vs_time',
            x,
            *y,
            line_labels = [
                fr'${self.sim.spec.modes[q].tex}$'
                for q in mode_numbers
            ],
            line_kwargs = [mode_kwargs(self, q, self.sim.spec.modes[q]) for q in mode_numbers],
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

    def mode_output_powers_vs_time(
        self,
        time_unit = 'nsec',
        power_unit = 'mW',
        y_log_axis = True,
        y_pad = .05,
        y_log_pad = 10,
        mode_filter = None,
        mode_kwargs = None,
        average_over = None,
        **kwargs,
    ):
        if mode_filter is None:
            mode_filter = lambda sim, q, mode: True

        if mode_kwargs is None:
            mode_kwargs = lambda sim, q, mode: {}

        mode_numbers = [q for mode, q in self.sim.mode_to_index.items() if mode_filter(self, q, mode)]

        x = self.sim.times
        y = [self.sim.mode_output_powers(self.sim.mode_amplitudes_vs_time)[:, q] for q in mode_numbers]

        if average_over is not None:
            l = self.sim.mode_amplitudes_vs_time[:, 0].size
            R = int(average_over / self.sim.spec.time_step)
            pad_size = int((np.ceil(l / R) * R) - l)

            x = np.append(x, np.zeros(pad_size) * np.NaN).reshape((-1, R)).mean(axis = 1)
            y = [np.append(yy, np.zeros(pad_size) * np.NaN).reshape((-1, R)).mean(axis = 1) for yy in y]

        si.vis.xy_plot(
            f'{self.sim.name}__mode_output_powers_vs_time',
            x,
            *y,
            line_labels = [
                fr'${self.sim.spec.modes[q].tex}$'
                for q in mode_numbers
            ],
            line_kwargs = [mode_kwargs(self, q, self.sim.spec.modes[q]) for q in mode_numbers],
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

    def stack_mode_energies_vs_time(
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

        mode_numbers = [q for mode, q in self.sim.mode_to_index.items() if mode_filter(self, q, mode)]

        si.vis.xy_stackplot(
            f'{self.sim.name}__mode_energies_vs_time__stacked',
            self.sim.times,
            *[self.sim.mode_energies(self.sim.mode_amplitudes_vs_time)[:, q] for q in mode_numbers],
            line_labels = [
                fr'${self.sim.spec.modes[q].tex}$'
                for q in mode_numbers
            ],
            line_kwargs = [mode_kwargs(self, q, self.sim.spec.modes[q]) for q in mode_numbers],
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

    def mode_photon_counts_vs_time(
        self,
        time_unit = 'nsec',
        y_log_axis = True,
        y_pad = .05,
        y_log_pad = 10,
        mode_filter = None,
        mode_kwargs = None,
        average_over = None,
        **kwargs,
    ):
        if mode_filter is None:
            mode_filter = lambda sim, q, mode: True

        if mode_kwargs is None:
            mode_kwargs = lambda sim, q, mode: {}

        mode_numbers = [q for mode, q in self.sim.mode_to_index.items() if mode_filter(self, q, mode)]

        x = self.sim.times
        y = [self.sim.mode_energies(self.sim.mode_amplitudes_vs_time)[:, q] / self.sim.mode_photon_energy[q] for q in mode_numbers]

        if average_over is not None:
            l = self.sim.mode_amplitudes_vs_time[:, 0].size
            R = int(average_over / self.sim.spec.time_step)
            pad_size = int((np.ceil(l / R) * R) - l)

            x = np.append(x, np.zeros(pad_size) * np.NaN).reshape((-1, R)).mean(axis = 1)
            y = [np.append(yy, np.zeros(pad_size) * np.NaN).reshape((-1, R)).mean(axis = 1) for yy in y]

        si.vis.xy_plot(
            f'{self.sim.name}__mode_photon_counts_vs_time',
            x,
            *y,
            line_labels = [
                fr'${self.sim.spec.modes[q].tex}$'
                for q in mode_numbers
            ],
            line_kwargs = [mode_kwargs(self, q, self.sim.spec.modes[q]) for q in mode_numbers],
            x_unit = time_unit,
            x_label = r'$t$',
            y_label = r'$U_q(t) \, / \, \hbar \omega_q$',
            y_log_axis = y_log_axis,
            legend_on_right = True,
            y_pad = y_pad,
            y_log_pad = y_log_pad,
            **kwargs,
        )

    def stack_mode_photon_counts_vs_time(
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

        mode_numbers = [q for mode, q in self.sim.mode_to_index.items() if mode_filter(self, q, mode)]

        si.vis.xy_stackplot(
            f'{self.sim.name}__mode_photon_counts_vs_time__stacked',
            self.sim.times,
            *[self.sim.mode_energies(self.sim.mode_amplitudes_vs_time)[:, q] / self.sim.mode_photon_energy[q] for q in mode_numbers],
            line_labels = [
                fr'${self.sim.spec.modes[q].tex}$'
                for q in mode_numbers
            ],
            line_kwargs = [mode_kwargs(self, q, self.sim.spec.modes[q]) for q in mode_numbers],
            x_unit = time_unit,
            x_label = r'$t$',
            y_label = r'$U_q(t) \, / \, \hbar \omega_q$',
            legend_on_right = True,
            y_pad = y_pad,
            y_log_pad = y_log_pad,
            **kwargs,
        )

    def mode_pump_powers_vs_time(
        self,
        time_unit = 'nsec',
        power_unit = 'uW',
        mode_filter = None,
        mode_kwargs = None,
        **kwargs,
    ):
        if mode_filter is None:
            mode_filter = lambda sim, q, mode: True

        if mode_kwargs is None:
            mode_kwargs = lambda sim, q, mode: {}

        mode_numbers = [q for mode, q in self.sim.mode_to_index.items() if mode_filter(self, q, mode)]

        si.vis.xy_plot(
            f'{self.sim.name}__mode_pump_powers_vs_time',
            self.sim.times,
            *[self.sim.spec.mode_pumps[q].get_power(self.sim.times) for q in mode_numbers],
            line_labels = [
                fr'${self.sim.spec.modes[q].tex}$'
                for q in mode_numbers
            ],
            line_kwargs = [mode_kwargs(self, q, self.sim.spec.modes[q]) for q in mode_numbers],
            x_unit = time_unit,
            x_label = r'$t$',
            y_unit = power_unit,
            y_label = r'$P^{\mathrm{in}}_q$',
            # legend_on_right = True,
            **kwargs,
        )
