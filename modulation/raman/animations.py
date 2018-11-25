import numpy as np
import matplotlib.pyplot as plt

import simulacra as si
import simulacra.units as u
from simulacra.vis import anim


class PolarComplexAmplitudeAxis(anim.AxisManager):
    def __init__(self, *args, trail_length = 100 * u.nsec, **kwargs):
        super().__init__(*args, **kwargs)

        self.trail_length = trail_length

    def initialize(self, simulation):
        super().initialize(simulation)

        self.trail_index_offset = int(self.trail_length / simulation.spec.time_step)

    def initialize_axis(self):
        self.axis.set_theta_zero_location('N')
        self.axis.set_theta_direction('clockwise')
        self.axis.set_rlabel_position(30)

        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        r, theta = self._get_r_theta(self.sim.mode_amplitudes)
        self.r_theta = np.vstack((theta, r)).T
        self.amplitude_scatter = self.axis.scatter(
            np.zeros_like(theta),
            np.zeros_like(r),
            color = colors,
        )
        self.redraw.append(self.amplitude_scatter)

        self.lines = []
        for q, mode in enumerate(self.sim.spec.modes):
            line = plt.plot(
                np.log10(np.abs(self.sim.mode_amplitudes_vs_time[:self.sim.time_index, q])),
                np.angle(self.sim.mode_amplitudes_vs_time[:self.sim.time_index, q]),
                animated = True,
            )[0]
            self.lines.append(line)
            self.redraw.append(line)

        self.axis.set_ylim(5, 10)
        self.axis.grid(True)

        super().initialize_axis()

    def update_axis(self):
        r, theta = self._get_r_theta(self.sim.mode_amplitudes)
        self.r_theta[:, 0] = theta
        self.r_theta[:, 1] = r
        self.amplitude_scatter.set_offsets(self.r_theta)

        bottom = self.sim.time_index - self.trail_index_offset
        if bottom < 0:
            bottom = 0
        angles = np.angle(self.sim.mode_amplitudes_vs_time[bottom:self.sim.time_index + 1, :])
        rs = np.log10(np.abs(self.sim.mode_amplitudes_vs_time[bottom:self.sim.time_index + 1, :]))

        for q in range(len(self.lines)):
            self.lines[q].set_data(
                angles[:, q],
                rs[:, q],
            )

        super().update_axis()

    def _get_r_theta(self, mode_amplitudes):
        r = np.log10(np.abs(mode_amplitudes))
        theta = (np.angle(mode_amplitudes) + u.twopi) % u.twopi

        return r, theta


class PolarComplexAmplitudeAnimator(anim.Animator):
    def __init__(
        self,
        axman_amplitude,
        time_text_unit: u.Unit = 'nsec',
        **kwargs
    ):
        super().__init__(**kwargs)

        self.axman_amplitude = axman_amplitude
        self.axis_managers.append(self.axman_amplitude)

        self.time_text_unit, self.time_text_unit_tex = u.get_unit_value_and_latex_from_unit(time_text_unit)

    def _initialize_figure(self):
        self.fig = si.vis.get_figure(
            fig_width = 6,
            fig_height = 6,
            fig_dpi_scale = 3,
        )

        self.ax_amplitude = self.fig.add_axes(
            [.05, .05, .9, .9],
            projection = 'polar',
        )
        self.axman_amplitude.assign_axis(self.ax_amplitude)

        self.time_text = plt.figtext(
            .025, .025,
            '',
            fontsize = 16,
            animated = True,
        )
        self.redraw.append(self.time_text)

        super()._initialize_figure()

    def _update_data(self):
        self.time_text.set_text(
            fr'$t = {self.sim.time / self.time_text_unit:.1f} \, {self.time_text_unit_tex}$'
        )

        super()._update_data()
