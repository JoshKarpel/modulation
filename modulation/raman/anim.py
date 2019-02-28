import logging

import numpy as np
import matplotlib.pyplot as plt

import simulacra as si
import simulacra.units as u
from simulacra.vis import anim

logger = logging.getLogger(__name__)


class PolarComplexAmplitudeAxis(anim.AxisManager):
    def __init__(
        self, trail_length=100 * u.nsec, r_log_lower_limit=5, r_log_upper_limit=10
    ):
        super().__init__()

        self.trail_length = trail_length
        self.theta = np.linspace(0, u.twopi, 500)

        self.r_log_lower_limit = r_log_lower_limit
        self.r_log_upper_limit = r_log_upper_limit

    def initialize(self, simulation):
        super().initialize(simulation)

        self.trail_index_offset = int(self.trail_length / simulation.spec.time_step)

    def initialize_axis(self):
        self.axis.set_theta_zero_location("N")
        self.axis.set_theta_direction("clockwise")
        self.axis.set_rlabel_position(30)

        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]
        r, theta = self._get_r_theta(self.sim.mode_amplitudes)
        self.r_theta = np.vstack((theta, r)).T
        self.amplitude_scatter = self.axis.scatter(
            np.zeros_like(theta), np.zeros_like(r), color=colors
        )
        self.redraw.append(self.amplitude_scatter)

        self.lines = []
        for q, (mode, color) in enumerate(zip(self.sim.spec.modes, colors)):
            line = plt.plot(
                np.log10(
                    np.abs(self.sim.mode_amplitudes_vs_time[: self.sim.time_index, q])
                ),
                np.angle(self.sim.mode_amplitudes_vs_time[: self.sim.time_index, q]),
                color=color,
                animated=True,
            )[0]
            self.lines.append(line)
            self.redraw.append(line)

        self.circles = []
        for q, (mode, color) in enumerate(zip(self.sim.spec.modes, colors)):
            circle = plt.plot(
                self.theta,
                np.log10(np.abs(self.sim.mode_amplitudes[q]))
                * np.ones_like(self.theta),
                alpha=0.5,
                color=color,
                animated=True,
            )[0]
            self.circles.append(circle)
            self.redraw.append(circle)

        self.axis.set_ylim(self.r_log_lower_limit, self.r_log_upper_limit)
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
        angles = np.angle(
            self.sim.mode_amplitudes_vs_time[bottom : self.sim.time_index + 1, :]
        )
        rs = np.log10(
            np.abs(
                self.sim.mode_amplitudes_vs_time[bottom : self.sim.time_index + 1, :]
            )
        )

        mask = rs <= np.log10(1.1 * self.sim.mode_background_magnitudes)
        angles = np.ma.masked_where(mask, angles)
        rs = np.ma.masked_where(mask, rs)

        for q in range(len(self.lines)):
            self.lines[q].set_data(angles[:, q], rs[:, q])

            self.circles[q].set_data(
                self.theta,
                np.log10(np.abs(self.sim.mode_amplitudes[q]))
                * np.ones_like(self.theta),
            )

        super().update_axis()

    def _get_r_theta(self, mode_amplitudes):
        r = np.log10(np.abs(mode_amplitudes))
        theta = (np.angle(mode_amplitudes) + u.twopi) % u.twopi

        return r, theta


class SquareAnimator(anim.Animator):
    def __init__(self, axman, time_text_unit: u.Unit = "nsec", **kwargs):
        super().__init__(**kwargs)

        self.axman = axman
        self.axis_managers.append(self.axman)

        self.time_text_unit, self.time_text_unit_tex = u.get_unit_value_and_latex_from_unit(
            time_text_unit
        )

    def _initialize_figure(self):
        self.fig = si.vis.get_figure(fig_width=6, fig_height=6, fig_dpi_scale=3)

        self.ax = self.fig.add_axes([0.05, 0.05, 0.9, 0.9], projection="polar")
        self.axman.assign_axis(self.ax)

        self.time_text = plt.figtext(0.025, 0.025, "", fontsize=16, animated=True)
        self.redraw.append(self.time_text)

        super()._initialize_figure()

    def _update_data(self):
        self.time_text.set_text(
            fr"$t = {self.sim.current_time / self.time_text_unit:.1f} \, {self.time_text_unit_tex}$"
        )

        super()._update_data()
