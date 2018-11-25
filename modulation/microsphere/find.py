from typing import NamedTuple, Collection, Tuple, Optional, List, Callable, Iterable

import functools
import itertools

import numpy as np
import scipy.special as spc
import scipy.optimize as opt

import simulacra as si
import simulacra.units as u

from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from ..refraction import IndexOfRefraction

from . import mode


def _group_by_nans(x):
    return itertools.groupby(x, key = lambda x_element: np.isnan(x_element))


def _calculate_breaks(y, x) -> Collection[Tuple[float, float]]:
    breaks = []
    for is_nan, group in _group_by_nans(_clip(y, x)):
        group = list(group)
        if not is_nan:
            breaks.append((group[0], group[-1]))

    return breaks


def _find_roots(eqn, breaks) -> List[float]:
    """Find one root inside each region defined by the breakpoints."""
    roots = []
    for a, b in breaks:
        try:
            roots.append(opt.brentq(eqn, a, b))
        except:
            pass

    return roots


def _clip(y: np.array, x: Optional[np.array] = None) -> np.array:
    if x is None:
        x = y

    return np.where(
        np.less_equal(np.abs(y), np.mean(np.abs(y))),
        x,
        np.NaN,
    )


class MicrosphereModeLocation(NamedTuple):
    wavelength: float
    l: int
    radial_mode_number: int
    polarization: mode.MicrosphereModePolarization

    @property
    def frequency(self):
        return u.c / self.wavelength

    def __repr__(self):
        return f'{self.__class__.__name__}(λ={u.uround(self.wavelength, u.nm, 6)} nm, ℓ={self.l}, n={self.radial_mode_number}, polarization = {self.polarization})'


def shift_wavelength_by_frequency(wavelength: float, frequency_shift: float) -> float:
    frequency = u.c / wavelength
    shifted_frequency = frequency + frequency_shift

    return u.c / shifted_frequency


class WavelengthBound(NamedTuple):
    min: float
    max: float

    def __contains__(self, item):
        return self.min <= item <= self.max

    def __str__(self):
        return f'{self.__class__.__name__}(min = {u.uround(self.min, u.nm, 6)} nm, max = {u.uround(self.max, u.nm, 6)} nm)'


class LBound:
    __slots__ = ('min', 'max')

    def __init__(self, min: int, max: int):
        self.min = int(min)
        self.max = int(max)

    def __contains__(self, item):
        return self.min <= item <= self.max

    def __repr__(self):
        return f'{self.__class__.__name__}(min = {self.min}, max = {self.max})'

    def __iter__(self) -> Iterable[int]:
        yield from range(self.min, self.max + 1)


def TE_P(wavelength, index):
    return index(wavelength)


def TM_P(wavelength, index):
    return 1 / index(wavelength)


class ModeFinder(si.Simulation):
    def __init__(self, spec):
        super().__init__(spec)

        self.mode_locations = []
        self.mode_locations_by_l = {}

    def alpha(self, wavelength, polarization):
        return self.P(wavelength, polarization) / (self.spec.n(wavelength) * np.sqrt((self.spec.n(wavelength) ** 2) - 1))

    def P(self, wavelength, polarization):
        return {
            mode.MicrosphereModePolarization.TRANSVERSE_ELECTRIC: functools.partial(TE_P, index = self.spec.n),
            mode.MicrosphereModePolarization.TRANSVERSE_MAGNETIC: functools.partial(TM_P, index = self.spec.n),
        }[polarization](wavelength)

    def run(self):
        self.status = si.Status.RUNNING

        self.mode_locations = []

        for polarization in mode.MicrosphereModePolarization:
            for wavelength_bound in self.spec.wavelength_bounds:
                l_bound = self.l_bound_from_wavelength_bound(wavelength_bound, polarization)

                for l in l_bound:
                    min_wavelength = opt.brentq(
                        lambda x: x - (u.twopi * self.spec.radius / (l + 0.5 - self.alpha(x, polarization))),
                        100 * u.nm,
                        2 * u.um,
                    )
                    max_wavelength = opt.brentq(
                        lambda x: x - (u.twopi * self.spec.radius * self.spec.n(x) / (l + 0.5 - (self.alpha(x, polarization) * self.spec.n(x)))),
                        100 * u.nm,
                        2 * u.um,
                    )

                    wavelengths = np.linspace(min_wavelength, max_wavelength, 1_000)[::-1]  # largest wavelengths first

                    breaks = _calculate_breaks(self.modal_equation(wavelengths, l, polarization), wavelengths)
                    roots = _find_roots(
                        functools.partial(self.modal_equation, l = l, polarization = polarization),
                        breaks,
                    )

                    mode_locations = [
                        MicrosphereModeLocation(
                            wavelength = root,
                            l = l,
                            radial_mode_number = n,
                            polarization = polarization,
                        )
                        for n, root in enumerate(roots, start = 1)
                        if root in wavelength_bound
                    ]

                    if len(mode_locations) != 0:
                        self.mode_locations.extend(mode_locations)

        self.mode_locations = sorted(self.mode_locations, key = lambda x: x.wavelength)

        self.status = si.Status.FINISHED

    def modal_equation(
        self,
        wavelength,
        l: int,
        polarization: mode.MicrosphereModePolarization,
    ):
        """

        eq 3.70 in Balac2016

        Parameters
        ----------
        wavelength
        l
        polarization

        Returns
        -------

        """
        lp = l + 0.5
        lm = l - 0.5

        k_0_R = self.spec.radius * u.twopi / wavelength
        k_R = self.spec.n(wavelength) * k_0_R

        y = spc.yv(lm, k_0_R) / spc.yv(lp, k_0_R)
        j = self.P(wavelength, polarization) * spc.jv(lm, k_R) / spc.jv(lp, k_R)

        result = y - j
        result += -l * ((1 / k_0_R) - (self.P(wavelength, polarization) / k_R))

        return result

    def _l_bounds_from_wavelength(
        self,
        wavelength: float,
        polarization: mode.MicrosphereModePolarization,
    ) -> LBound:
        n = self.spec.n(wavelength)
        p = self.P(wavelength, polarization)
        delta_P = wavelength * p / (u.twopi * n * np.sqrt((n ** 2) - 1))
        min = u.twopi * (self.spec.radius + delta_P) / wavelength
        max = n * min

        min -= 0.5
        max -= 0.5

        return LBound(np.floor(min), np.ceil(max))

    def l_bound_from_wavelength_bound(
        self,
        wavelength_bound: WavelengthBound,
        polarization: mode.MicrosphereModePolarization,
    ) -> LBound:
        bound_1 = self._l_bounds_from_wavelength(wavelength_bound.min, polarization)
        bound_2 = self._l_bounds_from_wavelength(wavelength_bound.max, polarization)

        return LBound(
            min = min(bound_1.min, bound_2.min),
            max = max(bound_1.max, bound_2.max),
        )

    def plot_mode_locations_by_wavelength(
        self,
        wavelength_unit = 'nm',
        wavelength_lower_bound: Optional[float] = None,
        wavelength_upper_bound: Optional[float] = None,
        mode_filter: Optional[Callable] = None,
        show_regions = False,
        relative = False,
        frequency = False,
        wavelength_center = 1064 * u.nm,
        **kwargs,
    ) -> si.vis.FigureManager:
        if mode_filter is None:
            mode_filter = lambda mode: True

        wavelength_unit_value, wavelength_unit_tex = u.get_unit_value_and_latex_from_unit(wavelength_unit)

        wavelength_lower_bound = wavelength_lower_bound or min(b.min for b in self.spec.wavelength_bounds)
        wavelength_upper_bound = wavelength_upper_bound or max(b.max for b in self.spec.wavelength_bounds)

        figman = si.vis.FigureManager(
            f'{self.file_name}__modes',
            fig_width = 10,
            fig_height = 2,
            **kwargs,
        )

        with figman:
            fig = figman.fig
            ax = fig.add_subplot(111)

            ax.set_ylim(0, 1)
            h = 0.35

            if show_regions:
                for bound in self.spec.wavelength_bounds:
                    ax.add_patch(
                        Rectangle(
                            (bound.min / wavelength_unit_value, 0),
                            (bound.max - bound.min) / wavelength_unit_value,
                            h * .3,
                            alpha = .3,
                            facecolor = 'gray',
                            edgecolor = 'none',
                        )
                    )

            colors = {
                mode.MicrosphereModePolarization.TRANSVERSE_ELECTRIC: '#1b9e77',
                mode.MicrosphereModePolarization.TRANSVERSE_MAGNETIC: '#d95f02',
            }

            for mode in filter(mode_filter, self.mode_locations):
                ax.axvline(
                    (mode.wavelength - (wavelength_center if relative else 0)) / wavelength_unit_value,
                    ymax = h,
                    color = colors[mode.polarization],
                    linewidth = 0.75,
                )
                ax.text(
                    (mode.wavelength - (wavelength_center if relative else 0)) / wavelength_unit_value,
                    h + 0.015,
                    s = fr'$ \lambda = {u.uround(mode.wavelength, wavelength_unit, 3)} \, \mathrm{{nm}}, \, \ell = {mode.l}, \, n = {mode.radial_mode_number}$',
                    fontsize = 5,
                    rotation = 60,
                    rotation_mode = 'anchor',
                )

            ax.set_xlim(
                (wavelength_lower_bound - (wavelength_center if relative else 0)) / wavelength_unit_value,
                (wavelength_upper_bound - (wavelength_center if relative else 0)) / wavelength_unit_value,
            )

            ax.axes.get_yaxis().set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)

            if relative:
                ax.set_xlabel(rf'Mode Wavelength $ \lambda - \lambda_0 \, (\mathrm{{{wavelength_unit_tex}}}) $')
            else:
                ax.set_xlabel(rf'Mode Wavelength $ \lambda \, (\mathrm{{{wavelength_unit_tex}}}) $')

            if relative:
                ax.set_title(rf'$ R = {u.uround(self.spec.radius, u.um)} \, \mathrm{{\mu m}} $, {self.spec.n.tex}, $ \lambda_0 = {u.uround(wavelength_center, wavelength_unit, digits = 0)} \, \mathrm{{{wavelength_unit_tex}}} $')
            else:
                ax.set_title(rf'$ R = {u.uround(self.spec.radius, u.um)} \, \mathrm{{\mu m}} $, {self.spec.n.tex}')

            if frequency:
                ax_freq = ax.twiny()

                freq_lower_bound = u.c / wavelength_upper_bound
                freq_upper_bound = u.c / wavelength_lower_bound
                if relative:
                    freq_lower_bound -= u.c / wavelength_center
                    freq_upper_bound -= u.c / wavelength_center
                freq_lower_bound /= u.GHz
                freq_upper_bound /= u.GHz
                ax_freq.set_xlim(freq_upper_bound, freq_lower_bound)

                if relative:
                    ax_freq.set_xlabel(rf'Mode Frequency $ f - f_0 \, (\mathrm{{GHz}}) $')
                else:
                    ax_freq.set_xlabel(rf'Mode Frequency $ f \, (\mathrm{{GHz}}) $')

                ax_freq.xaxis.set_ticks_position('bottom')
                ax_freq.xaxis.set_label_position('bottom')

                ax_freq.axes.get_yaxis().set_visible(False)
                ax_freq.spines['top'].set_visible(False)
                ax_freq.spines['right'].set_visible(False)
                ax_freq.spines['left'].set_visible(False)
                ax_freq.spines['bottom'].set_position(('outward', 36))

            legend_handles = [
                Line2D([0], [0], color = colors[mode.ModePolarization.TRANSVERSE_ELECTRIC], lw = 4),
                Line2D([0], [0], color = colors[mode.ModePolarization.TRANSVERSE_MAGNETIC], lw = 4),
            ]
            ax.legend(
                legend_handles,
                ['TE', 'TM'],
                bbox_to_anchor = (0.75, -.45) if frequency else (0.75, -.15),
            )

        return figman


class ModeFinderSpecification(si.Specification):
    simulation_type = ModeFinder

    def __init__(
        self,
        name,
        *,
        radius: float,
        n: IndexOfRefraction,
        wavelength_bounds: Collection[WavelengthBound],
        **kwargs,
    ):
        super().__init__(name, **kwargs)

        self.radius = radius
        self.n = n

        self.wavelength_bounds = wavelength_bounds

    def info(self) -> si.Info:
        info = super().info()

        sphere_info = si.Info(header = 'Microsphere')
        sphere_info.add_field('Index of Refraction', self.n)
        sphere_info.add_field('Sphere Radius', f'{u.uround(self.radius, u.um)} µm')
        info.add_info(sphere_info)

        info.add_field('Wavelength Bounds', ', '.join(str(b) for b in self.wavelength_bounds))

        return info
