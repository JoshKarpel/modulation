#!/usr/bin/env python

import itertools
import logging
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use('pgf')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

import simulacra as si
import simulacra.units as u

from modulation import raman
from modulation.resonator import mock

config = {  # setup matplotlib to use latex for output
    "text.usetex": True,  # use LaTeX to write all text
    'font.family': 'serif',
    'font.serif': 'Computer Modern',
    "axes.labelsize": 11,  # LaTeX default is 10pt font.
    "font.size": 10,
    "legend.fontsize": 11,  # Make the legend/label fonts a little smaller
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}
matplotlib.rcParams.update(config)

THIS_FILE = Path(__file__)
OUT_DIR = THIS_FILE.parent / 'out' / THIS_FILE.stem
SIM_LIB = OUT_DIR / 'SIMLIB'
TEX_DIR = THIS_FILE.parent.parent

LOGMAN = si.utils.LogManager('simulacra', 'modulation', stdout_level = logging.INFO)

ANIM_KWARGS = dict(
    target_dir = OUT_DIR,
)

TEXTWIDTH = si.vis.points_to_inches(470)

FIG_KWARGS = [
    dict(  # test png
        img_format = 'png',
        fig_dpi_scale = 6,
        target_dir = OUT_DIR,
    ),
    dict(  # test pdf
        img_format = 'pdf',
        fig_dpi_scale = 6,
        target_dir = OUT_DIR,
    ),
]


def find_pump_mode(modes, pump_omega: float):
    return sorted(modes, key = lambda m: abs(m.omega - pump_omega))[0]


def make_plot(sim):
    for fig_kwargs in FIG_KWARGS:
        with si.vis.FigureManager(sim.name, fig_width = TEXTWIDTH, **fig_kwargs) as figman:
            fig = figman.fig
            ax = fig.add_axes([0, 0, 1, 1], projection = 'polar')

            mode_numbers = list(range(len(sim.spec.modes)))

            radii = np.log10(np.abs(sim.mode_amplitudes_vs_time))
            angles = np.angle(sim.mode_amplitudes_vs_time)

            for q in mode_numbers[:-1]:
                ax.plot(angles[-1, q], radii[-1, q], color = 'black', marker = 'o', linestyle = '', )
                ax.plot(angles[:, q], radii[:, q], color = 'black', )

            ax.set_ylim(8, 10)
            ax.set_rorigin(7.5)
            ax.set_theta_zero_location('N')
            ax.set_rlabel_position(20)
            ax.set_theta_direction(-1)

            ax.grid(False)
            ax.set_rgrids([], labels = [])
            ax.set_thetagrids([], labels = [])


if __name__ == '__main__':
    with LOGMAN as logger:
        pump_wavelength = 800 * u.nm
        pump_power = 500 * u.uW

        ###

        material = raman.material.RamanMaterial.silica()
        pump_omega = u.twopi * u.c / pump_wavelength

        modes = [
            mock.MockMode(
                label = f'q = {q}',
                omega = pump_omega - (q * material.modulation_omega),
                index_of_refraction = 1.45,
                mode_volume_inside_resonator = 1e-20,
            )
            for q in range(10)
        ]
        print('Modes:')
        for mode in modes:
            print(mode)
        print()

        pump_mode = find_pump_mode(modes, pump_omega)
        pumps = {pump_mode: raman.pump.ConstantPump(pump_power)}
        print(f'Pump: {pump_mode}')

        for x in range(100):
            spec = raman.StimulatedRamanScatteringSpecification(
                name = f'art-{x}',
                material = material,
                mode_volume_integrator = mock.MockVolumeIntegrator(
                    volume_integral_result = 1e-25,
                ),
                modes = modes,
                mode_initial_amplitudes = dict(zip(modes, itertools.repeat(0))),
                mode_intrinsic_quality_factors = dict(zip(modes, itertools.repeat(1e8))),
                mode_coupling_quality_factors = dict(zip(modes, itertools.repeat(1e8))),
                mode_pumps = pumps,
                time_initial = 0,
                time_final = 1000 * u.nsec,
                time_step = .5 * u.nsec,
                store_mode_amplitudes_vs_time = True,
            )

            sim = spec.to_sim()

            sim.run(show_progress_bar = True)

            make_plot(sim)
