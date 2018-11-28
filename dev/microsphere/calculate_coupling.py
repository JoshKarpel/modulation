#!/usr/bin/env python
import itertools
import logging
from pathlib import Path

import numpy as np

import simulacra as si
import simulacra.units as u

from modulation import refraction
from modulation.resonator import microsphere

THIS_FILE = Path(__file__)
OUT_DIR = THIS_FILE.parent / 'out' / THIS_FILE.stem
SIM_LIB = OUT_DIR / 'SIMLIB'

LOGMAN = si.utils.LogManager('simulacra', 'modulation', stdout_level = logging.INFO)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 6,
)

ANIM_KWARGS = dict(
    target_dir = OUT_DIR,
)


def quality_factor_vs_separation(index, microsphere_radius, fiber_radius, wavelength):
    separations = np.linspace(0, 1000, 1000) * u.nm

    quality_factors = np.empty_like(separations)
    for idx, sep in enumerate(separations):
        quality_factors[idx] = microsphere.coupling_quality_factor_for_tapered_fiber(
            microsphere_index_of_refraction = index,
            fiber_index_of_refraction = index,
            microsphere_radius = microsphere_radius,
            fiber_taper_radius = fiber_radius,
            wavelength = wavelength,
            separation = sep,
            l = 0,
            m = 0,
        )

    si.vis.xy_plot(
        'quality_factor_vs_separation',
        separations,
        quality_factors,
        x_label = r'Fiber-Sphere Separation $s$',
        x_unit = 'nm',
        y_label = r'$Q_c$',
        y_log_axis = True,
        hlines = [1.5e8],
        **PLOT_KWARGS,
    )


def quality_factor_vs_wavelength(index, microsphere_radius, fiber_radius, separation):
    wavelengths = np.linspace(500, 2000, 1000) * u.nm

    quality_factors = np.empty_like(wavelengths)
    for idx, wavelength in enumerate(wavelengths):
        quality_factors[idx] = microsphere.coupling_quality_factor_for_tapered_fiber(
            microsphere_index_of_refraction = index,
            fiber_index_of_refraction = index,
            microsphere_radius = microsphere_radius,
            fiber_taper_radius = fiber_radius,
            wavelength = wavelength,
            separation = separation,
            l = 0,
            m = 0,
        )

    si.vis.xy_plot(
        'quality_factor_vs_wavelength',
        wavelengths,
        quality_factors,
        x_label = r'Wavelength $\lambda$',
        x_unit = 'nm',
        y_label = r'$Q_c$',
        y_log_axis = True,
        hlines = [1.5e8],
        vlines = [980 * u.nm],
        **PLOT_KWARGS,
    )


if __name__ == '__main__':
    with LOGMAN as logger:
        index = refraction.ConstantIndex(1.45)
        microsphere_radius = 50 * u.um
        fiber_radius = 1 * u.um

        quality_factor_vs_separation(index, microsphere_radius, fiber_radius, 980 * u.nm)
        quality_factor_vs_wavelength(index, microsphere_radius, fiber_radius, 300 * u.nm)
