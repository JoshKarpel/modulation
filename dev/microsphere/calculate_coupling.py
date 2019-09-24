#!/usr/bin/env python
import itertools
import logging
from pathlib import Path

import numpy as np

import simulacra as si
import simulacra.units as u

from modulation import refraction
from modulation.resonators import microspheres

THIS_FILE = Path(__file__)
OUT_DIR = THIS_FILE.parent / "out" / THIS_FILE.stem
SIM_LIB = OUT_DIR / "SIMLIB"

LOGMAN = si.utils.LogManager("simulacra", "modulation", stdout_level=logging.INFO)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=6)

ANIM_KWARGS = dict(target_dir=OUT_DIR)


def quality_factor_vs_separation(index, microsphere_radius, fiber_radii, wavelengths):
    separations = np.linspace(0, 500, 1000) * u.nm

    quality_factors = []
    labels = []
    for i, (fiber_radius, wavelength) in enumerate(
        itertools.product(fiber_radii, wavelengths)
    ):
        q = microspheres.coupling_quality_factor_for_tapered_fiber(
            microsphere_index_of_refraction=index,
            fiber_index_of_refraction=index,
            microsphere_radius=microsphere_radius,
            fiber_taper_radius=fiber_radius,
            wavelength=wavelength,
            separation=separations,
            l=0,
            m=0,
        )

        quality_factors.append(q)
        labels.append(
            fr'$R_{{\mathrm{{fiber}}}} = {fiber_radius / u.um:.1f} \, \mathrm{{\mu m}}, \, \lambda = {int(wavelength / u.nm)} \, \mathrm{{nm}}$'
        )

    si.vis.xy_plot(
        "quality_factor_vs_separation",
        separations,
        *quality_factors,
        line_labels=labels,
        x_label=r"Fiber-Sphere Separation $s$",
        x_unit="nm",
        y_label=r"$Q_c$",
        y_log_axis=True,
        hlines=[1e8],
        font_size_legend=10,
        legend_on_right=True,
        **PLOT_KWARGS,
    )


def quality_factor_vs_wavelength(index, microsphere_radius, fiber_radii, separations):
    wavelengths = np.linspace(700, 1200, 1000) * u.nm

    quality_factors = []
    labels = []
    for i, (fiber_radius, separation) in enumerate(
        itertools.product(fiber_radii, separations)
    ):
        q = microspheres.coupling_quality_factor_for_tapered_fiber(
            microsphere_index_of_refraction=index,
            fiber_index_of_refraction=index,
            microsphere_radius=microsphere_radius,
            fiber_taper_radius=fiber_radius,
            wavelength=wavelengths,
            separation=separation,
            l=0,
            m=0,
        )

        quality_factors.append(q)
        labels.append(fr'$s = {separation / u.um:.3f} \, \mathrm{{\mu m}}$')

    si.vis.xy_plot(
        "quality_factor_vs_wavelength",
        wavelengths,
        *quality_factors,
        line_labels=labels,
        x_label=r"Wavelength $\lambda$",
        x_unit="nm",
        y_label=r"$Q_c$",
        y_log_axis=True,
        hlines=[1e8],
        vlines=np.array([1120, 1064, 800, 771]) * u.nm,
        **PLOT_KWARGS,
    )


if __name__ == "__main__":
    with LOGMAN as logger:
        index = refraction.SellmeierIndex.from_name("silica")
        microsphere_radius = 50 * u.um

        fiber_radii = [1 * u.um]
        # wavelengths = (
        #     np.array([1180, 1120, 1064, 1010, 968, 865, 831, 800, 771, 744]) * u.nm
        # )
        separations = [341 * u.nm, 175 * u.nm]

        # quality_factor_vs_separation(
        #     index, microsphere_radius, fiber_radii=fiber_radii, wavelengths=wavelengths
        # )
        quality_factor_vs_wavelength(
            index, microsphere_radius, fiber_radii=fiber_radii, separations=separations
        )
