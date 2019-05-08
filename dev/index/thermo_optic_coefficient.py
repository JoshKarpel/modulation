#!/usr/bin/env python
import itertools
import logging
from pathlib import Path
from copy import deepcopy

from tqdm import tqdm

import numpy as np

import simulacra as si
import simulacra.units as u

from modulation import raman, refraction
from modulation.resonators import microspheres

THIS_FILE = Path(__file__)
OUT_DIR = THIS_FILE.parent / "out" / THIS_FILE.stem
SIM_LIB = OUT_DIR / "SIMLIB"

LOGMAN = si.utils.LogManager("simulacra", "modulation", stdout_level=logging.INFO)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=6)


def plot_index_vs_temperature(index, wavelengths, temperatures):
    si.vis.xy_plot(
        f"{index.name}__index_vs_temperature",
        temperatures,
        *[index(wavelength, temperatures) for wavelength in wavelengths],
        x_unit="K",
        x_label="Temperature $T$",
        y_label=r"$n(T)$",
        **PLOT_KWARGS,
    )


def plot_mode_frequencies_vs_temperature(spheres, wavelength_bounds):
    sphere_to_modes = {}
    for sphere in tqdm(spheres):
        sphere_to_modes[sphere] = microspheres.find_mode_locations(
            wavelength_bounds, sphere
        )

    print(sphere_to_modes)

    frequencies = [[] for _ in range(len(list(sphere_to_modes.values())[0]))]

    for k, v in sphere_to_modes.items():
        print(k)
        for vv, line in zip(v, frequencies):
            print(f"  {vv.wavelength / u.nm:.6f}")
            line.append(u.c / vv.wavelength)

    x = np.array([sphere.temperature - spheres[0].temperature for sphere in spheres])
    diffs = [np.array(line) - line[0] for line in frequencies]
    frac_diffs = [d / l for d, l in zip(diffs, frequencies)]

    si.vis.xy_plot(
        "frequency_diffs",
        x,
        *diffs,
        x_unit="K",
        x_label=r"Temperature Difference $T$",
        y_unit="GHz",
        y_label=r"Mode Frequency Difference $\Delta f$",
        **PLOT_KWARGS,
    )
    si.vis.xy_plot(
        "frequency_diffs_fractional",
        x,
        *frac_diffs,
        x_unit="K",
        x_label=r"Temperature Difference $T$",
        **PLOT_KWARGS,
    )


if __name__ == "__main__":
    index = refraction.SellmeierIndex.from_name("silica")
    print(index.info())

    plot_index_vs_temperature(
        index,
        np.array([632, 800, 1064]) * u.nm,
        u.zero_celsius + np.linspace(-50, 100) * u.K,
    )

    temps = u.zero_celsius + (20 * u.K) + (np.linspace(0, 10, 10) * u.K)

    spheres = [
        microspheres.Microsphere(
            radius=50 * u.um, index_of_refraction=index, temperature=temp
        )
        for temp in temps
    ]

    wavelength_bounds = [
        microspheres.WavelengthBound(lower=1063.5 * u.nm, upper=1064.5 * u.nm),
        microspheres.WavelengthBound(lower=799.5 * u.nm, upper=800.5 * u.nm),
    ]

    plot_mode_frequencies_vs_temperature(spheres, wavelength_bounds)
