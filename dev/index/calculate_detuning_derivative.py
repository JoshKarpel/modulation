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


def term(wavelength, index, temperature=None):
    omega = u.twopi * u.c / wavelength

    return omega / (index(wavelength, temperature=temperature))


if __name__ == "__main__":
    index = refraction.SellmeierIndex.from_name("silica")
    print("alpha_n", index.thermo_optic_coefficient)

    freq_m = 12 * u.THz

    temperature = (20 + 273.15) * u.K

    wavelength_P = 1064 * u.nm
    wavelength_M = 800 * u.nm

    wavelength_S = u.c / ((u.c / wavelength_P) - freq_m)
    wavelength_T = u.c / ((u.c / wavelength_M) + freq_m)

    print(f"wavelength M: {wavelength_M / u.nm:.3f} nm")
    print(f"wavelength S: {wavelength_S / u.nm:.3f} nm")
    print(f"wavelength P: {wavelength_P / u.nm:.3f} nm")
    print(f"wavelength T: {wavelength_T / u.nm:.3f} nm")

    term_M = term(wavelength_M, index, temperature=temperature)
    term_S = term(wavelength_S, index, temperature=temperature)
    term_P = term(wavelength_P, index, temperature=temperature)
    term_T = term(wavelength_T, index, temperature=temperature)

    print(f"term M {term_M / u.THz:.3f} THz")
    print(f"term S {term_S / u.THz:.3f} THz")
    print(f"term P {term_P / u.THz:.3f} THz")
    print(f"term T {term_T / u.THz:.3f} THz")

    term_total = term_M - term_S + term_P - term_T
    print(f"term total {term_total / u.GHz:.3f} GHz")

    deriv = index.thermo_optic_coefficient * term_total
    print(f"deriv {deriv / u.MHz:.3f} MHz/deg C")
