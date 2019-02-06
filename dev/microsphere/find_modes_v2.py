#!/usr/bin/env python
import itertools
import logging
from pathlib import Path

import numpy as np
import scipy.special as spc
import scipy.optimize as opt

import simulacra as si
import simulacra.units as u

from modulation import refraction
from modulation.resonators import microspheres

import modulation.resonators.microspheres.find as find

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


def plot_modal_equation(wavelengths, l, microsphere, polarization):
    me = find.modal_equation(
        wavelengths,
        l,
        polarization,
        microsphere,
    )

    y_bound = 2 * np.nanmean(np.abs(me))

    # arg_zeros = find_zeros(order = l, which = spc.ai_zeros)
    # wavelength_over_index = (u.twopi * microspheres.radius) / arg_zeros
    #
    # wavelength_with_index_zeros = opt.root(
    #     lambda wavelength: (wavelength / microspheres.index_of_refraction(wavelength)) - wavelength_over_index,
    #     1000 * u.nm * np.ones_like(wavelength_over_index),
    # ).x
    #
    # other_arg_zeros = find_zeros(order = l, which = spc.bi_zeros)
    # wavelength_without_index_zeros = microspheres.radius * u.twopi / other_arg_zeros
    #
    # all_zeros = np.concatenate((wavelength_with_index_zeros, wavelength_without_index_zeros))
    # vline_kwargs = [{'alpha': 0.9, 'linewidth': 1, 'color': 'blue'} for _ in range(len(wavelength_with_index_zeros))]
    # vline_kwargs.extend([{'alpha': 0.9, 'linewidth': 1, 'color': 'red'} for _ in range(len(wavelength_without_index_zeros))])
    #
    # roots = find_roots(
    #     l,
    #     polarization,
    #     microspheres,
    #     wavelength_with_index_zeros,
    #     wavelength_without_index_zeros,
    # )
    modes = find.find_mode_locations(
        [find.WavelengthBound(min(wavelengths), max(wavelengths))],
        microsphere,
    )
    roots = [mode.wavelength for mode in modes if mode.l == l]
    root_kwargs = [{'alpha': 0.9, 'linewidth': 1, 'color': 'magenta'} for _ in range(len(roots))]

    si.vis.xy_plot(
        f'modal_equation__l={l}',
        wavelengths,
        me,
        find.modal_equation(
            wavelengths,
            l,
            microspheres.MicrosphereModePolarization.TRANSVERSE_MAGNETIC,
            microsphere,
        ),
        line_kwargs = [{'linewidth': 1, 'color': 'black'}, {'linewidth': 1, 'color': 'black', 'linestyle': '--'}],
        title = rf'Modal Equation for $\ell = {l}$',
        x_label = r'Wavelength $\lambda$',
        y_label = 'Modal Equation',
        x_unit = 'nm',
        y_lower_limit = -y_bound,
        y_upper_limit = y_bound,
        vlines = roots,
        vline_kwargs = root_kwargs,
        **PLOT_KWARGS,
    )


NUM_ZEROS = 500


def func(z, xi):
    return (2 / 3) * ((-xi) ** (3 / 2)) - np.sqrt((z ** 2) - 1) + np.arccos(1 / z)


def jac(z, xi):
    return np.diag(-(z / np.sqrt((z ** 2) - 1)) + 1 / (z ** 2 * np.sqrt(1 - (1 / z ** 2))))


def find_zeros(order = 500, which = spc.ai_zeros):
    """DLMF 10.21"""
    order += 0.5
    ai_zeros, *_ = which(NUM_ZEROS)

    xi = (order ** (-2 / 3)) * ai_zeros

    z = opt.root(
        func,
        1.1 * np.ones_like(xi),
        args = (xi,),
        jac = jac,
    ).x

    h = (4 * xi / (1 - (z ** 2))) ** (1 / 4)

    p = 1j * ((z ** 2) - 1) ** (-1 / 2)
    U = (1 / 24) * ((3 * p) - 5 * (p ** 3))
    B_0 = np.real(1j * ((-xi) ** (-1 / 2)) * U)

    zeros = (order * z) + ((z * (h ** 2) * B_0) / (2 * order))
    return zeros


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def find_roots(
    l,
    polarization,
    microsphere,
    wavelength_with_index_zeros,
    wavelength_without_index_zeros,
):
    args = (l, polarization, microsphere)
    wavelength_with_index_zeros = np.array(sorted(wavelength_with_index_zeros, key = lambda x: -x))

    zeros = []

    # todo: very first zero
    print(f'searching near {wavelength_with_index_zeros[0] / u.nm:.6f} for first root')
    first_zero = opt.brentq(
        find.modal_equation,
        wavelength_with_index_zeros[0],
        wavelength_with_index_zeros[0] * 1.5,
        args = args,
    )
    print(f'found first zero at {first_zero / u.nm:.6f}')
    zeros.append(first_zero)

    for (top, bottom) in pairwise(wavelength_with_index_zeros):
        print(f'outer bounds {top / u.nm:.6f} to {bottom / u.nm:.6f}')
        if any(bottom < w < top for w in wavelength_without_index_zeros):
            print(f'skipping\n')
            continue

        center = (top + bottom) / 2
        upper = (center + top) / 2
        lower = (center + bottom) / 2

        while not (find.modal_equation(upper, *args) < 0):
            upper = (upper + top) / 2
        while not (find.modal_equation(lower, *args) > 0):
            lower = (lower + bottom) / 2

        print(f'search from {upper / u.nm:.6f} to {lower / u.nm:.6f}')

        zero = opt.brentq(
            find.modal_equation,
            lower,
            upper,
            args = args,
        )
        zeros.append(zero)
        print(f'found zero at {zero / u.nm:.6f}')

        print()

    return zeros


if __name__ == '__main__':
    ms = microspheres.Microsphere(
        radius = 50 * u.um,
        # index_of_refraction = refraction.ConstantIndex(1.45),
        index_of_refraction = refraction.SellmeierIndex.from_name('SiO2'),
    )

    # wavelengths = np.linspace(950, 1000, 100000) * u.nm
    #
    # for l in [450]:
    #     plot_modal_equation(
    #         wavelengths,
    #         l,
    #         ms,
    #         microspheres.MicrosphereModePolarization.TRANSVERSE_ELECTRIC,
    #     )

    # find_zeros(450)

    modes = find.find_mode_locations(
        [
            find.WavelengthBound(950 * u.nm, 1000 * u.nm)
        ],
        ms,
    )
    print(len(modes))
