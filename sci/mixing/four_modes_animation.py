#!/usr/bin/env python
import itertools
import logging
from pathlib import Path

import numpy as np

import simulacra as si
import simulacra.units as u

from modulation import raman
from modulation.resonators import mock

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


def find_mode(modes, omega: float):
    return sorted(modes, key = lambda m: abs(m.omega - omega))[0]


if __name__ == '__main__':
    with LOGMAN as logger:
        pump_wavelength = 1064 * u.nm
        mixing_wavelength = 632 * u.nm
        pump_power = 100 * u.uW
        mixing_power = 1 * u.uW

        ###

        material = raman.material.RamanMaterial.from_database('silica')
        pump_omega = u.twopi * u.c / pump_wavelength
        mixing_omega = u.twopi * u.c / mixing_wavelength

        modes = [
            mock.MockMode(
                label = f'pump',
                omega = pump_omega,
                index_of_refraction = 1.45,
                mode_volume_inside_resonator = 1e-20,
            ),
            mock.MockMode(
                label = f'stokes',
                omega = pump_omega - material.modulation_omega,
                index_of_refraction = 1.45,
                mode_volume_inside_resonator = 1e-20,
            ),
            mock.MockMode(
                label = f'mixing',
                omega = mixing_omega,
                index_of_refraction = 1.45,
                mode_volume_inside_resonator = 1e-20,
            ),
            mock.MockMode(
                label = f'mixing+',
                omega = mixing_omega + material.modulation_omega,
                index_of_refraction = 1.45,
                mode_volume_inside_resonator = 1e-20,
            ),
            mock.MockMode(
                label = f'mixing-',
                omega = mixing_omega - material.modulation_omega,
                index_of_refraction = 1.45,
                mode_volume_inside_resonator = 1e-20,
            )
        ]
        print('Modes:')
        for mode in modes:
            print(mode)
        print()

        pump_mode = find_mode(modes, pump_omega)
        mixing_mode = find_mode(modes, mixing_omega)
        pumps = {
            pump_mode: raman.pump.ConstantPump(pump_power),
            mixing_mode: raman.pump.RectangularPump(
                power = mixing_power,
                start_time = 200 * u.nsec,
            ),
        }
        print(f'Pump: {pump_mode}')
        print(f'Mixing: {mixing_mode}')

        spec = raman.FourWaveMixingSpecification(
            name = f'fwm',
            material = material,
            mode_volume_integrator = mock.MockVolumeIntegrator(
                volume_integral_result = 1e-25,
            ),
            modes = modes,
            mode_initial_amplitudes = dict(zip(modes, itertools.repeat(0))),
            mode_intrinsic_quality_factors = dict(zip(modes, itertools.repeat(1e8))),
            mode_coupling_quality_factors = dict(zip(modes, itertools.repeat(1e8))),
            mode_pumps = pumps,
            time_initial = 0 * u.nsec,
            time_final = 1 * u.usec,
            time_step = .01 * u.nsec,
            store_mode_amplitudes_vs_time = True,
            animators = [
                raman.anim.SquareAnimator(
                    axman = raman.anim.PolarComplexAmplitudeAxis(),
                    postfix = '__polar_complex_amplitude_animation',
                    length = 60,
                    fps = 60,
                    **ANIM_KWARGS,
                ),
            ],
        )

        print(spec.info())

        sim = spec.to_sim()

        sim.plot.mode_pump_powers_vs_time(
            **PLOT_KWARGS,
            y_lower_limit = .1 * u.uW,
            y_upper_limit = 1 * u.mW,
            y_log_axis = True,
        )

        sim.run(progress_bar = True)

        sim.plot.mode_magnitudes_vs_time(**PLOT_KWARGS)
        sim.plot.mode_energies_vs_time(**PLOT_KWARGS)
