#!/usr/bin/env python
import itertools
import logging
from pathlib import Path

import numpy as np

import simulacra as si
import simulacra.units as u

from modulation import raman
from modulation.resonator import mock

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


def find_pump_mode(modes, pump_omega: float):
    return sorted(modes, key = lambda m: abs(m.omega - pump_omega))[0]


if __name__ == '__main__':
    with LOGMAN as logger:
        pump_wavelength = 800 * u.nm
        pump_power = 500 * u.uW

        ###

        material = raman.material.RamanMaterial.from_database('silica')
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

        coupling_q = {}  # critical at pump
        for idx, mode in enumerate(modes):
            coupling_q[mode] = 1e8 * (0.9 ** idx)

        spec = raman.StimulatedRamanScatteringSpecification(
            name = f'sidebands__power={pump_power / u.uW:.6f}uW',
            material = material,
            mode_volume_integrator = mock.MockVolumeIntegrator(
                volume_integral_result = 1e-25,
            ),
            modes = modes,
            mode_initial_amplitudes = dict(zip(modes, itertools.repeat(0))),
            mode_intrinsic_quality_factors = dict(zip(modes, itertools.repeat(1e8))),
            mode_coupling_quality_factors = coupling_q,
            mode_pumps = pumps,
            time_initial = 0 * u.usec,
            time_final = 1 * u.usec,
            time_step = .5 * u.nsec,
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

        sim = spec.to_sim()

        print(sim.spec.mode_intrinsic_quality_factors)
        print(sim.spec.mode_coupling_quality_factors)
        print(sim.spec.mode_total_quality_factors)

        # sim.run(show_progress_bar = True)
        #
        # sim.plot.mode_magnitudes_vs_time(**PLOT_KWARGS)
        # sim.plot.mode_energies_vs_time(**PLOT_KWARGS)
