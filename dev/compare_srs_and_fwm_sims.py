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
        ]
        print('Modes:')
        for mode in modes:
            print(mode)
        print()

        pump_mode = find_mode(modes, pump_omega)
        stokes_modes = modes[1]
        pumps = {pump_mode: raman.pump.ConstantPump(pump_power)}

        spec_kwargs = dict(
            material = material,
            mode_volume_integrator = mock.MockVolumeIntegrator(
                volume_integral_result = 1e-25,
            ),
            modes = modes,
            mode_initial_amplitudes = {stokes_modes: 1},
            mode_intrinsic_quality_factors = dict(zip(modes, itertools.repeat(1e8))),
            mode_coupling_quality_factors = dict(zip(modes, itertools.repeat(1e8))),
            mode_pumps = pumps,
            time_initial = 0 * u.nsec,
            time_final = 1 * u.usec,
            time_step = .01 * u.nsec,
            # time_step = 1 * u.psec,
            # time_step = 1e5 * u.fsec,
            store_mode_amplitudes_vs_time = True,
            # animators = [
            #     raman.anim.SquareAnimator(
            #         axman = raman.anim.PolarComplexAmplitudeAxis(),
            #         postfix = '__polar_complex_amplitude_animation',
            #         length = 60,
            #         fps = 60,
            #         **ANIM_KWARGS,
            #     ),
            # ],
        )

        srs = raman.StimulatedRamanScatteringSpecification(
            'srs',
            **spec_kwargs,
        ).to_sim()
        fwm = raman.FourWaveMixingSpecification(
            'fwm',
            **spec_kwargs,
        ).to_sim()

        print(srs.info())
        print(fwm.info())

        # srs.polarization_sum_factors = np.real(srs.polarization_sum_factors)
        # srs.polarization_sum_factors = -srs.polarization_sum_factors.T
        # fwm.polarization_sum_factors = np.real(fwm.polarization_sum_factors)

        srs.run(progress_bar = True)
        srs.plot.mode_magnitudes_vs_time(
            y_log_axis = False,
            **PLOT_KWARGS,
        )

        fwm.run(progress_bar = True)
        fwm.plot.mode_magnitudes_vs_time(
            y_log_axis = False,
            **PLOT_KWARGS,
        )

        si.vis.xy_plot(
            'compare',
            srs.times,
            srs.mode_magnitudes_vs_time[:, 0],
            srs.mode_magnitudes_vs_time[:, 1],
            fwm.mode_magnitudes_vs_time[:, 0],
            fwm.mode_magnitudes_vs_time[:, 1],
            line_labels = [
                'srs stokes',
                'srs pump',
                'fwm stokes',
                'fwm pump',
            ],
            line_kwargs = [
                None,
                None,
                {'linestyle': '--'},
                {'linestyle': '--'},
            ],
            x_unit = 'nsec',
            **PLOT_KWARGS,
        )
        #
        # print("SRS")
        # print(srs.polarization_sum_factors)
        #
        # print("FWM")
        # print(fwm.polarization_sum_factors)

        # print(srs.mode_omegas)
        # print(fwm.mode_omegas)
        #
        # print(srs.polarization_prefactor)
        # print(fwm.polarization_prefactor)
        # # print(srs.calculate_polarization(mode_amplitudes = np.array([1, 1]), time = 0))
        #
        # # print(srs.mode_amplitudes)
        # # print(fwm.mode_amplitudes)
        # #
        # # print(srs.polarization_sum_factors[0, 1])
        # # print(srs.spec.modes[0])
        # # print(srs.spec.modes[1])
        # print(pump_mode)

        # f = np.linspace(-50, 50, 10_000) * u.THz
        # d = srs._double_inverse_detuning(u.twopi * f, 0)
        # l = 2e-13
        # si.vis.xy_plot(
        #     'inv_det',
        #     f,
        #     np.imag(d),
        #     np.where(np.imag(d) > 0, l / 2, 0),
        #     np.where(np.imag(d) < 0, l / 2, 0),
        #     x_unit = 'THz',
        #     y_lower_limit = -l,
        #     y_upper_limit = l,
        #     vlines = [-material.modulation_omega / u.twopi, material.modulation_omega / u.twopi],
        #     **PLOT_KWARGS,
        # )

        # print(srs._double_inverse_detuning(material.modulation_omega, 0))
        # print(srs._double_inverse_detuning(0, material.modulation_omega))
