#!/usr/bin/env python
import itertools
import logging
from pathlib import Path

import numpy as np
import scipy.optimize as opt

import simulacra as si
import simulacra.units as u

from modulation import raman
from modulation.refraction import ConstantIndex
from modulation.resonator import mock
from modulation.resonator.microsphere import coupling_quality_factor_for_tapered_fiber

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
    length = 30,
    fps = 60,
)


def run(spec):
    with LOGMAN as logger:
        sim = si.utils.find_or_init_sim_from_spec(spec, search_dir = SIM_LIB)

        # if sim.status is not si.Status.FINISHED:
        #     sim.run()
        #     sim.save(target_dir = SIM_LIB)

        sim.run()
        sim.plot.mode_magnitudes_vs_time(**PLOT_KWARGS)

        return sim


def find_mode(modes, omega: float):
    return sorted(modes, key = lambda m: abs(m.omega - omega))[0]


def make_mode_energy_scan_plot(pump_powers, results, mixing_power):
    pump_to_sim = dict(zip(pump_powers, results))
    modes = results[0].spec.modes

    mode_energies = []
    mode_output_powers = []
    labels = []
    line_kwargs = []
    for q, mode in enumerate(modes):
        averaging_index = int(100 * u.nsec / results[0].spec.time_step)
        mode_energy = np.array([
            np.mean(sim.mode_energies(sim.mode_amplitudes_vs_time)[-averaging_index:, q])
            for pump, sim in pump_to_sim.items()
        ])

        output_power = np.array([
            np.mean(sim.mode_output_powers(sim.mode_amplitudes_vs_time)[-averaging_index:, q])
            for pump, sim in pump_to_sim.items()
        ])

        mode_energies.append(mode_energy)
        mode_output_powers.append(output_power)
        labels.append(mode.tex)
        line_kwargs.append(dict(alpha = 0.8))

    si.vis.xy_plot(
        f'mixing_mode_energy_scaling',
        pump_powers,
        *mode_energies,
        line_labels = labels,
        line_kwargs = line_kwargs,
        x_unit = 'uW',
        x_label = 'Pump Power',
        y_unit = 'pJ',
        y_label = 'Mode Energy',
        legend_on_right = True,
        **PLOT_KWARGS,
    )
    si.vis.xy_plot(
        f'mixing_mode_energy_scaling__log_y',
        pump_powers,
        *mode_energies,
        line_labels = labels,
        line_kwargs = line_kwargs,
        x_unit = 'uW',
        x_label = 'Pump Power',
        y_unit = 'pJ',
        y_label = 'Mode Energy',
        legend_on_right = True,
        y_log_axis = True,
        y_lower_limit = 1e-8 * u.pJ,
        y_upper_limit = 1e2 * u.pJ,
        **PLOT_KWARGS,
    )
    si.vis.xy_plot(
        f'mixing_output_power_scaling',
        pump_powers,
        *mode_output_powers,
        line_labels = labels,
        line_kwargs = line_kwargs,
        x_unit = 'uW',
        x_label = 'Pump Power',
        y_unit = 'uW',
        y_label = 'Output Power',
        legend_on_right = True,
        **PLOT_KWARGS,
    )
    si.vis.xy_plot(
        f'mixing_output_power_scaling__log_y',
        pump_powers,
        *mode_output_powers,
        line_labels = labels,
        line_kwargs = line_kwargs,
        x_unit = 'uW',
        x_label = 'Pump Power',
        y_unit = 'uW',
        y_label = 'Output Power',
        legend_on_right = True,
        y_log_axis = True,
        hlines = [mixing_power],
        y_lower_limit = 1e-3 * u.uW,
        y_upper_limit = 1e2 * u.uW,
        **PLOT_KWARGS,
    )


def calculate_thresholds(mode_energies, pump_powers):
    thresholds = []
    for (q, mode), mode_energy in zip(enumerate(modes), mode_energies):
        indices, = np.where(mode_energy > 2 * mode.photon_energy)
        if len(indices) > 0:
            thresholds.append(pump_powers[indices[0]])

    return thresholds


if __name__ == '__main__':
    np.set_printoptions(linewidth = 200)
    with LOGMAN as logger:
        pump_wavelength = 1064 * u.nm
        mixing_wavelength = 632 * u.nm
        mixing_power = 1 * u.uW

        pump_powers = np.linspace(0, 500, 100) * u.uW

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
            # mock.MockMode(
            #     label = f'mixing-',
            #     omega = mixing_omega - material.modulation_omega,
            #     index_of_refraction = 1.45,
            #     mode_volume_inside_resonator = 1e-20,
            # )
        ]
        print('Modes:')
        for mode in modes:
            print(mode)
        print()

        pump_mode = find_mode(modes, pump_omega)
        mixing_mode = find_mode(modes, mixing_omega)
        print(f'Pump: {pump_mode}')
        print(f'Mixing: {mixing_mode}')

        Q = dict(zip(modes, [1e8, 1e8, 1e5, 1e5]))

        spec_kwargs = dict(
            material = material,
            mode_volume_integrator = mock.MockVolumeIntegrator(
                volume_integral_result = 1e-25,
            ),
            modes = modes,
            mode_initial_amplitudes = dict(zip(modes, itertools.repeat(0))),
            mode_intrinsic_quality_factors = Q,
            mode_coupling_quality_factors = dict(zip(modes, itertools.repeat(1e8))),
            time_initial = 0 * u.nsec,
            time_final = 1 * u.usec,
            time_step = 10 * u.psec,
            store_mode_amplitudes_vs_time = True,
        )

        specs = []
        for pump_power in pump_powers:
            pumps = {
                pump_mode: raman.pump.ConstantPump(pump_power),
                mixing_mode: raman.pump.RectangularPump(
                    power = mixing_power,
                    start_time = 200 * u.nsec,
                ),
            }
            spec = raman.FourWaveMixingSpecification(
                name = f'power={pump_power / u.uW:.6f}uW',
                mode_pumps = pumps,
                **spec_kwargs,
            )
            specs.append(spec)

        results = si.utils.multi_map(run, specs, processes = 10)

        make_mode_energy_scan_plot(pump_powers, results, mixing_power = mixing_power)
