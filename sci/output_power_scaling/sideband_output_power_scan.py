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
from modulation.resonators import mock
from modulation.resonators.microspheres import coupling_quality_factor_for_tapered_fiber

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

        sim = spec.to_sim()
        sim.run()

        # sim.plot.mode_magnitudes_vs_time(**PLOT_KWARGS)

        return sim


def find_pump_mode(modes, pump_omega: float):
    return sorted(modes, key = lambda m: abs(m.omega - pump_omega))[0]


def make_output_power_scan_plot(pump_powers, results):
    pump_to_sim = dict(zip(pump_powers, results))
    modes = results[0].spec.modes

    y = []
    labels = []
    line_kwargs = []
    mode_energies = []
    for q, mode in enumerate(modes):
        output_power = np.array([
            sim.mode_output_powers(sim.mode_amplitudes)[q]
            for pump, sim in pump_to_sim.items()
        ])
        mode_energy = np.array([
            sim.mode_energies(sim.mode_amplitudes)[q]
            for pump, sim in pump_to_sim.items()
        ])

        y.append(output_power)
        labels.append(mode.tex)
        line_kwargs.append({})
        mode_energies.append(mode_energy)

    thresholds = calculate_thresholds(mode_energies, pump_powers)
    thresholds = np.array(list(reversed(thresholds)))

    si.vis.xy_plot(
        f'sideband_output_power_scaling',
        pump_powers,
        *y,
        line_labels = labels,
        line_kwargs = line_kwargs,
        x_unit = 'uW',
        x_label = 'Pump Power',
        y_unit = 'uW',
        y_label = 'Output Power',
        vlines = thresholds,
        legend_on_right = True,
        # y_lower_limit = 0 * u.uW,
        # y_upper_limit = 35 * u.uW,
        **PLOT_KWARGS,
    )


def make_mode_energy_scan_plot(pump_powers, results):
    pump_to_sim = dict(zip(pump_powers, results))
    modes = results[0].spec.modes

    y = []
    labels = []
    line_kwargs = []
    for q, mode in enumerate(modes):
        mode_energy = np.array([
            sim.mode_energies(sim.mode_amplitudes)[q]
            for pump, sim in pump_to_sim.items()
        ])

        y.append(mode_energy)
        labels.append(mode.tex)
        line_kwargs.append(dict())

    thresholds = calculate_thresholds(y, pump_powers)
    orders = np.array(list(range(len(thresholds))))
    thresholds = np.array(list(reversed(thresholds)))

    si.vis.xy_plot(
        f'sideband_mode_energy_scaling',
        pump_powers,
        *y,
        line_labels = labels,
        line_kwargs = line_kwargs,
        x_unit = 'uW',
        x_label = 'Pump Power',
        y_unit = 'pJ',
        y_label = 'Mode Energy',
        vlines = thresholds,
        # y_lower_limit = 0 * u.pJ,
        # y_upper_limit = 1.2 * u.pJ,
        legend_on_right = True,
        **PLOT_KWARGS,
    )

    fit, _ = opt.curve_fit(
        lambda t, a, n: a * (t ** n),
        orders,
        thresholds,
        p0 = [1, 3],
    )

    dense_orders = np.linspace(orders[0], orders[-1], 1000)
    fitted_thresholds = fit[0] * (dense_orders ** fit[1])

    si.vis.xxyy_plot(
        f'sideband_thresholds',
        [orders, dense_orders],
        [thresholds, fitted_thresholds],
        x_label = 'Stokes Order',
        line_kwargs = [
            {'linestyle': '', 'marker': 'o', 'color': 'blue'},
            {'linestyle': '-', 'color': 'black', },
        ],
        line_labels = [
            'Simulation',
            rf'${fit[0]:.3g} \times n^{{{fit[1]:.3g}}}$',
        ],
        y_unit = 'uW',
        y_label = 'Pump Power',
        vlines = thresholds,
        legend_on_right = True,
        **PLOT_KWARGS,
    )


def calculate_thresholds(mode_energies, pump_powers):
    thresholds = []
    for (q, mode), mode_energy in zip(enumerate(modes), mode_energies):
        indices, = np.where(mode_energy > 2 * mode.photon_energy)
        if len(indices) > 0:
            thresholds.append(pump_powers[indices[0]])

    return thresholds


def calculate_coupling_quality_factors(
    modes,
    microsphere_radius = 50 * u.um,
    index_of_refraction = 1.45,
    separation = 300 * u.nm,
    fiber_taper_radius = 1 * u.um,
):
    return {
        mode:
            coupling_quality_factor_for_tapered_fiber(
                microsphere_index_of_refraction = ConstantIndex(index_of_refraction),
                fiber_index_of_refraction = ConstantIndex(index_of_refraction),
                microsphere_radius = microsphere_radius,
                fiber_taper_radius = fiber_taper_radius,
                wavelength = u.c / (mode.omega / u.twopi),
                separation = separation,
                l = 0,
                m = 0,
            )
        for mode in modes}


if __name__ == '__main__':
    np.set_printoptions(linewidth = 200)
    with LOGMAN as logger:
        pump_wavelength = 980 * u.nm
        pump_powers = np.linspace(0, 60, 50) * u.uW

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
            for q in range(4)
        ]
        print('Modes:')
        for mode in modes:
            print(mode)

        pump_mode = find_pump_mode(modes, pump_omega)
        print(f'Pump: {pump_mode}')

        coupling_q = calculate_coupling_quality_factors(modes)

        for k, v in coupling_q.items():
            print(k, f'{v:.3g}')

        spec_kwargs = dict(
            material = material,
            mode_volume_integrator = mock.MockVolumeIntegrator(
                volume_integral_result = 1e-25,
            ),
            modes = modes,
            mode_initial_amplitudes = dict(zip(modes, itertools.repeat(0))),
            mode_intrinsic_quality_factors = dict(zip(modes, itertools.repeat(coupling_q[pump_mode]))),
            mode_coupling_quality_factors = coupling_q,
            time_initial = 0 * u.usec,
            time_final = 10 * u.usec,
            time_step = 1 * u.nsec,
            store_mode_amplitudes_vs_time = True,
        )

        specs = []
        for pump_power in pump_powers:
            pumps = {pump_mode: raman.pump.ConstantPump(pump_power)}
            spec = raman.RamanSidebandSpecification(
                name = f'sideband_output_power__power={pump_power / u.uW:.6f}uW',
                mode_pumps = pumps,
                **spec_kwargs,
            )
            specs.append(spec)

        results = si.utils.multi_map(run, specs, processes = 3)

        make_mode_energy_scan_plot(pump_powers, results)
        make_output_power_scan_plot(pump_powers, results)

        spec = specs[-1]
        sim = spec.to_sim()

        print(sim.info())
        sim.run(progress_bar = True)

        G = np.real(sim.polarization_sum_factors / sim.mode_omegas)[0, 1]
        print('G')
        print(G)

        print('Energy of q = 1 from 1/2GQ')
        amp_squared = (1 / (2 * sim.spec.mode_total_quality_factors[1] * G))
        predicted = sim.mode_energy_prefactor[1] * amp_squared
        print(predicted / u.pJ)

        print('Energy of q = 1 from Amplitudes')
        actual = sim.mode_energies(sim.mode_amplitudes)[1]
        print(actual / u.pJ)

        print(.5 * (actual - predicted) / (actual + predicted))
