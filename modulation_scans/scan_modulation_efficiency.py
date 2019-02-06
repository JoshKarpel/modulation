import sys

import numpy as np

import simulacra as si
import simulacra.units as u

import modulation

from . import shared


def main():
    # QUESTIONS

    tag = shared.ask_for_tag()

    parameters = []

    spec_type = modulation.raman.FourWaveMixingSpecification
    material = shared.ask_material()
    index_of_refraction = si.cluster.ask_for_input(
        'Index of refraction?',
        default = 1.45,
        cast_to = float,
    )

    time_final = shared.ask_time_final(default = 100)
    time_step = shared.ask_time_step(default = .001)

    mode_volume = si.cluster.ask_for_input(
        'Mode volume inside resonator?',
        default = 1e-20,
        cast_to = float,
    )
    mode_volume_integrator = modulation.resonators.mock.MockVolumeIntegrator(
        volume_integral_result = si.cluster.ask_for_input(
            'Four-mode overlap integral result?',
            default = 1e-25,
            cast_to = float,
        ),
    )

    pump_mode = modulation.resonators.mock.MockMode(
        label = 'Pump',
        omega = u.twopi * u.c / (u.nm * si.cluster.ask_for_input(
            'Pump mode wavelength (in nm)?',
            default = 1064,
            cast_to = float,
        )),
        mode_volume_inside_resonator = mode_volume,
        mode_volume_outside_resonator = 0,
        index_of_refraction = index_of_refraction,
    )
    stokes_mode = modulation.resonators.mock.MockMode(
        label = 'Stokes',
        omega = pump_mode.omega - material.modulation_omega,
        mode_volume_inside_resonator = mode_volume,
        mode_volume_outside_resonator = 0,
        index_of_refraction = index_of_refraction,
    )
    mixing_mode = modulation.resonators.mock.MockMode(
        label = 'Mixing',
        omega = u.twopi * u.c / (u.nm * si.cluster.ask_for_input(
            'Mixing mode wavelength (in nm)?',
            default = 632,
            cast_to = float,
        )),
        mode_volume_inside_resonator = mode_volume,
        mode_volume_outside_resonator = 0,
        index_of_refraction = index_of_refraction,
    )
    modulated_mode = modulation.resonators.mock.MockMode(
        label = 'Modulated',
        omega = mixing_mode.omega + material.modulation_omega,
        mode_volume_inside_resonator = mode_volume,
        mode_volume_outside_resonator = 0,
        index_of_refraction = index_of_refraction,
    )

    DEFAULT_Q = 1e8
    pump_and_stokes_intrinsic_q = si.cluster.ask_for_input(
        'Pump & Stokes modes intrinsic quality factor?',
        default = DEFAULT_Q,
        cast_to = float,
    )
    pump_and_stokes_coupling_q = si.cluster.ask_for_input(
        'Pump & Stokes modes coupling quality factor?',
        default = DEFAULT_Q,
        cast_to = float,
    )
    mixing_intrinsic_q = si.cluster.ask_for_input(
        'Mixing modes intrinsic quality factor?',
        default = DEFAULT_Q,
        cast_to = float,
    )
    mixing_coupling_q = si.cluster.ask_for_input(
        'Mixing mode coupling quality factor?',
        default = DEFAULT_Q,
        cast_to = float,
    )
    modulated_intrinsic_q = si.cluster.ask_for_input(
        'Modulated mode intrinsic quality factor?',
        default = DEFAULT_Q,
        cast_to = float,
    )
    modulated_coupling_q = si.cluster.ask_for_input(
        'Modulated mode coupling quality factor?',
        default = DEFAULT_Q,
        cast_to = float,
    )

    scan_mode, fixed_mode = si.cluster.ask_for_choices(
        'Which launched power is scanned?',
        choices = {
            'Pump': (pump_mode, mixing_mode),
            'Mixing': (mixing_mode, pump_mode),
        },
        default = 'Pump',
    )

    parameters.append(
        si.cluster.Parameter(
            '_scan_power',
            u.uW * si.cluster.ask_for_eval(
                f'Scan mode ({scan_mode.label}) launched power (in uW)?',
                default = 'np.linspace(0, 5000, 100)',
            ),
            expandable = True,
        )
    )
    parameters.append(
        si.cluster.Parameter(
            '_fixed_power',
            u.uW * si.cluster.ask_for_eval(
                f'Fixed mode ({fixed_mode.label}) launched power (in uW)?',
                default = '[1]',
                cast_to = float,
            ),
            expandable = True,
        )
    )

    store_mode_amplitudes_vs_time = si.cluster.ask_for_bool('Store mode amplitudes vs time?')

    lookback_time = shared.ask_lookback_time(time_step, num_modes = 4)

    # CREATE SPECS

    base_spec_kwargs = dict(
        time_final = time_final,
        time_step = time_step,
        material = material,
        mode_initial_amplitudes = {stokes_mode: 1},
        mode_volume_integrator = mode_volume_integrator,
        checkpoints = True,
        store_mode_amplitudes_vs_time = store_mode_amplitudes_vs_time,
        lookback = modulation.raman.Lookback(lookback_time = lookback_time),
    )

    specs = []
    for params in si.cluster.expand_parameters(parameters):
        scan_power = params['_scan_power']
        fixed_power = params['_fixed_power']

        spec = spec_type(
            f'pump_power={scan_power / u.uW:.6f}uW',
            modes = [pump_mode, stokes_mode, mixing_mode, modulated_mode],
            mode_pumps = {
                scan_mode: modulation.raman.ConstantPump(power = scan_power),
                fixed_mode: modulation.raman.ConstantPump(power = fixed_power),
            },
            mode_intrinsic_quality_factors = {
                pump_mode: pump_and_stokes_intrinsic_q,
                stokes_mode: pump_and_stokes_intrinsic_q,
                mixing_mode: mixing_intrinsic_q,
                modulated_mode: modulated_intrinsic_q,
            },
            mode_coupling_quality_factors = {
                pump_mode: pump_and_stokes_coupling_q,
                stokes_mode: pump_and_stokes_coupling_q,
                mixing_mode: mixing_coupling_q,
                modulated_mode: modulated_coupling_q,
            },
            **base_spec_kwargs,
            _pump_mode = pump_mode,
            _stokes_mode = stokes_mode,
            _mixing_mode = mixing_mode,
            _modulated_mode = modulated_mode,
            _scan_mode = scan_mode,
            _fixed_mode = fixed_mode,
            _scan_power = scan_power,
        )

        specs.append(spec)

    if not si.cluster.ask_for_bool(f'Launch a map with {len(specs)} simulations?'):
        sys.exit(1)

    # CREATE MAP
    shared.create_map(tag, specs)


if __name__ == '__main__':
    main()
