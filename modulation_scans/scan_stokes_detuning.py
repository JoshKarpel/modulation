"""
Investigate the effect of detuning the Stokes modes from being exactly pump - modulation.
Uses just two mock modes, a pump and a Stokes.
"""

import numpy as np

import simulacra as si
import simulacra.units as u

import modulation

from . import shared


def main():
    ### QUESTIONS

    map_id = shared.ask_map_id()
    spec_type = shared.ask_spec_type()
    material = shared.ask_material()
    index_of_refraction = si.cluster.ask_for_input(
        'Index of refraction?',
        default = 1.45,
        cast_to = float,
    )

    time_final = shared.ask_time_final()
    time_step = shared.ask_time_step()

    pump_intrinsic_q = si.cluster.ask_for_input(
        'Pump mode intrinsic quality factor?',
        default = 1e8,
        cast_to = float,
    )
    pump_coupling_q = si.cluster.ask_for_input(
        'Pump mode coupling quality factor?',
        default = pump_intrinsic_q,
        cast_to = float,
    )
    stokes_intrinsic_q = si.cluster.ask_for_input(
        'Stokes mode intrinsic quality factor?',
        default = 1e8,
        cast_to = float,
    )
    stokes_coupling_q = si.cluster.ask_for_input(
        'Stokes mode coupling quality factor?',
        default = stokes_intrinsic_q,
        cast_to = float,
    )

    pump_mode = modulation.resonator.mock.MockMode(
        label = 'Pump',
        omega = u.twopi * u.c / (u.nm * si.cluster.ask_for_input(
            'Pump mode wavelength (in nm)?',
            default = 1064,
            cast_to = float,
        )),
        mode_volume_inside_resonator = si.cluster.ask_for_input(
            'Pump mode volume inside resonator?',
            default = 1e-20,
            cast_to = float,
        ),
        mode_volume_outside_resonator = 0,
        index_of_refraction = index_of_refraction,
    )
    stokes_mode_volume = si.cluster.ask_for_input(
        'Stokes mode volume inside resonator?',
        default = pump_mode.mode_volume_inside_resonator,
        cast_to = float,
    )
    mode_volume_integrator = modulation.resonator.mock.MockVolumeIntegrator(
        volume_integral_result = si.cluster.ask_for_input(
            'Four-mode overlap integral result?',
            default = 1e-25,
            cast_to = float,
        ),
    )

    stokes_detunings = u.twopi * u.GHz * np.array(si.cluster.ask_for_eval(
        'Stokes mode detunings (in GHz)?',
        default = 'np.linspace(-100, 100, 200)',
    ))

    pump = modulation.raman.pump.ConstantPump(
        power = u.uW * si.cluster.ask_for_input(
            'Pump power (in uW)?',
            default = 100,
            cast_to = float,
        )
    )

    store_mode_amplitudes_vs_time = si.cluster.ask_for_bool('Store mode amplitudes vs time?')

    ### CREATE SPECS

    base_spec_kwargs = dict(
        time_final = time_final,
        time_step = time_step,
        material = material,
        mode_pumps = {pump_mode: pump},
        mode_volume_integrator = mode_volume_integrator,
        checkpoints = True,
        store_mode_amplitudes_vs_time = store_mode_amplitudes_vs_time,
    )

    specs = []
    for stokes_detuning in stokes_detunings:
        stokes_mode = modulation.resonator.mock.MockMode(
            label = 'Stokes',
            omega = pump_mode.omega - material.modulation_omega + stokes_detuning,
            mode_volume_inside_resonator = stokes_mode_volume,
            mode_volume_outside_resonator = 0,
            index_of_refraction = index_of_refraction,
        )

        spec = spec_type(
            f'detuning={stokes_detuning / u.twopi / u.GHz:.6f}GHz',
            modes = [pump_mode, stokes_mode],
            mode_intrinsic_quality_factors = {
                pump_mode: pump_intrinsic_q,
                stokes_mode: stokes_intrinsic_q,
            },
            mode_coupling_quality_factors = {
                pump_mode: pump_coupling_q,
                stokes_mode: stokes_coupling_q,
            },
            **base_spec_kwargs,
            _pump_mode = pump_mode,
            _stokes_mode = stokes_mode,
            _detuning_omega = stokes_detuning,
        )

        specs.append(spec)

    ### CREATE MAP

    map = shared.create_map(map_id, specs)


if __name__ == '__main__':
    main()
