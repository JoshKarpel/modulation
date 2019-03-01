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
        "Index of refraction?", default=1.45, cast_to=float
    )

    time_final = shared.ask_time_final(default=10)
    time_step = shared.ask_time_step(default=0.01)

    mode_volume = si.cluster.ask_for_input(
        "Mode volume inside resonator?", default=1e-20, cast_to=float
    )
    mode_volume_integrator = modulation.resonators.mock.MockVolumeIntegrator(
        volume_integral_result=si.cluster.ask_for_input(
            "Four-mode overlap integral result?", default=1e-25, cast_to=float
        )
    )

    pump_mode = modulation.resonators.mock.MockMode(
        label="Pump",
        omega=u.twopi
        * u.c
        / (
            u.nm
            * si.cluster.ask_for_input(
                "Pump mode wavelength (in nm)?", default=1064, cast_to=float
            )
        ),
        mode_volume_inside_resonator=mode_volume,
        mode_volume_outside_resonator=0,
        index_of_refraction=index_of_refraction,
    )
    stokes_mode = modulation.resonators.mock.MockMode(
        label="Stokes",
        omega=pump_mode.omega - material.modulation_omega,
        mode_volume_inside_resonator=mode_volume,
        mode_volume_outside_resonator=0,
        index_of_refraction=index_of_refraction,
    )
    mixing_mode = modulation.resonators.mock.MockMode(
        label="Mixing",
        omega=u.twopi
        * u.c
        / (
            u.nm
            * si.cluster.ask_for_input(
                "Mixing mode wavelength (in nm)?", default=632, cast_to=float
            )
        ),
        mode_volume_inside_resonator=mode_volume,
        mode_volume_outside_resonator=0,
        index_of_refraction=index_of_refraction,
    )
    modulated_mode = modulation.resonators.mock.MockMode(
        label="Modulated",
        omega=mixing_mode.omega + material.modulation_omega,
        mode_volume_inside_resonator=mode_volume,
        mode_volume_outside_resonator=0,
        index_of_refraction=index_of_refraction,
    )

    DEFAULT_Q = 1e8
    parameters.extend(
        [
            si.cluster.Parameter(
                "pump_and_stokes_intrinsic_q",
                si.cluster.ask_for_eval(
                    "Pump & Stokes intrinsic quality factor?",
                    default=f"[{DEFAULT_Q:.4g}]",
                ),
                expandable=True,
            ),
            si.cluster.Parameter(
                "pump_and_stokes_coupling_q",
                si.cluster.ask_for_eval(
                    "Pump & Stokes coupling quality factor?",
                    default=f"[{DEFAULT_Q:.4g}]",
                ),
                expandable=True,
            ),
            si.cluster.Parameter(
                "mixing_and_modulated_intrinsic_q",
                si.cluster.ask_for_eval(
                    "Mixing & Modulated intrinsic quality factor?",
                    default=f"[{DEFAULT_Q:.4g}]",
                ),
                expandable=True,
            ),
            si.cluster.Parameter(
                "mixing_and_modulated_coupling_q",
                si.cluster.ask_for_eval(
                    "Mixing & Modulated coupling quality factor?",
                    default=f"[{DEFAULT_Q:.4g}]",
                ),
                expandable=True,
            ),
        ]
    )

    parameters.append(
        si.cluster.Parameter(
            "pump_power",
            u.uW
            * np.array(
                si.cluster.ask_for_eval(
                    f"Pump mode ({pump_mode}) launched power (in uW)?",
                    default="np.linspace(0, 5000, 100)",
                )
            ),
            expandable=True,
        )
    )
    parameters.append(
        si.cluster.Parameter(
            "mixing_power",
            u.uW
            * np.array(
                si.cluster.ask_for_eval(
                    f"Mixing mode ({mixing_mode}) launched power (in uW)?",
                    default="[1]",
                )
            ),
            expandable=True,
        )
    )

    store_mode_amplitudes_vs_time = si.cluster.ask_for_bool(
        "Store mode amplitudes vs time?"
    )

    lookback_time = shared.ask_lookback_time(time_step, num_modes=4)

    # CREATE SPECS

    base_spec_kwargs = dict(
        time_final=time_final,
        time_step=time_step,
        material=material,
        mode_initial_amplitudes={stokes_mode: 1},
        mode_volume_integrator=mode_volume_integrator,
        checkpoints=True,
        store_mode_amplitudes_vs_time=store_mode_amplitudes_vs_time,
        lookback=modulation.raman.Lookback(lookback_time=lookback_time),
    )

    specs = []
    for c, params in enumerate(si.cluster.expand_parameters(parameters)):
        pump_and_stokes_intrinsic_q = params["pump_and_stokes_intrinsic_q"]
        pump_and_stokes_coupling_q = params["pump_and_stokes_coupling_q"]
        mixing_and_modulated_intrinsic_q = params["mixing_and_modulated_intrinsic_q"]
        mixing_and_modulated_coupling_q = params["mixing_and_modulated_coupling_q"]
        pump_power = params["pump_power"]
        mixing_power = params["mixing_power"]

        spec = spec_type(
            str(c),
            modes=[pump_mode, stokes_mode, mixing_mode, modulated_mode],
            mode_pumps={
                pump_mode: modulation.raman.ConstantMonochromaticPump(power=pump_power),
                mixing_mode: modulation.raman.ConstantMonochromaticPump(
                    power=mixing_power
                ),
            },
            mode_intrinsic_quality_factors={
                pump_mode: pump_and_stokes_intrinsic_q,
                stokes_mode: pump_and_stokes_intrinsic_q,
                mixing_mode: mixing_and_modulated_intrinsic_q,
                modulated_mode: mixing_and_modulated_intrinsic_q,
            },
            mode_coupling_quality_factors={
                pump_mode: pump_and_stokes_coupling_q,
                stokes_mode: pump_and_stokes_coupling_q,
                mixing_mode: mixing_and_modulated_coupling_q,
                modulated_mode: mixing_and_modulated_coupling_q,
            },
            **base_spec_kwargs,
            _pump_mode=pump_mode,
            _stokes_mode=stokes_mode,
            _mixing_mode=mixing_mode,
            _modulated_mode=modulated_mode,
            _pump_power=pump_power,
            _mixing_power=mixing_power,
            _pump_and_stokes_intrinsic_q=pump_and_stokes_intrinsic_q,
            _pump_and_stokes_coupling_q=pump_and_stokes_coupling_q,
            _mixing_and_modulated_intrinsic_q=mixing_and_modulated_intrinsic_q,
            _mixing_and_modulated_coupling_q=mixing_and_modulated_coupling_q,
        )

        specs.append(spec)

    if not si.cluster.ask_for_bool(f"Launch a map with {len(specs)} simulations?"):
        sys.exit(1)

    # CREATE MAP
    shared.create_map(tag, specs)


if __name__ == "__main__":
    main()
