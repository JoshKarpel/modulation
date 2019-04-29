import datetime
from pathlib import Path

import numpy as np
import scipy.optimize as opt
from scipy.special import comb

import simulacra as si
import simulacra.units as u

import modulation
from modulation import raman
from modulation.resonators import mock

from . import shared

import htmap


def create_scan(tag):
    parameters = []

    parameters.append(si.cluster.Parameter("spec_type", shared.ask_spec_type()))

    material = shared.ask_material()

    parameters.append(
        si.cluster.Parameter(
            "mode_volume",
            si.cluster.ask_for_input(
                "Mode volume inside resonator?", default=1e-20, cast_to=float
            ),
        )
    )
    mvi = modulation.resonators.mock.MockVolumeIntegrator(
        volume_integral_result=si.cluster.ask_for_input(
            "Four-mode overlap integral result?", default=1e-25, cast_to=float
        )
    )

    shared.ask_laser_parameters("pump", parameters)
    shared.ask_laser_parameters("mixing", parameters)

    parameters.extend(
        [
            si.cluster.Parameter(
                "stokes_mode_detuning",
                u.MHz
                * np.array(
                    si.cluster.ask_for_eval("Stokes mode detunings (in MHz)?", "[0]")
                ),
                expandable=True,
            ),
            si.cluster.Parameter(
                "modulated_mode_detuning",
                u.MHz
                * np.array(
                    si.cluster.ask_for_eval("Modulated mode detunings (in MHz)?", "[0]")
                ),
                expandable=True,
            ),
        ]
    )

    shared.ask_time_final(parameters)
    shared.ask_time_step(parameters)

    shared.ask_intrinsic_q(parameters)

    shared.ask_four_mode_detuning_cutoff(parameters)

    store_mode_amplitudes_vs_time = si.cluster.ask_for_bool(
        "Store mode amplitudes vs time?"
    )
    lookback_time = shared.ask_lookback_time()

    # CREATE SPECS

    extra_parameters = dict(
        material=material,
        mode_volume_integrator=mvi,
        checkpoints=True,
        checkpoint_every=datetime.timedelta(minutes=20),
        store_mode_amplitudes_vs_time=store_mode_amplitudes_vs_time,
        lookback=modulation.raman.Lookback(lookback_time=lookback_time),
    )

    print("Expanding parameters...")
    expanded_parameters = si.cluster.expand_parameters(parameters)

    final_parameters = [
        dict(component=c, **params, **extra_parameters)
        for c, params in enumerate(expanded_parameters)
    ]

    opts, custom = shared.ask_map_options()

    # CREATE MAP

    map = run.map(
        final_parameters,
        map_options=htmap.MapOptions(**opts, custom_options=custom),
        tag=tag,
    )

    print(f"Created map {map.tag}")

    return map


@htmap.mapped(map_options=htmap.MapOptions(custom_options={"is_resumable": "true"}))
def run(params):
    with si.utils.LogManager("modulation", "simulacra") as logger:
        sim_path = Path.cwd() / f"{params['component']}.sim"

        try:
            sim = si.Simulation.load(sim_path.as_posix())
            logger.info(f"Recovered checkpoint from {sim_path}")
        except (FileNotFoundError, EOFError) as e:
            logger.info(f"Checkpoint not found at {sim_path}")

            mode_volume = params["mode_volume"]
            material = params["material"]

            pump_mode_omega = u.twopi * u.c / params["launched_pump_wavelength"]
            stokes_mode_omega = (
                pump_mode_omega
                - material.modulation_omega
                + (u.twopi * params["stokes_mode_detuning"])
            )
            mixing_mode_omega = u.twopi * u.c / params["launched_mixing_wavelength"]
            modulated_mode_omega = (
                mixing_mode_omega
                + material.modulation_omega
                + (u.twopi * params["modulated_mode_detuning"])
            )

            params["pump_mode_omega"] = pump_mode_omega
            params["stokes_mode_omega"] = stokes_mode_omega
            params["mixing_mode_omega"] = mixing_mode_omega
            params["modulated_mode_omega"] = modulated_mode_omega

            pump_mode = modulation.resonators.mock.MockMode(
                label="Pump",
                omega=pump_mode_omega,
                mode_volume_inside_resonator=mode_volume,
                mode_volume_outside_resonator=0,
                index_of_refraction=material.index_of_refraction(
                    u.c / (pump_mode_omega / u.twopi)
                ),
            )
            stokes_mode = modulation.resonators.mock.MockMode(
                label="Stokes",
                omega=stokes_mode_omega,
                mode_volume_inside_resonator=mode_volume,
                mode_volume_outside_resonator=0,
                index_of_refraction=material.index_of_refraction(
                    u.c / (stokes_mode_omega / u.twopi)
                ),
            )
            mixing_mode = modulation.resonators.mock.MockMode(
                label="Mixing",
                omega=mixing_mode_omega,
                mode_volume_inside_resonator=mode_volume,
                mode_volume_outside_resonator=0,
                index_of_refraction=material.index_of_refraction(
                    u.c / (mixing_mode_omega / u.twopi)
                ),
            )
            modulated_mode = modulation.resonators.mock.MockMode(
                label="Modulated",
                omega=modulated_mode_omega,
                mode_volume_inside_resonator=mode_volume,
                mode_volume_outside_resonator=0,
                index_of_refraction=material.index_of_refraction(
                    u.c / (modulated_mode_omega / u.twopi)
                ),
            )

            modes = [pump_mode, stokes_mode, mixing_mode, modulated_mode]

            pumps = [
                raman.pump.ConstantMonochromaticPump(
                    frequency=pump_mode.frequency
                    + params.get(f"launched_pump_detuning", 0),
                    power=params[f"launched_pump_power"],
                ),
                raman.pump.ConstantMonochromaticPump(
                    frequency=mixing_mode.frequency
                    + params.get(f"launched_mixing_detuning", 0),
                    power=params[f"launched_mixing_power"],
                ),
            ]

            spec = params["spec_type"](
                params["component"],
                modes=modes,
                pumps=pumps,
                mode_intrinsic_quality_factors={
                    m: params["intrinsic_q"] for m in modes
                },
                mode_coupling_quality_factors={m: params["intrinsic_q"] for m in modes},
                mode_initial_amplitudes={m: 1e-15 for m in modes},
                **params,
            )

            sim = spec.to_sim()

        print(sim.info())
        print(sim.spec.mode_info())

        sim.run(checkpoint_callback=htmap.checkpoint)

        print(sim.info())
        print(sim.spec.mode_info())

        sim.polarization_sum_factors = None
        return sim


def main():
    shared.ask_htmap_settings()

    tag = shared.ask_for_tag()

    create_scan(tag)


if __name__ == "__main__":
    main()
