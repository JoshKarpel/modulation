import datetime
import itertools
from pathlib import Path

import numpy as np
import scipy.optimize as opt
from scipy.special import comb

import simulacra as si
import simulacra.units as u

import modulation
from modulation import raman
from modulation.resonators import mock, microspheres

from . import shared

import htmap


def create_scan(tag):
    parameters = []

    parameters.append(si.cluster.Parameter("spec_type", shared.ask_spec_type()))

    material = shared.ask_material()

    parameters.extend(
        [
            si.cluster.Parameter(
                "microsphere_radius",
                u.um
                * si.cluster.ask_for_input(
                    "Microsphere radius (in um)?", default=50, cast_to=float
                ),
            ),
            si.cluster.Parameter(
                "fiber_taper_radius",
                u.um
                * si.cluster.ask_for_input(
                    "Fiber taper radius (in um)?", default=1, cast_to=float
                ),
            ),
        ]
    )

    parameters.append(
        si.cluster.Parameter(
            "mode_volume",
            si.cluster.ask_for_eval("Mode volume inside resonator?", default="[1e-20]"),
            expandable=True,
        )
    )
    mvi = modulation.resonators.mock.MockVolumeIntegrator(
        volume_integral_result=si.cluster.ask_for_input(
            "Four-mode overlap integral result?", default=1e-25, cast_to=float
        )
    )

    shared.ask_laser_parameters("pump", parameters)
    shared.ask_laser_parameters("mixing", parameters)

    num_pump_stokes = si.cluster.ask_for_input(
        "Number of Pump Stokes Orders?", default=1, cast_to=int
    )
    num_pump_antistokes = si.cluster.ask_for_input(
        "Number of Pump Antistokes Orders?", default=1, cast_to=int
    )
    num_mixing_stokes = si.cluster.ask_for_input(
        "Number of Mixing Stokes Orders?", default=1, cast_to=int
    )
    num_mixing_antistokes = si.cluster.ask_for_input(
        "Number of Mixing Antistokes Orders?", default=1, cast_to=int
    )

    orders = [f"pump|{n:+}" for n in range(-num_pump_stokes, num_pump_antistokes + 1)]
    orders += [
        f"mixing|{n:+}" for n in range(-num_mixing_stokes, num_mixing_antistokes + 1)
    ]

    parameters.append(si.cluster.Parameter("orders", orders))
    for order in orders:
        parameters.append(
            si.cluster.Parameter(
                f"{order}_mode_detuning",
                u.MHz
                * np.array(
                    si.cluster.ask_for_eval(f"{order} mode detuning (in MHz)?", "[0]")
                ),
                expandable=True,
            )
        )

    shared.ask_time_final(parameters)
    shared.ask_time_step(parameters)

    shared.ask_intrinsic_q(parameters)
    parameters.append(
        si.cluster.Parameter(
            "use_scaling_coupling_quality_factor",
            si.cluster.ask_for_bool(
                "Use Scaling Coupling Quality Factor (critical at pump)?", default=True
            ),
        )
    )

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
            mixing_mode_omega = u.twopi * u.c / params["launched_mixing_wavelength"]

            omega_selector = {"pump": pump_mode_omega, "mixing": mixing_mode_omega}

            modes = []
            for order in params["orders"]:
                pump_or_mixing, n = order.split("|")
                base_omega = omega_selector[pump_or_mixing]
                n = int(n)

                mode = modulation.resonators.mock.MockMode(
                    label=order,
                    omega=base_omega
                    + (material.modulation_omega * n)
                    + (u.twopi * params[f"{order}_mode_detuning"]),
                    mode_volume_inside_resonator=mode_volume,
                    mode_volume_outside_resonator=0,
                    index_of_refraction=material.index_of_refraction,
                )

                modes.append(mode)

            order_to_mode = dict(zip(params["orders"], modes))
            params["order_to_mode"] = order_to_mode

            pumps = [
                raman.pump.ConstantMonochromaticPump(
                    frequency=order_to_mode["pump|+0"].frequency
                    + params.get(f"launched_pump_detuning", 0),
                    power=params[f"launched_pump_power"],
                ),
                raman.pump.ConstantMonochromaticPump(
                    frequency=order_to_mode["mixing|+0"].frequency
                    + params.get(f"launched_mixing_detuning", 0),
                    power=params[f"launched_mixing_power"],
                ),
            ]

            intrinsic_q = {m: params["intrinsic_q"] for m in modes}

            if params["use_scaling_coupling_quality_factor"]:
                pump_mode = order_to_mode["pump|+0"]
                kwargs_for_coupling_q = dict(
                    microsphere_index_of_refraction=material.index_of_refraction,
                    fiber_index_of_refraction=material.index_of_refraction,
                    microsphere_radius=params["microsphere_radius"],
                    fiber_taper_radius=params["fiber_taper_radius"],
                    l=0,
                    m=0,
                )
                separation = opt.brentq(
                    lambda x: intrinsic_q[pump_mode]
                    - microspheres.coupling_quality_factor_for_tapered_fiber(
                        separation=x,
                        wavelength=pump_mode.wavelength,
                        **kwargs_for_coupling_q,
                    ),
                    0,
                    10 * u.um,
                )
                logger.info(
                    f"fiber-microsphere separation is {separation / u.nm:.3f} nm"
                )
                coupling_q = {
                    m: microspheres.coupling_quality_factor_for_tapered_fiber(
                        separation=separation,
                        wavelength=m.wavelength,
                        **kwargs_for_coupling_q,
                    )
                    for m in modes
                }
            else:
                coupling_q = intrinsic_q  # all modes critically coupled

            spec = params["spec_type"](
                params["component"],
                modes=modes,
                pumps=pumps,
                mode_intrinsic_quality_factors=intrinsic_q,
                mode_coupling_quality_factors=coupling_q,
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
        del sim.times
        return sim


def main():
    shared.ask_htmap_settings()

    tag = shared.ask_for_tag()

    create_scan(tag)


if __name__ == "__main__":
    main()
