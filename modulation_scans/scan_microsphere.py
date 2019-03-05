import sys
import datetime
from pathlib import Path

import numpy as np

import htmap

import simulacra as si
import simulacra.units as u

import modulation
from modulation import raman
from modulation.resonators import microspheres

from . import shared


def main():
    shared.set_htmap_settings()

    tag = shared.ask_for_tag()

    parameters = []

    parameters.append(si.cluster.Parameter("spec_type", shared.ask_spec_type()))

    material = shared.ask_material()
    microsphere = microspheres.Microsphere(
        radius=u.um
        * si.cluster.ask_for_input(
            "Microsphere radius (in um)?", cast_to=float, default=50
        ),
        index_of_refraction=material.index_of_refraction,
    )
    mvi = microspheres.FixedGridSimpsonMicrosphereVolumeIntegrator(
        microsphere=microsphere
    )

    parameters.extend(
        [
            si.cluster.Parameter(
                "stokes_orders",
                si.cluster.ask_for_input(
                    "Number of Stokes Orders?", cast_to=int, default=1
                ),
            ),
            si.cluster.Parameter(
                "antistokes_orders",
                si.cluster.ask_for_input(
                    "Number of Anti-Stokes Orders?", cast_to=int, default=0
                ),
            ),
            si.cluster.Parameter(
                "group_bandwidth",
                material.raman_linewidth
                * si.cluster.ask_for_input(
                    "Mode Group Bandwidth (in Raman Linewidths)",
                    cast_to=float,
                    default=0.2,
                ),
            ),
            si.cluster.Parameter(
                "max_radial_mode_number",
                si.cluster.ask_for_input(
                    "Maximum Radial Mode Number?", cast_to=int, default=5
                ),
            ),
        ]
    )

    pump_selection_method = si.cluster.ask_for_choices(
        "Pump Wavelength Selection Method?",
        choices={"raw": "raw", "offset": "offset", "symmetric": "symmetric"},
    )
    parameters.append(
        si.cluster.Parameter("pump_selection_method", pump_selection_method)
    )

    if pump_selection_method == "raw":
        parameters.append(
            si.cluster.Parameter(
                "pump_wavelength",
                u.nm
                * np.array(
                    si.cluster.ask_for_eval(
                        "Pump laser wavelength (in nm)?", default="[1064]"
                    )
                ),
                expandable=True,
            )
        )
    elif pump_selection_method == "offset":
        parameters.extend(
            [
                si.cluster.Parameter(
                    "pump_wavelength",
                    u.nm
                    * si.cluster.ask_for_input(
                        "Pump laser wavelength (in nm)?", default=1064, cast_to=float
                    ),
                ),
                si.cluster.Parameter(
                    "pump_frequency_offset",
                    u.GHz
                    * np.array(
                        si.cluster.ask_for_eval(
                            "Pump laser frequency offsets (in GHz)", default="[0]"
                        )
                    ),
                    expandable=True,
                ),
            ]
        )
    elif pump_selection_method == "symmetric":
        pump_wavelength = u.nm * si.cluster.ask_for_input(
            "Pump laser wavelength (in nm)?", default=1064, cast_to=float
        )
        pump_frequency_offsets_raw = u.GHz * np.array(
            si.cluster.ask_for_eval(
                "Pump laser frequency offsets (in GHz)", default="[0]"
            )
        )
        pump_frequency_offsets_abs = np.array(
            sorted(set(np.abs(pump_frequency_offsets_raw)))
        )
        if pump_frequency_offsets_abs[0] != 0:
            pump_frequency_offsets_abs = np.insert(pump_frequency_offsets_abs, 0, 0)
        pump_frequency_offsets = np.concatenate(
            (-pump_frequency_offsets_abs[:0:-1], pump_frequency_offsets_abs)
        )

        parameters.extend(
            [
                si.cluster.Parameter("pump_wavelength", pump_wavelength),
                si.cluster.Parameter(
                    "pump_frequency_offset", pump_frequency_offsets, expandable=True
                ),
            ]
        )

    parameters.append(
        si.cluster.Parameter(
            "pump_power",
            u.uW
            * np.array(
                si.cluster.ask_for_eval(f"Launched power (in uW)?", default="[1000]")
            ),
            expandable=True,
        )
    )

    parameters.extend(
        [
            si.cluster.Parameter("time_final", shared.ask_time_final(default=10)),
            si.cluster.Parameter("time_step", shared.ask_time_step(default=1)),
        ]
    )

    DEFAULT_Q = 1e8
    parameters.append(
        si.cluster.Parameter(
            "intrinsic_q",
            si.cluster.ask_for_input(
                "Mode Intrinsic Quality Factor?", cast_to=float, default=DEFAULT_Q
            ),
        )
    )

    store_mode_amplitudes_vs_time = si.cluster.ask_for_bool(
        "Store mode amplitudes vs time?"
    )

    lookback_time = shared.ask_lookback_time()

    # CREATE SPECS

    extra_parameters = dict(
        microsphere=microsphere,
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

    if not si.cluster.ask_for_bool(
        f"Launch a map with {len(final_parameters)} simulations?"
    ):
        sys.exit(1)

    # CREATE MAP
    opts, custom = shared.ask_map_options()

    map = run.map(
        final_parameters,
        map_options=htmap.MapOptions(**opts, custom_options=custom),
        tag=tag,
    )

    print(f"Created map {map.tag}")

    return map


@htmap.mapped
def run(params):
    with si.utils.LogManager("modulation", "simulacra") as logman:
        sim_path = Path.cwd() / f"{params['component']}.sim"
        try:
            sim = si.Simulation.load(str(sim_path))
            logman.info(f"Recovered checkpoint from {sim_path}")
        except (FileNotFoundError, EOFError):
            logman.info("No checkpoint found")

            bounds = microspheres.sideband_bounds(
                pump_wavelength=params["pump_wavelength"],
                stokes_orders=params["stokes_orders"],
                antistokes_orders=params["antistokes_orders"],
                sideband_frequency=params["material"].modulation_frequency,
                bandwidth_frequency=params["group_bandwidth"],
            )
            logman.info(f"Found {len(bounds)} bounds:")
            for bound in bounds:
                print(bound)
            print()

            modes = shared.find_modes(
                bounds, params["microsphere"], params["max_radial_mode_number"]
            )
            logman.info(f"Found {len(modes)} modes:")
            for mode in modes:
                print(mode)
            print()

            if params["pump_selection_method"] == "raw":
                pumps = [
                    raman.pump.ConstantMonochromaticPump.from_wavelength(
                        wavelength=params["pump_wavelength"], power=params["pump_power"]
                    )
                ]
            elif params["pump_selection_method"] in ("offset", "symmetric"):
                pump_mode = shared.find_mode_nearest_wavelength(
                    params["pump_wavelength"]
                )
                pumps = [
                    raman.pump.ConstantMonochromaticPump(
                        frequency=pump_mode.frequency + params["pump_frequency_offset"],
                        power=params["pump_power"],
                    )
                ]

            spec = params["spec_type"](
                params["component"],
                modes=modes,
                pumps=pumps,
                mode_initial_amplitudes={m: 1 for m in modes},
                mode_intrinsic_quality_factors={
                    m: params["intrinsic_q"] for m in modes
                },
                mode_coupling_quality_factors={m: params["intrinsic_q"] for m in modes},
                **params,
            )

            sim = spec.to_sim()

        print(sim.info())

        sim.run(checkpoint_callback=htmap.checkpoint)

        print(sim.info())

        return sim


if __name__ == "__main__":
    map = main()
