import sys
import datetime
from pathlib import Path

import numpy as np
from scipy.special import comb

import htmap

import simulacra as si
import simulacra.units as u

import modulation
from modulation import raman
from modulation.resonators import microspheres

from . import shared


def init_scan(tag):
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

    # MEMORY AND RUNTIME CHECK

    p = final_parameters[0]

    bounds = microspheres.sideband_bounds(
        pump_wavelength=p["pump_wavelength"],
        stokes_orders=p["stokes_orders"],
        antistokes_orders=p["antistokes_orders"],
        sideband_frequency=p["material"].modulation_frequency,
        bandwidth_frequency=p["group_bandwidth"],
    )

    modes = shared.find_modes(bounds, p["microsphere"], p["max_radial_mode_number"])

    print(f"Approximate number of modes in each simulation: {len(modes)}")

    if p["spec_type"] is modulation.raman.StimulatedRamanScatteringSpecification:
        approximate_psf_time = comb(len(modes), 2, repetition=True) * 0.05
    elif ["spec_type"] is modulation.raman.FourWaveMixingSpecification:
        approximate_psf_time = comb(len(modes), 4, repetition=True) * 0.05
    approximate_psf_time = datetime.timedelta(seconds=int(approximate_psf_time))
    print(
        f"Approximate time to calculate polarization sum factors: {approximate_psf_time}"
    )

    psf_generation = 1.5 * mvi.r_points * mvi.theta_points * (128 / 8) * len(modes)
    print(
        f"Approximate memory requirements for polarization sum factor calculation: {si.utils.bytes_to_str(psf_generation)}"
    )

    psf_storage = (len(modes) ** 4) * (128 / 8)
    print(
        f"Approximate memory requirements for polarization sum factor storage: {si.utils.bytes_to_str(psf_storage)}"
    )
    print("Remember: add about 100 MB for everything else!")

    opts, custom = shared.ask_map_options()

    # CREATE MAP

    map = init.map(
        final_parameters,
        map_options=htmap.MapOptions(**opts, custom_options=custom),
        tag="_INIT__" + tag,
    )

    print(f"Created map {map.tag}")

    return map


def run_scan(tag):
    sims = list(htmap.load(tag))
    max_psf_storage = max(s.polarization_sum_factors.nbytes for s in sims)

    print(
        f"Maximum polarization sum factor memory requirements: {si.utils.bytes_to_str(max_psf_storage)}"
    )
    print("Remember: add about 100 MB for everything else!")

    opts, custom = shared.ask_map_options()

    map = run.map(
        sims,
        map_options=htmap.MapOptions(**opts, custom_options=custom),
        tag=tag.replace("_INIT__", ""),
    )

    print(f"Created map {map.tag}")

    return map


@htmap.mapped
def init(params):
    with si.utils.LogManager("modulation", "simulacra") as logman:
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
                modes, params["pump_wavelength"]
            )
            params["pump_mode"] = pump_mode
            pumps = [
                raman.pump.ConstantMonochromaticPump(
                    frequency=pump_mode.frequency + params["pump_frequency_offset"],
                    power=params["pump_power"],
                )
            ]

            # re-center wavelengths bounds
            bounds = microspheres.sideband_bounds(
                pump_wavelength=pumps[0].wavelength,
                stokes_orders=params["stokes_orders"],
                antistokes_orders=params["antistokes_orders"],
                sideband_frequency=params["material"].modulation_frequency,
                bandwidth_frequency=params["group_bandwidth"],
            )

            # re-find modes
            print("redoing mode finding using new pump wavelength")
            modes = shared.find_modes(
                bounds, params["microsphere"], params["max_radial_mode_number"]
            )
            logman.info(f"Found {len(modes)} modes:")
            for mode in modes:
                print(mode)
            print()

            check_pump_mode = shared.find_mode_nearest_wavelength(
                modes, params["pump_wavelength"]
            )

            if check_pump_mode != pump_mode:
                print(f"original pump mode: {pump_mode}")
                print(f"new pump mode: {check_pump_mode}")
                raise ValueError("pump mode mismatch!")

        params["wavelength_bounds"] = bounds
        params["pump_wavelength"] = pumps[0].wavelength
        params["pump_frequency"] = pumps[0].frequency

        spec = params["spec_type"](
            params["component"],
            modes=modes,
            pumps=pumps,
            mode_initial_amplitudes={m: 1 for m in modes},
            mode_intrinsic_quality_factors={m: params["intrinsic_q"] for m in modes},
            mode_coupling_quality_factors={m: params["intrinsic_q"] for m in modes},
            **params,
        )

        sim = spec.to_sim()

        print(sim.info())

        return sim


@htmap.mapped(map_options=htmap.MapOptions(custom_options={"is_resumable": "true"}))
def run(sim):
    with si.utils.LogManager("modulation", "simulacra") as logman:
        sim_path = Path.cwd() / f"{sim.name}.sim"

        try:
            sim = si.Simulation.load(str(sim_path))
            logman.info(f"Recovered checkpoint from {sim_path}")
        except (FileNotFoundError, EOFError) as e:
            logman.info(f"Checkpoint not found at {sim_path}")

        sim.run(checkpoint_callback=htmap.checkpoint)

        print(sim.info())

        sim.polarization_sum_factors = None
        return sim


def main():
    shared.set_htmap_settings()

    tag = shared.ask_for_tag()

    if sys.argv[1] == "init":
        init_scan(tag)
    elif sys.argv[1] == "run":
        run_scan(tag)
    else:
        print("unrecognized command")
        sys.exit(1)


if __name__ == "__main__":
    main()
