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


def create_scan(tag):
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
                (material.raman_linewidth / u.twopi)
                * si.cluster.ask_for_input(
                    "Mode Group Bandwidth (in Raman Linewidths)",
                    cast_to=float,
                    default=0.1,
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

    get_laser_parameters("pump", parameters)
    get_laser_parameters("mixing", parameters)

    # sort longest times and shortest time steps into earlier components
    # to get them running first
    parameters.extend(
        [
            si.cluster.Parameter(
                "time_final",
                sorted(
                    u.usec
                    * np.array(
                        si.cluster.ask_for_eval("Final time (in us)?", default="[10]")
                    ),
                    key=lambda x: -x,
                ),
                expandable=True,
            ),
            si.cluster.Parameter(
                "time_step",
                sorted(
                    u.psec
                    * np.array(
                        si.cluster.ask_for_eval("Time step (in ps)?", default="[10]")
                    )
                ),
                expandable=True,
            ),
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
    print("Checking memory and runtime...")

    p = final_parameters[0]

    bounds = get_bounds(p)
    modes = shared.find_modes(bounds, p["microsphere"], p["max_radial_mode_number"])

    print(f"Approximate number of modes in each simulation: {len(modes)}")

    if p["spec_type"] is modulation.raman.StimulatedRamanScatteringSpecification:
        approximate_psf_time = comb(len(modes), 2, repetition=True) * 0.05
        psf_storage = (len(modes) ** 2) * 16  # bytes
    elif p["spec_type"] is modulation.raman.FourWaveMixingSpecification:
        approximate_psf_time = comb(len(modes), 4, repetition=True) * 0.2
        psf_storage = (len(modes) ** 4) * 16  # bytes
    approximate_psf_time = datetime.timedelta(seconds=int(approximate_psf_time))
    print(
        f"Approximate time to calculate polarization sum factors: {approximate_psf_time}"
    )

    psf_generation = 1.1 * mvi.r_points * mvi.theta_points * (128 / 8) * len(modes)
    print(
        f"Approximate memory requirements for polarization sum factor calculation: {si.utils.bytes_to_str(psf_generation)}"
    )

    print(
        f"Approximate memory requirements for polarization sum factor storage: {si.utils.bytes_to_str(psf_storage)}"
    )

    shortest_time_step = min(p["time_step"] for p in final_parameters)
    lookback_mem = len(modes) * shortest_time_step * (128 / 8)
    print(
        f"Approximate memory requirements for largest lookback: {si.utils.bytes_to_str(lookback_mem)}"
    )

    print("Remember: add about 100 MB for everything else!")

    opts, custom = shared.ask_map_options()

    # CREATE MAP

    map = run.map(
        final_parameters,
        map_options=htmap.MapOptions(**opts, custom_options=custom),
        tag=tag,
    )

    print(f"Created map {map.tag}")

    return map


def get_laser_parameters(name, parameters):
    selection_method = si.cluster.ask_for_choices(
        f"{name.title()} Wavelength Selection Method?",
        choices={"raw": "raw", "offset": "offset", "symmetric": "symmetric"},
    )
    parameters.append(
        si.cluster.Parameter(f"{name}_selection_method", selection_method)
    )

    if selection_method == "raw":
        parameters.append(
            si.cluster.Parameter(
                f"{name}_wavelength",
                u.nm
                * np.array(
                    si.cluster.ask_for_eval(
                        f"{name.title()} laser wavelength (in nm)?", default="[1064]"
                    )
                ),
                expandable=True,
            )
        )
    elif selection_method == "offset":
        parameters.extend(
            [
                si.cluster.Parameter(
                    f"{name}_wavelength",
                    u.nm
                    * si.cluster.ask_for_input(
                        "Pump laser wavelength (in nm)?", default=1064, cast_to=float
                    ),
                ),
                si.cluster.Parameter(
                    f"{name}_frequency_offset",
                    u.GHz
                    * np.array(
                        si.cluster.ask_for_eval(
                            f"{name.title()} laser frequency offsets (in GHz)",
                            default="[0]",
                        )
                    ),
                    expandable=True,
                ),
            ]
        )
    elif selection_method == "symmetric":
        pump_wavelength = u.nm * si.cluster.ask_for_input(
            f"{name.title()} laser wavelength (in nm)?", default=1064, cast_to=float
        )
        pump_frequency_offsets_raw = u.GHz * np.array(
            si.cluster.ask_for_eval(
                f"{name.title()} laser frequency offsets (in GHz)", default="[0]"
            )
        )
        frequency_offsets_abs = np.array(
            sorted(set(np.abs(pump_frequency_offsets_raw)))
        )
        if frequency_offsets_abs[0] != 0:
            frequency_offsets_abs = np.insert(frequency_offsets_abs, 0, 0)
        frequency_offsets = np.concatenate(
            (-frequency_offsets_abs[:0:-1], frequency_offsets_abs)
        )

        parameters.extend(
            [
                si.cluster.Parameter(f"{name}_wavelength", pump_wavelength),
                si.cluster.Parameter(
                    f"{name}_frequency_offset", frequency_offsets, expandable=True
                ),
            ]
        )

    parameters.append(
        si.cluster.Parameter(
            f"{name}_power",
            u.uW
            * np.array(
                si.cluster.ask_for_eval(
                    f"Launched {name} power (in mW)?", default="[1]"
                )
            ),
            expandable=True,
        )
    )


def get_bounds(params):
    pump_bounds = microspheres.sideband_bounds(
        center_wavelength=params["pump_wavelength"],
        stokes_orders=params["stokes_orders"],
        antistokes_orders=params["antistokes_orders"],
        sideband_frequency=params["material"].modulation_frequency,
        bandwidth_frequency=params["group_bandwidth"],
    )
    mixing_bounds = microspheres.sideband_bounds(
        center_wavelength=params["mixing_wavelength"],
        stokes_orders=params["stokes_orders"],
        antistokes_orders=params["antistokes_orders"],
        sideband_frequency=params["material"].modulation_frequency,
        bandwidth_frequency=params["group_bandwidth"],
    )

    return microspheres.merge_wavelength_bounds(pump_bounds + mixing_bounds)


@htmap.mapped(map_options=htmap.MapOptions(custom_options={"is_resumable": "true"}))
def run(params):
    with si.utils.LogManager("modulation", "simulacra") as logger:
        sim_path = Path.cwd() / f"{params['component']}.sim"

        try:
            sim = si.Simulation.load(sim_path.as_posix())
            logger.info(f"Recovered checkpoint from {sim_path}")
        except (FileNotFoundError, EOFError) as e:
            logger.info(f"Checkpoint not found at {sim_path}")

            # note: bounds and modes may be replaced in the loop that follows
            # and params may be modified!
            # but it IS all deterministic...
            bounds = get_bounds(params)
            modes = shared.find_modes(
                bounds, params["microsphere"], params["max_radial_mode_number"]
            )
            pumps = []
            for name in ("pump", "mixing"):
                if params[f"{name}_selection_method"] == "raw":
                    logger.info(f"{name} is raw, keeping modes as-is")
                    pumps.append(
                        raman.pump.ConstantMonochromaticPump.from_wavelength(
                            wavelength=params[f"{name}_wavelength"],
                            power=params[f"{name}_power"],
                        )
                    )
                elif params[f"{name}_selection_method"] in ("offset", "symmetric"):
                    logger.info(
                        f"redoing mode finding based on real value of {name} pump"
                    )

                    target_mode = shared.find_mode_nearest_wavelength(
                        modes, params[f"{name}_wavelength"]
                    )
                    logger.info(f"{name} target mode is {target_mode}")

                    pump = raman.pump.ConstantMonochromaticPump(
                        frequency=target_mode.frequency
                        + params[f"{name}_frequency_offset"],
                        power=params[f"{name}_power"],
                    )
                    pumps.append(pump)
                    logger.info(f"{name} wavelength is {pump.wavelength}")
                    params[f"{name}_wavelength"] = pump.wavelength

                    # re-center wavelengths bounds
                    bounds = get_bounds(params)
                    modes = shared.find_modes(
                        bounds, params["microsphere"], params["max_radial_mode_number"]
                    )

                    refind_target_mode = shared.find_mode_nearest_wavelength(
                        modes, params[f"{name}_wavelength"]
                    )

                    if refind_target_mode != target_mode:
                        raise Exception(
                            f"disagreement about which mode is the target after refind!\noriginal: {target_mode}\nnew: {refind_target_mode}"
                        )

                    logger.info(f"{name} frequency is {pumps[0].frequency}")
                    params[f"{name}_frequency"] = pumps[0].frequency
                    params[f"{name}_mode"] = target_mode

            params["wavelength_bounds"] = bounds

            logger.info(f"Found {len(bounds)} bounds:")
            for bound in bounds:
                print(bound)
            print()

            logger.info(f"Found {len(modes)} modes:")
            for mode in modes:
                print(mode)
            print()

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

        sim.run(checkpoint_callback=htmap.checkpoint)

        print(sim.info())

        sim.polarization_sum_factors = None
        return sim


def main():
    shared.set_htmap_settings()

    tag = shared.ask_for_tag()

    create_scan(tag)


if __name__ == "__main__":
    main()
