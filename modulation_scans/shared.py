import functools
from pathlib import Path
import random
import sys

import htmap

import numpy as np

import simulacra as si
import simulacra.units as u

import modulation
from modulation.resonators import microspheres
from modulation.raman import AUTO_CUTOFF

# CLI

from halo import Halo
from spinners import Spinners

CLI_CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])

SPINNERS = list(name for name in Spinners.__members__ if "dots" in name)


def make_spinner(*args, **kwargs):
    return Halo(*args, spinner=random.choice(SPINNERS), stream=sys.stderr, **kwargs)


# SHARED QUESTIONS


def ask_for_tag():
    tag = si.ask_for_input("Map Tag?", default=None)
    if tag is None:
        raise ValueError("tag cannot be None")
    return tag


def ask_spec_type():
    return si.ask_for_choices(
        "SRS or FWM?",
        choices={
            "SRS": modulation.raman.StimulatedRamanScatteringSpecification,
            "FWM": modulation.raman.FourWaveMixingSpecification,
        },
        default="FWM",
    )


def ask_material():
    choices = {k: k for k in modulation.raman.material.MATERIAL_DATA.keys()}
    choice = si.ask_for_choices(
        "What Raman material to use?", choices=choices, default="silica"
    )

    return modulation.raman.RamanMaterial.from_name(choice)


def ask_fiber_parameters(parameters):
    parameters.append(
        si.Parameter(
            "fiber_taper_radius",
            value=u.um
            * si.ask_for_input("Fiber Taper Radius (in um)?", default=1, callback=int),
        )
    )


def ask_sideband_parameters(parameters, material):
    pump_stokes_orders = si.Parameter(
        "pump_stokes_orders",
        si.ask_for_input("Number of Stokes Orders for Pump?", callback=int, default=1),
    )
    pump_antistokes_orders = si.Parameter(
        "pump_antistokes_orders",
        si.ask_for_input(
            "Number of Anti-Stokes Orders for Pump?", callback=int, default=0
        ),
    )
    mixing_stokes_orders = si.Parameter(
        "mixing_stokes_orders",
        si.ask_for_input(
            "Number of Stokes Orders for Mixing?",
            callback=int,
            default=pump_stokes_orders.value,
        ),
    )
    mixing_antistokes_orders = si.Parameter(
        "mixing_antistokes_orders",
        si.ask_for_input(
            "Number of Anti-Stokes Orders for Mixing?",
            callback=int,
            default=pump_antistokes_orders.value,
        ),
    )
    parameters.extend(
        [
            pump_stokes_orders,
            pump_antistokes_orders,
            mixing_stokes_orders,
            mixing_antistokes_orders,
            si.Parameter(
                "group_bandwidth",
                (material.raman_linewidth / u.twopi)
                * si.ask_for_input(
                    "Mode Group Bandwidth (in Raman Linewidths)",
                    callback=float,
                    default=0.1,
                ),
            ),
            si.Parameter(
                "max_radial_mode_number",
                si.ask_for_input(
                    "Maximum Radial Mode Number?", callback=int, default=5
                ),
            ),
        ]
    )


def ask_laser_parameters(name, parameters):
    selection_method = si.ask_for_choices(
        f"{name.title()} Wavelength Selection Method?",
        choices={"raw": "raw", "offset": "offset", "symmetric": "symmetric"},
        default="raw",
    )

    if selection_method == "raw":
        parameters.append(
            si.Parameter(
                f"launched_{name}_wavelength",
                u.nm
                * np.array(
                    si.ask_for_eval(
                        f"{name.title()} laser wavelength (in nm)?", default="[1064]"
                    )
                ),
                expandable=True,
            )
        )
    elif selection_method == "offset":
        parameters.extend(
            [
                si.Parameter(
                    f"launched_{name}_wavelength",
                    u.nm
                    * si.ask_for_input(
                        f"Launched {name.title()} wavelength (in nm)?",
                        default=1064,
                        callback=float,
                    ),
                ),
                si.Parameter(
                    f"launched_{name}_detuning",
                    u.MHz
                    * np.array(
                        si.ask_for_eval(
                            f"Launched {name.title()} detunings (in MHz)", default="[0]"
                        )
                    ),
                    expandable=True,
                ),
            ]
        )
    elif selection_method == "symmetric":
        pump_wavelength = u.nm * si.ask_for_input(
            f"{name.title()} laser wavelength (in nm)?", default=1064, callback=float
        )
        pump_detunings_raw = u.MHz * np.array(
            si.ask_for_eval(
                f"Launched {name.title()} detunings (in MHz)", default="[0]"
            )
        )
        frequency_offsets_abs = np.array(sorted(set(np.abs(pump_detunings_raw))))
        if frequency_offsets_abs[0] != 0:
            frequency_offsets_abs = np.insert(frequency_offsets_abs, 0, 0)
        frequency_offsets = np.concatenate(
            (-frequency_offsets_abs[:0:-1], frequency_offsets_abs)
        )

        parameters.extend(
            [
                si.Parameter(f"launched_{name}_wavelength", pump_wavelength),
                si.Parameter(
                    f"launched_{name}_detuning", frequency_offsets, expandable=True
                ),
            ]
        )

    parameters.append(
        si.Parameter(
            f"launched_{name}_power",
            u.mW
            * np.array(
                si.ask_for_eval(
                    f"Launched {name.title()} power (in mW)?", default="[1]"
                )
            ),
            expandable=True,
        )
    )


def ask_time_final(parameters):
    parameters.append(
        si.Parameter(
            "time_final",
            sorted(
                u.usec
                * np.array(si.ask_for_eval("Final time (in us)?", default="[10]")),
                key=lambda x: -x,
            ),
            expandable=True,
        )
    )


def ask_time_step(parameters):
    parameters.append(
        si.Parameter(
            "time_step",
            sorted(
                u.psec * np.array(si.ask_for_eval("Time step (in ps)?", default="[10]"))
            ),
            expandable=True,
        )
    )


def ask_intrinsic_q(parameters):
    q = si.ask_for_eval("Mode Intrinsic Quality Factor?", default="[1e8]")
    parameters.append(si.Parameter("intrinsic_q", q, expandable=True))

    return q


def ask_four_mode_detuning_cutoff(parameters):
    cutoff = si.ask_for_eval("Four-mode Detuning Cutoff (in THz)?", default="[None]")
    cutoff = [c * u.THz if c is not None else AUTO_CUTOFF for c in cutoff]
    parameters.append(
        si.Parameter("four_mode_detuning_cutoff", cutoff, expandable=True)
    )


def ask_ignore_self_interaction(parameters):
    parameters.append(
        si.Parameter(
            "ignore_self_interaction",
            si.ask_for_eval("Ignore Self-Interaction Terms?", default="[False]"),
            expandable=True,
        )
    )


def ask_ignore_triplets(parameters):
    parameters.append(
        si.Parameter(
            "ignore_triplets",
            si.ask_for_eval("Ignore Triplet Terms?", default="[False]"),
            expandable=True,
        )
    )


def ask_ignore_doublets(parameters):
    parameters.append(
        si.Parameter(
            "ignore_doublets",
            si.ask_for_eval("Ignore Doublet Terms?", default="[False]"),
            expandable=True,
        )
    )


def ask_lookback_time():
    return u.nsec * si.ask_for_input(
        "Lookback time (in ns)?", default=10, callback=float
    )


def estimate_lookback_memory(lookback_time, time_step, num_modes):
    bytes_per_mode = lookback_time / time_step * 128

    total_bytes = bytes_per_mode * num_modes

    print(f"Lookback memory estimate (max): {si.utils.bytes_to_str(total_bytes)}")


def find_modes(wavelength_bounds, microsphere, max_radial_mode_number):
    mode_locations = microspheres.find_mode_locations(
        wavelength_bounds=wavelength_bounds,
        microsphere=microsphere,
        max_radial_mode_number=max_radial_mode_number,
    )

    modes = [
        microspheres.MicrosphereMode.from_mode_location(
            mode_location, m=mode_location.l
        )
        for mode_location in mode_locations
    ]

    return modes


def find_mode_nearest_wavelength(modes, wavelength: float):
    return sorted(modes, key=lambda m: abs(m.wavelength - wavelength))[0]


def find_mode_nearest_omega(modes, omega: float):
    return sorted(modes, key=lambda m: abs(m.omega - omega))[0]


# JUST A TEMPLATE
# FOR A CHECKPOINTING RUN FUNCTION
@htmap.mapped
def run(spec):
    sim_path = Path.cwd() / f"{spec.file_name}.sim"

    try:
        sim = si.Simulation.load(str(sim_path))
        print(f"Recovered checkpoint from {sim_path}")
    except (FileNotFoundError, EOFError):
        sim = spec.to_sim()
        print("No checkpoint found")

    print(sim.info())

    sim.run(checkpoint_callback=htmap.checkpoint)

    print(sim.info())

    return sim


def ask_htmap_settings():
    docker_image = si.ask_for_input("Docker image (repository:tag)?")
    htmap.settings["DOCKER.IMAGE"] = docker_image
    htmap.settings["SINGULARITY.IMAGE"] = f"docker://{docker_image}"

    delivery_method = si.ask_for_choices(
        "Use Docker or Singularity?",
        choices={"docker": "docker", "singularity": "singularity"},
        default="docker",
    )

    htmap.settings["DELIVERY_METHOD"] = delivery_method
    if delivery_method == "singularity":
        htmap.settings["MAP_OPTIONS.requirements"] = "OpSysMajorVer =?= 7"


def ask_map_options() -> (dict, dict):
    opts = {
        "request_memory": si.ask_for_input("Memory?", default="500MB"),
        "request_disk": si.ask_for_input("Disk?", default="1GB"),
        "max_idle": "100",
    }
    custom_opts = {
        "wantflocking": str(si.ask_for_bool("Want flocking?", default=False)).lower(),
        "wantglidein": str(si.ask_for_bool("Want gliding?", default=False)).lower(),
    }

    return opts, custom_opts


def get_bounds(params):
    pump_bounds = microspheres.sideband_bounds(
        center_wavelength=params["pump_wavelength"],
        stokes_orders=params["pump_stokes_orders"],
        antistokes_orders=params["pump_antistokes_orders"],
        sideband_frequency=params["material"].modulation_frequency,
        bandwidth_frequency=params["group_bandwidth"],
    )
    mixing_bounds = microspheres.sideband_bounds(
        center_wavelength=params["mixing_wavelength"],
        stokes_orders=params["mixing_stokes_orders"],
        antistokes_orders=params["mixing_antistokes_orders"],
        sideband_frequency=params["material"].modulation_frequency,
        bandwidth_frequency=params["group_bandwidth"],
    )

    return microspheres.merge_wavelength_bounds(pump_bounds + mixing_bounds)
