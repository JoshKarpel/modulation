import functools
from pathlib import Path
import random
import sys

import htmap

import simulacra as si
import simulacra.units as u

import modulation
from modulation.resonators import microspheres

# CLI

from halo import Halo
from spinners import Spinners

CLI_CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])

SPINNERS = list(name for name in Spinners.__members__ if "dots" in name)


def make_spinner(*args, **kwargs):
    return Halo(*args, spinner=random.choice(SPINNERS), stream=sys.stderr, **kwargs)


# SHARED QUESTIONS


def ask_for_tag():
    tag = si.cluster.ask_for_input("Map Tag?", default=None)
    if tag is None:
        raise ValueError("tag cannot be None")
    return tag


def ask_spec_type():
    return si.cluster.ask_for_choices(
        "SRS or FWM?",
        choices={
            "SRS": modulation.raman.StimulatedRamanScatteringSpecification,
            "FWM": modulation.raman.FourWaveMixingSpecification,
        },
    )


def ask_material():
    choices = {k: k for k in modulation.raman.material.MATERIAL_DATA.keys()}
    choices["CUSTOM"] = "CUSTOM"
    choice = si.cluster.ask_for_choices(
        "What Raman material to use?", choices=choices, default="silica"
    )

    if choice == "CUSTOM":
        return modulation.raman.RamanMaterial(
            modulation_omega=si.cluster.ask_for_eval("Modulation Omega?"),
            coupling_prefactor=si.cluster.ask_for_eval("Coupling Prefactor?"),
            raman_linewidth=si.cluster.ask_for_eval("Raman Linewidth?"),
            number_density=si.cluster.ask_for_eval("Number Density?"),
        )

    return modulation.raman.RamanMaterial.from_name(choice)


def ask_time_final(default=1):
    return u.usec * si.cluster.ask_for_input(
        "Final time (in us)?", default=default, cast_to=float
    )


def ask_time_step(default=1):
    return u.psec * si.cluster.ask_for_input(
        "Time step (in ps)?", default=default, cast_to=float
    )


def ask_lookback_time(time_step, num_modes=None):
    lookback_time = u.nsec * si.cluster.ask_for_input(
        "Lookback time (in ns)?", default=10, cast_to=float
    )

    bytes_per_mode = (lookback_time / time_step) * 128

    msg = f"Lookback will use ~{si.utils.bytes_to_str(bytes_per_mode)} per mode"
    if num_modes is not None:
        total_bytes = bytes_per_mode * num_modes

        msg += f" x {num_modes} modes = ~{si.utils.bytes_to_str(total_bytes)}"

    print(msg)

    return lookback_time


def estimate_lookback_memory(lookback_time, time_step, num_modes):
    bytes_per_mode = lookback_time / time_step * 128

    total_bytes = bytes_per_mode * num_modes

    print(f"Lookback memory estimate (max): {si.utils.bytes_to_str(total_bytes)}")


@functools.lru_cache(maxsize=None)
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


# MAP CREATION


@htmap.mapped
def _run(spec):
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


def _set_htmap_settings():
    htmap.settings[
        "DOCKER.IMAGE"
    ] = f'maventree/modulation:{si.cluster.ask_for_input("Docker image version?")}'


def _ask_about_map_options() -> (dict, dict):
    opts = {
        "request_memory": si.cluster.ask_for_input("Memory?", default="250MB"),
        "request_disk": si.cluster.ask_for_input("Disk?", default="500MB"),
    }
    custom_opts = {
        "wantflocking": str(
            si.cluster.ask_for_bool("Want flocking?", default=True)
        ).lower(),
        "wantglidein": str(
            si.cluster.ask_for_bool("Want gliding?", default=True)
        ).lower(),
        "is_resumable": "true",
    }

    return opts, custom_opts


def create_map(tag: str, specs) -> htmap.Map:
    _set_htmap_settings()
    opts, custom = _ask_about_map_options()

    map = _run.map(
        specs, map_options=htmap.MapOptions(**opts, custom_options=custom), tag=tag
    )

    print(f"Created map {map.tag}")

    return map
