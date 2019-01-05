from pathlib import Path
import random
import sys

import htmap

import simulacra as si
import simulacra.units as u

import modulation

# CLI

from halo import Halo
from spinners import Spinners

CLI_CONTEXT_SETTINGS = dict(help_option_names = ['-h', '--help'])

SPINNERS = list(name for name in Spinners.__members__ if 'dots' in name)


def make_spinner(*args, **kwargs):
    return Halo(
        *args,
        spinner = random.choice(SPINNERS),
        stream = sys.stderr,
        **kwargs,
    )


# SHARED QUESTIONS

def ask_map_id():
    return si.cluster.ask_for_input('Map ID?')


def ask_spec_type():
    return si.cluster.ask_for_choices(
        'SRS or FWM?',
        choices = {
            'SRS': modulation.raman.StimulatedRamanScatteringSpecification,
            'FWM': modulation.raman.FourWaveMixingSpecification,
        },
    )


def ask_material():
    choices = {k: k for k in modulation.raman.material.MATERIAL_DATA.keys()}
    choices['CUSTOM'] = 'CUSTOM'
    choice = si.cluster.ask_for_choices(
        'What Raman material to use?',
        choices = choices,
        default = 'silica',
    )

    if choice == 'CUSTOM':
        return modulation.raman.RamanMaterial(
            modulation_omega = si.cluster.ask_for_eval('Modulation Omega?'),
            coupling_prefactor = si.cluster.ask_for_eval('Coupling Prefactor?'),
            raman_linewidth = si.cluster.ask_for_eval('Raman Linewidth?'),
            number_density = si.cluster.ask_for_eval('Number Density?'),
        )

    return modulation.raman.RamanMaterial(**modulation.raman.material.MATERIAL_DATA[choice])


def ask_time_final():
    return u.usec * si.cluster.ask_for_input(
        'Final time (in us)?',
        default = 1,
        cast_to = float,
    )


def ask_time_step():
    return u.nsec * si.cluster.ask_for_input(
        'Time step (in ns)?',
        default = 1,
        cast_to = float,
    )


# MAP CREATION

@htmap.mapped
def _run(spec):
    sim_path = Path.cwd() / f'{spec.file_name}.sim'

    try:
        sim = si.Simulation.load(sim_path)
        print(f'Recovered checkpoint from {sim_path}')
    except (FileNotFoundError, EOFError):
        sim = spec.to_sim()
        print('No checkpoint found')

    print(sim.info())

    sim.run()

    print(sim.info())

    return sim


def _set_htmap_settings():
    htmap.settings['DOCKER.IMAGE'] = f'maventree/modulation:{si.cluster.ask_for_input("Docker image version?")}'


def _ask_about_map_options() -> (dict, dict):
    opts = {
        'request_memory': si.cluster.ask_for_input('Memory?', default = '128MB'),
        'request_disk': si.cluster.ask_for_input("Disk?", default = '10GB'),
    }
    custom_opts = {
        'wantflocking': str(si.cluster.ask_for_bool('Want flocking?', default = True)).lower(),
        'wantglidein': str(si.cluster.ask_for_bool('Want gliding?', default = True)).lower(),
    }

    return opts, custom_opts


def create_map(map_id: str, specs) -> htmap.Map:
    _set_htmap_settings()
    opts, custom = _ask_about_map_options()

    return _run.map(
        map_id,
        specs,
        map_options = htmap.MapOptions(
            **opts,
            custom_options = custom,
        ),
    )
