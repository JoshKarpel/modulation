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
from modulation.resonators import microspheres

from . import shared

import htmap


def create_scan(tag):
    parameters = []

    parameters.append(si.cluster.Parameter("spec_type", shared.ask_spec_type()))

    material = shared.ask_material()
    microsphere = microspheres.Microsphere(
        radius=u.um
        * si.cluster.ask_for_input(
            "Microsphere radius (in um)?", callback=float, default=50
        ),
        index_of_refraction=material.index_of_refraction,
    )
    mvi = microspheres.FixedGridSimpsonMicrosphereVolumeIntegrator(
        microsphere=microsphere
    )

    shared.ask_fiber_parameters(parameters)

    shared.ask_sideband_parameters(parameters, material)

    stripped_ask_laser_parameters("pump", parameters)
    stripped_ask_laser_parameters("mixing", parameters)

    shared.ask_time_final(parameters)
    shared.ask_time_step(parameters)

    shared.ask_intrinsic_q(parameters)

    store_mode_amplitudes_vs_time = si.cluster.ask_for_bool(
        "Store mode amplitudes vs time?", default="No"
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
        dict(**params, **extra_parameters) for params in expanded_parameters
    ]

    p = final_parameters[0]

    bounds = shared.get_bounds(p)
    modes = shared.find_modes(bounds, p["microsphere"], p["max_radial_mode_number"])

    modes_by_sideband = microspheres.group_modes_by_sideband(modes, bounds)
    pump_group = microspheres.sideband_of_wavelength(p["pump_wavelength"], bounds)
    mixing_group = microspheres.sideband_of_wavelength(p["mixing_wavelength"], bounds)

    pump_modes = modes_by_sideband[pump_group]
    mixing_modes = modes_by_sideband[mixing_group]

    specs = []
    for c, (params, pump_mode, mixing_mode) in enumerate(
        itertools.product(final_parameters, pump_modes, mixing_modes)
    ):
        pumps = [
            raman.pump.ConstantMonochromaticPump(
                frequency=pump_mode.frequency, power=params["pump_power"]
            ),
            raman.pump.ConstantMonochromaticPump(
                frequency=mixing_mode.frequency, power=params["mixing_power"]
            ),
        ]

        initial_amps = {
            m: 1
            if params["spec_type"] is raman.StimulatedRamanScatteringSpecification
            else 0
            for m in modes
        }

        intrinsic_q = params["intrinsic_q"]
        kwargs_for_coupling_q = dict(
            microsphere_index_of_refraction=params["microsphere"].index_of_refraction,
            fiber_index_of_refraction=params[
                "microsphere"
            ].index_of_refraction,  # assume fiber is made of same material as microsphere
            microsphere_radius=params["microsphere"].radius,
            fiber_taper_radius=params["fiber_taper_radius"],
        )
        separation = opt.brentq(
            lambda x: intrinsic_q
            - microspheres.coupling_quality_factor_for_tapered_fiber(
                separation=x,
                wavelength=pump_mode.wavelength,
                l=pump_mode.l,
                m=pump_mode.m,
                **kwargs_for_coupling_q,
            ),
            0,
            10 * u.um,
        )

        coupling_qs = {
            m: microspheres.coupling_quality_factor_for_tapered_fiber(
                separation=separation,
                wavelength=m.wavelength,
                l=m.l,
                m=m.m,
                **kwargs_for_coupling_q,
            )
            for m in modes
        }

        params["pump_wavelength"] = pump_mode.wavelength
        params["mixing_wavelength"] = mixing_mode.wavelength
        params["wavelength_bounds"] = bounds

        spec = params["spec_type"](
            name=c,
            modes=modes,
            pumps=pumps,
            mode_initial_amplitudes=initial_amps,
            mode_intrinsic_quality_factors={m: intrinsic_q for m in modes},
            mode_coupling_quality_factors=coupling_qs,
            pump_mode=pump_mode,
            mixing_mode=mixing_mode,
            **params,
        )
        specs.append(spec)

    print(f"This map will contain {len(specs)} components")

    # MEMORY AND RUNTIME CHECK
    print("Checking memory and runtime...")
    print(f"Number of modes in each simulation: {len(modes)}")

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
    lookback_mem = len(modes) * int(lookback_time / shortest_time_step) * (128 / 8)
    print(
        f"Approximate memory requirements for largest lookback: {si.utils.bytes_to_str(lookback_mem)}"
    )

    print("Remember: add about 100 MB for everything else!")

    opts, custom = shared.ask_map_options()

    # CREATE MAP

    map = run.map(
        specs, map_options=htmap.MapOptions(**opts, custom_options=custom), tag=tag
    )

    print(f"Created map {map.tag}")

    return map


def stripped_ask_laser_parameters(name, parameters):
    parameters.append(
        si.cluster.Parameter(
            f"{name}_wavelength",
            u.nm
            * si.cluster.ask_for_eval(
                f"{name.title()} laser wavelength (in nm)?", default="1064"
            ),
        )
    )
    parameters.append(
        si.cluster.Parameter(
            f"{name}_power",
            u.mW
            * np.array(
                si.cluster.ask_for_eval(
                    f"Launched {name} power (in mW)?", default="[1]"
                )
            ),
            expandable=True,
        )
    )


@htmap.mapped(map_options=htmap.MapOptions(custom_options={"is_resumable": "true"}))
def run(spec):
    with si.utils.LogManager("modulation", "simulacra") as logger:
        sim_path = Path.cwd() / f"{spec.name}.sim"

        try:
            sim = si.Simulation.load(sim_path.as_posix())
            logger.info(f"Recovered checkpoint from {sim_path}")
        except (FileNotFoundError, EOFError) as e:
            logger.info(f"Checkpoint not found at {sim_path}")
            sim = spec.to_sim()

        print(sim.info())

        sim.run(checkpoint_callback=htmap.checkpoint)

        print(sim.info())

        sim.polarization_sum_factors = None
        return sim


def main():
    shared.ask_htmap_settings()

    tag = shared.ask_for_tag()

    create_scan(tag)


if __name__ == "__main__":
    main()
