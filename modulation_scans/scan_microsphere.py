import sys
import datetime

from tqdm import tqdm

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
    # QUESTIONS

    tag = shared.ask_for_tag()

    parameters = []

    spec_type = shared.ask_spec_type()

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
        ]
    )
    max_radial_mode_number = si.cluster.ask_for_input(
        "Maximum Radial Mode Number?", cast_to=int, default=5
    )
    parameters.append(
        si.cluster.Parameter("max_radial_mode_number", max_radial_mode_number)
    )

    parameters.extend(
        [
            si.cluster.Parameter(
                "pump_wavelength",
                u.nm
                * np.array(
                    si.cluster.ask_for_eval(
                        "Pump laser wavelength (in nm)?", default="[1064]"
                    )
                ),
                expandable=True,
            ),
            si.cluster.Parameter(
                "pump_power",
                u.uW
                * np.array(
                    si.cluster.ask_for_eval(
                        f"Launched power (in uW)?", default="np.linspace(0, 5000, 100)"
                    )
                ),
                expandable=True,
            ),
        ]
    )

    time_final = shared.ask_time_final(default=10)
    time_step = shared.ask_time_step(default=1)

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

    lookback_time = shared.ask_lookback_time(time_step, num_modes=None)

    # CREATE SPECS

    base_spec_kwargs = dict(
        time_final=time_final,
        time_step=time_step,
        material=material,
        mode_volume_integrator=mvi,
        checkpoints=True,
        checkpoint_every=datetime.timedelta(minutes=20),
        store_mode_amplitudes_vs_time=store_mode_amplitudes_vs_time,
        lookback=modulation.raman.Lookback(lookback_time=lookback_time),
    )

    print("Expanding parameters...")
    expanded_parameters = si.cluster.expand_parameters(parameters)

    print("Generating wavelength bounds...")
    wavelength_to_bounds = {
        params["pump_wavelength"]: microspheres.sideband_bounds(
            pump_wavelength=params["pump_wavelength"],
            stokes_orders=params["stokes_orders"],
            antistokes_orders=params["antistokes_orders"],
            sideband_frequency=material.modulation_frequency,
            bandwidth_frequency=params["group_bandwidth"],
        )
        for params in expanded_parameters
    }

    if len(wavelength_to_bounds) <= 10:
        print("Generating modes locally...")
        modes_by_wavelength = {
            wavelength: shared.find_modes(bounds, microsphere, max_radial_mode_number)
            for wavelength, bounds in wavelength_to_bounds.items()
        }
    elif len(wavelength_to_bounds) <= 100:
        print("Generating modes locally...")

        def modes(bounds):
            return shared.find_modes(bounds, microsphere, max_radial_mode_number)

        m = si.utils.multi_map(modes, list(wavelength_to_bounds.values()), processes=4)
        modes_by_wavelength = dict(zip(wavelength_to_bounds.keys(), m))
    else:
        print(
            "Need to generate a large number of different mode sets, mapping over the cluster..."
        )
        try:
            m = htmap.map(
                lambda bounds: shared.find_modes(
                    bounds, microsphere, max_radial_mode_number
                ),
                list(wavelength_to_bounds.values()),
                map_options=htmap.MapOptions(
                    custom_opts={"wantflocking": "true", "wantglidein": "true"}
                ),
            )
            m.wait(show_progress_bar=True)
            modes_by_wavelength = dict(zip(wavelength_to_bounds.keys(), m))
        finally:
            m.remove()

    print("Generating specifications...")
    specs = []
    for c, params in enumerate(tqdm(expanded_parameters)):
        modes = modes_by_wavelength[params["pump_wavelength"]]

        spec = spec_type(
            str(c),
            modes=modes,
            mode_initial_amplitudes={m: 1 for m in modes},
            pumps=[
                raman.pump.ConstantMonochromaticPump.from_wavelength(
                    wavelength=params["pump_wavelength"], power=params["pump_power"]
                )
            ],
            mode_intrinsic_quality_factors={m: params["intrinsic_q"] for m in modes},
            mode_coupling_quality_factors={m: params["intrinsic_q"] for m in modes},
            **base_spec_kwargs,
            **{f"_{k}": v for k, v in params.items()},
        )

        specs.append(spec)

    min_time_step = min(spec.time_step for spec in specs)
    max_num_modes = max(len(spec.modes) for spec in specs)

    shared.estimate_lookback_memory(lookback_time, min_time_step, max_num_modes)

    if not si.cluster.ask_for_bool(f"Launch a map with {len(specs)} simulations?"):
        sys.exit(1)

    # CREATE MAP
    shared.create_map(tag, specs)


if __name__ == "__main__":
    main()
