import sys
import datetime

from tqdm import tqdm

import numpy as np

import simulacra as si
import simulacra.units as u

import modulation
from modulation import raman
from modulation.resonators import microspheres

from . import shared


def main():
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
            si.cluster.Parameter(
                "max_radial_mode_number",
                si.cluster.ask_for_input(
                    "Maximum Radial Mode Number?", cast_to=int, default=5
                ),
            ),
        ]
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

    print("Generating specifications...")
    specs = []
    for c, params in enumerate(tqdm(si.cluster.expand_parameters(parameters))):
        wavelength_bounds = microspheres.sideband_bounds(
            pump_wavelength=params["pump_wavelength"],
            stokes_orders=params["stokes_orders"],
            antistokes_orders=params["antistokes_orders"],
            sideband_frequency=material.modulation_frequency,
            bandwidth_frequency=params["group_bandwidth"],
        )

        modes = shared.find_modes(
            wavelength_bounds, microsphere, params["max_radial_mode_number"]
        )

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
