from pathlib import Path

import simulacra as si
import simulacra.units as u

from modulation import raman, refraction
from modulation.resonators import microspheres, mock

THIS_FILE = Path(__file__)
OUT_DIR = THIS_FILE.parent / 'out' / THIS_FILE.stem
SIM_LIB = OUT_DIR / 'SIMLIB'

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 6,
)

if __name__ == '__main__':
    mode = mock.MockMode(
        label = rf'q=0',
        omega = u.twopi * 250 * u.THz,
        index_of_refraction = 1.45,
        mode_volume_inside_resonator = 1e-20,
    )

    t_final = 10 * u.usec

    spec = raman.StimulatedRamanScatteringSpecification(
        'foobar',
        time_final = t_final,
        time_step = .1 * u.nsec,
        modes = [mode],
        mode_initial_amplitudes = {mode: 0},
        mode_coupling_quality_factors = {mode: 1e8},
        mode_intrinsic_quality_factors = {mode: 1e8},
        mode_pumps = {mode: raman.pump.ConstantPump(100 * u.uW)},
        mode_volume_integrator = mock.MockVolumeIntegrator(volume_integral_result = 1e-25),
        material = raman.RamanMaterial.from_database('silica'),
        store_mode_amplitudes_vs_time = True,
        # lookback = raman.Lookback(lookback_time = t_final / 10),
    )

    sim = spec.to_sim()

    sim.run(progress_bar = True)
    print(sim.running_time)

    # sim.plot.mode_magnitudes_vs_time(**PLOT_KWARGS)
    # #
    # # print(sim.lookback)
    # #
    # print('mean', sim.lookback.mean)
    # print('max', sim.lookback.max)
    # print('min', sim.lookback.min)
    # print('std', sim.lookback.std)
    # print('ptp', sim.lookback.ptp)
