import time

import simulacra.units as u

from modulation import raman, refraction
from modulation.resonators import microspheres, mock

mode = mock.MockMode(
    label=rf"q=0",
    omega=u.twopi * 250 * u.THz,
    index_of_refraction=1.45,
    mode_volume_inside_resonator=1e-20,
)

spec = raman.StimulatedRamanScatteringSpecification(
    "foobar",
    modes=[mode],
    mode_initial_amplitudes={mode: 0},
    mode_coupling_quality_factors={mode: 1e8},
    mode_intrinsic_quality_factors={mode: 1e8},
    mode_pumps={mode: raman.pump.ConstantMonochromaticPump(100 * u.uW)},
    mode_volume_integrator=mock.MockVolumeIntegrator(volume_integral_result=1e-25),
    material=raman.RamanMaterial.from_database("silica"),
    store_mode_amplitudes_vs_time=True,
)

sim = spec.to_sim()

print(sim.times)
print(list(sim.iter_times()))

sim.run()
