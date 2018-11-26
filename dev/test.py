import time

import simulacra.units as u

from modulation import raman, refraction
from modulation.resonators import microsphere, mock

mode = mock.MockMode(
    omega = u.twopi * 250 * u.THz,
    index_of_refraction = 1.45,
    mode_volume_inside_resonator = 1e-20,
)
d = {mode: 0}

print(mode.info())
time.sleep(.1)

spec = raman.StimulatedRamanScatteringSpecification(
    'foobar',
    modes = [mode],
    mode_initial_amplitudes = {mode: 0},
    mode_coupling_quality_factors = {mode: 1e8},
    mode_intrinsic_quality_factors = {mode: 1e8},
    mode_pump_rates = {mode: raman.pumps.ConstantPump(0)},
    mode_volume_integrator = mock.MockVolumeIntegrator(volume_integral_result = 1e-26),
    material = raman.RamanMaterial.silicon(),
)
time.sleep(.1)
print(spec.info())

time.sleep(.1)
print()
sim = spec.to_sim()
time.sleep(.1)
print(sim.info())

sim.run()
