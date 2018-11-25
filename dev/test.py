import time

import simulacra.units as u

from modulation import raman, refraction
from modulation.resonators import microsphere, mock

mode = mock.MockMode(
    omega = u.twopi * 250 * u.THz,
    index_of_refraction = 1.45,
)
d = {mode: 1}

print(mode.info())
time.sleep(.1)

spec = raman.StimulatedRamanScatteringSpecification(
    'foobar',
    coupling_prefactor = 1,
    modes = [mode],
    mode_initial_amplitudes = d,
    mode_coupling_quality_factors = d,
    mode_intrinsic_quality_factors = d,
    mode_pump_rates = {mode: raman.pumps.ConstantPump(0)},
    modulation_omega = 100 * u.THz,
    gamma_b = 1 * u.THz,
    number_density = 5e22,
    mode_volume_integrator = mock.MockVolumeIntegrator(volume_integral_result = 1e-26),
)
time.sleep(.1)
print(spec.info())

time.sleep(.1)
print()
sim = spec.to_sim()
time.sleep(.1)
print(sim.info())
