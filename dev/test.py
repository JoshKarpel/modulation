import simulacra.units as u

from modulation import raman, refraction
from modulation.resonators import microsphere, mock

spec = raman.StimulatedRamanScatteringSpecification(
    'foobar',
    coupling_prefactor = 1,
    modes = [],
    mode_initial_amplitudes = {},
    mode_coupling_quality_factors = {},
    mode_intrinsic_quality_factors = {},
    mode_pump_rates = {},
    modulation_omega = 100 * u.THz,
    gamma_b = 1 * u.THz,
    number_density = 5e22,
    mode_volume_integrator = mock.MockVolumeIntegrator(volume_integral_result = 1e-26),
)
print(spec.info())
print()
sim = spec.to_sim()
print(sim.info())
