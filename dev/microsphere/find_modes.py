import simulacra.units as u

from modulation import refraction
from modulation.resonator import microsphere

ms = microsphere.Microsphere(
    radius = 50 * u.um,
    index_of_refraction = refraction.ConstantIndex(1.45),
)

print(ms)
print(repr(ms))
print(ms.info())
print()

wavelength_bounds = [
    microsphere.WavelengthBound(799 * u.nm, 801 * u.nm),
]

modes = microsphere.find_modes(
    wavelength_bounds = wavelength_bounds,
    microsphere = ms,
)
print(len(modes))
for mode in modes:
    print(mode)
