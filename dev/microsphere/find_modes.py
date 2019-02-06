import simulacra.units as u

from modulation import refraction
from modulation.resonators import microspheres

ms = microspheres.Microsphere(
    radius = 50 * u.um,
    index_of_refraction = refraction.ConstantIndex(1.45),
)

print(ms)
print(repr(ms))
print(ms.info())
print()

wavelength_bounds = [
    microspheres.WavelengthBound(799 * u.nm, 801 * u.nm),
]

modes = microspheres.find_mode_locations(
    wavelength_bounds = wavelength_bounds,
    microsphere = ms,
)
print(len(modes))
for mode in modes:
    print(mode)
