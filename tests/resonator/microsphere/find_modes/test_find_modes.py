import pytest

import simulacra.units as u

from modulation import refraction
from modulation.resonator import microsphere

ms = microsphere.Microsphere(
    radius = 50 * u.um,
    index_of_refraction = refraction.ConstantIndex(1.45),
)


def test_empty_bounds_gives_no_modes():
    wavelength_bounds = []

    modes = microsphere.find_modes(
        wavelength_bounds = wavelength_bounds,
        microsphere = ms,
    )

    assert len(modes) == 0


def test_for_regressions_in_number_of_modes():
    wavelength_bounds = [
        microsphere.WavelengthBound(799 * u.nm, 801 * u.nm),
    ]

    modes = microsphere.find_modes(
        wavelength_bounds = wavelength_bounds,
        microsphere = ms,
    )

    assert len(modes) == 78
