import pytest

import simulacra.units as u

from modulation import refraction
from modulation.resonators.microspheres import WavelengthBound, merge_wavelength_bounds


@pytest.mark.parametrize(
    'a, b, answer',
    [
        (WavelengthBound(0.1, 1), WavelengthBound(0.1, 1), False),
        (WavelengthBound(2, 3), WavelengthBound(0.1, 1), True),
        (WavelengthBound(0.1, 1), WavelengthBound(1, 2), False),
        (WavelengthBound(0.1, 2), WavelengthBound(1, 3), False),
        (WavelengthBound(0.1, 2), WavelengthBound(4, 5), True),
        (WavelengthBound(1, 3), WavelengthBound(2, 3), False),
    ]
)
def test_is_disjoint(a, b, answer):
    assert a.is_disjoint(b) is answer


def test_sorting_disjoint_bounds():
    bounds = [
        WavelengthBound(2, 3),
        WavelengthBound(0.1, 1),
    ]

    assert sorted(bounds) == [
        WavelengthBound(0.1, 1),
        WavelengthBound(2, 3),
    ]


def test_sorting_overlapping_bounds():
    bounds = [
        WavelengthBound(2, 3),
        WavelengthBound(1, 2),
    ]

    assert sorted(bounds) == [
        WavelengthBound(1, 2),
        WavelengthBound(2, 3),
    ]


def test_no_bounds():
    assert merge_wavelength_bounds([]) == []


def test_one_bound():
    assert merge_wavelength_bounds([WavelengthBound(1, 2)]) == [WavelengthBound(1, 2)]


def test_two_disjoint_bounds():
    bounds = [
        WavelengthBound(1, 2),
        WavelengthBound(4, 5),
    ]

    assert merge_wavelength_bounds(bounds) == bounds


def test_two_overlapping_bounds():
    bounds = [
        WavelengthBound(2, 3),
        WavelengthBound(1, 3),
    ]

    assert merge_wavelength_bounds(bounds) == [WavelengthBound(1, 3)]


def test_three_overlapping_bounds():
    bounds = [
        WavelengthBound(1, 2),
        WavelengthBound(1, 3),
        WavelengthBound(2, 4),
    ]

    assert merge_wavelength_bounds(bounds) == [WavelengthBound(1, 4)]


def test_four_overlapping_bounds():
    bounds = [
        WavelengthBound(1, 2),
        WavelengthBound(1, 3),
        WavelengthBound(2, 4),
        WavelengthBound(4, 5),
    ]

    assert merge_wavelength_bounds(bounds) == [WavelengthBound(1, 5)]


def test_two_overlapping_and_one_disjoint():
    bounds = [
        WavelengthBound(2, 3),
        WavelengthBound(5, 7),
        WavelengthBound(1, 3),
    ]

    assert merge_wavelength_bounds(bounds) == [
        WavelengthBound(1, 3),
        WavelengthBound(5, 7),
    ]


def test_two_sets_of_two_overlapping():
    bounds = [
        WavelengthBound(6, 8),
        WavelengthBound(2, 3),
        WavelengthBound(1, 3),
        WavelengthBound(5, 7),
    ]

    assert merge_wavelength_bounds(bounds) == [
        WavelengthBound(1, 3),
        WavelengthBound(5, 8),
    ]
