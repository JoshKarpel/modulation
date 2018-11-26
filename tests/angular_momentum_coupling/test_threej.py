import pytest

import numpy as np

from .conftest import EXPLICIT_THREEJ

from whisper import threej, racah


@pytest.mark.parametrize(
    'lm, target',
    EXPLICIT_THREEJ
)
def test_explicit_threej(lm, target):
    # print('lm', lm)
    result = threej.threej(*lm)
    # print('result', result)
    assert np.isclose(
        result,
        target,
        rtol = 1e-12,
        atol = 1e-15,
    )


def valid_lm_builder(l_min, l_max):
    """Construct symbols that are not trivially zero."""
    yield from (
        (l1, l2, l3, -(m2 + m3), m2, m3)
        for l2 in range(l_min, l_max + 1)
        for l3 in range(l_min, l_max + 1)
        for m2 in range(-l2, l2 + 1)
        for m3 in range(-l3, l3 + 1)
        for l1 in range(max(abs(l2 - l3), abs(m2 + m3)), l2 + l3 + 1)
    )


def any_lm_builder(l_max):
    """Construct symbols that might be trivially zero."""
    yield from (
        (l1, l2, l3, m1, m2, m3)
        for l1 in range(0, l_max + 1)
        for l2 in range(0, l_max + 1)
        for l3 in range(0, l_max + 1)
        for m1 in range(-l_max, l_max + 1)
        for m2 in range(-l_max, l_max + 1)
        for m3 in range(-l_max, l_max + 1)
    )


@pytest.mark.parametrize(
    'lm',
    valid_lm_builder(0, 8),
)
def test_small_threej_against_threej_via_racah_valid_integers(lm):
    """This test only makes sense for small enough l,m that the Racah formula can evaluate it!"""
    # print('lm', lm)
    result = threej.threej(*lm)
    # print('result', result)
    expected = racah.threej_via_racah(*lm)
    # print('expected', expected)
    assert np.isclose(
        result,
        expected,
        rtol = 1e-12,
        atol = 1e-15,
    )


@pytest.mark.parametrize(
    'lm',
    any_lm_builder(4),
)
def test_small_threej_against_threej_via_racah_any_integers(lm):
    """This test only makes sense for small enough l,m that the Racah formula can evaluate it!"""
    # print('lm', lm)
    result = threej.threej(*lm)
    # print('result', result)
    expected = racah.threej_via_racah(*lm)
    # print('expected', expected)
    assert np.isclose(
        result,
        expected,
        rtol = 1e-12,
        atol = 1e-15,
    )
