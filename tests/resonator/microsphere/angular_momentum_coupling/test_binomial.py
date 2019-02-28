import pytest

from whisper import racah


@pytest.mark.parametrize(
    "n, k, target",
    [
        (3, 2, 3),
        (9, 7, 36),
        (20, 15, 15504),
        (200, 30, 409_681_705_022_127_773_530_866_523_638_950_880),
        (205, 190, 21_474_679_368_627_581_088_160),
        (400, 7, 308_364_541_201_200),
    ],
)
def test_binomial_coefficients(n, k, target):
    assert racah.binomial_coefficient(n, k) == target


def test_binomial_with_k_negative_is_zero():
    assert racah.binomial_coefficient(10, -10) == 0


def test_binomial_coefficient_k_too_large_is_zero():
    assert racah.binomial_coefficient(10, 15) == 0
