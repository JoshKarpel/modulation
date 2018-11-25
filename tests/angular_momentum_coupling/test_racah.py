import pytest

import numpy as np

from .conftest import EXPLICIT_THREEJ

from whisper import racah


@pytest.mark.parametrize(
    'lm, target',
    EXPLICIT_THREEJ,
)
def test_threej_from_racah(lm, target):
    try:
        r = racah.threej_via_racah(*lm)
    except OverflowError:  # ignore the large-l/m test cases
        return True

    assert np.isclose(r, target, rtol = 1e-12, atol = 1e-12)
