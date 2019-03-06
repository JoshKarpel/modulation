import pytest

import hypothesis as hyp
import hypothesis.strategies as st
import hypothesis.extra.numpy as stnp

import numpy as np

import simulacra.units as u

from modulation.raman import pump


@pytest.mark.parametrize(
    "pump_type", [pump.RectangularMonochromaticPump, pump.ConstantMonochromaticPump]
)
def test_from_wavelength(pump_type):
    wavelength = 1064 * u.nm

    pump = pump_type.from_wavelength(wavelength=wavelength, power=0)

    assert pump.wavelength == wavelength


@pytest.mark.parametrize(
    "pump_type", [pump.RectangularMonochromaticPump, pump.ConstantMonochromaticPump]
)
def test_from_omega(pump_type):
    omega = u.twopi * 375 * u.THz

    pump = pump_type.from_omega(omega=omega, power=0)

    assert pump.omega == omega


@pytest.mark.parametrize(
    "pump_type", [pump.RectangularMonochromaticPump, pump.ConstantMonochromaticPump]
)
@hyp.given(power=st.floats(min_value=0))
def test_power_can_be_non_negative(pump_type, power):
    pump_type(frequency=0, power=power)


@pytest.mark.parametrize(
    "pump_type", [pump.RectangularMonochromaticPump, pump.ConstantMonochromaticPump]
)
@hyp.given(power=st.floats(max_value=0))
def test_power_cannot_be_negative(pump_type, power):
    hyp.assume(power < 0)
    with pytest.raises(ValueError):
        pump_type(frequency=0, power=power)


@hyp.given(
    time=st.one_of(
        st.floats(allow_infinity=True, allow_nan=False),
        stnp.arrays(dtype=np.float64, shape=(st.integers(min_value=0))),
    ),
    power=st.floats(min_value=0, allow_infinity=False, allow_nan=False),
)
def test_constant_monochromatic_pump_is_always_power(time, power):
    p = pump.ConstantMonochromaticPump(frequency=0, power=power)

    assert np.all(p.get_power(time) == power)


@hyp.given(
    data=st.data(),
    power=st.floats(min_value=0, allow_infinity=False, allow_nan=False),
    start_time=st.floats(allow_infinity=True, allow_nan=False),
    end_time=st.floats(allow_infinity=True, allow_nan=False),
)
def test_rectangular_monochromatic_pump_inside_bounds(
    data, power, start_time, end_time
):
    p = pump.RectangularMonochromaticPump(
        frequency=0, power=power, start_time=start_time, end_time=end_time
    )
    hyp.assume(p.start_time <= p.end_time)
    time = data.draw(st.floats(min_value=p.start_time, max_value=p.end_time))

    assert p.get_power(time) == power


@hyp.given(
    data=st.data(),
    power=st.floats(min_value=0, allow_infinity=False, allow_nan=False),
    start_time=st.one_of(st.none(), st.floats(allow_infinity=True, allow_nan=False)),
    end_time=st.one_of(st.none(), st.floats(allow_infinity=True, allow_nan=False)),
)
def test_rectangular_monochromatic_pump_outside_bounds(
    data, power, start_time, end_time
):
    p = pump.RectangularMonochromaticPump(
        frequency=0, power=power, start_time=start_time, end_time=end_time
    )
    hyp.assume(p.start_time <= p.end_time)
    time = data.draw(
        st.one_of(st.floats(min_value=p.end_time), st.floats(max_value=p.start_time))
    )
    hyp.assume(time not in (p.start_time, p.end_time))

    assert p.get_power(time) == 0
