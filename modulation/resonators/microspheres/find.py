import collections
import logging
from typing import List, Iterable, Union, Tuple, Optional, Iterator

import itertools
from dataclasses import dataclass

import numpy as np
import scipy.special as spc
import scipy.optimize as opt

import simulacra as si
import simulacra.units as u

from modulation.refraction import IndexOfRefraction

from . import mode
from ... import fmt

logger = logging.getLogger(__name__)

FloatOrArray = Union[float, np.ndarray]


class Microsphere:
    def __init__(
        self,
        *,
        radius: float,
        index_of_refraction: IndexOfRefraction,
        temperature: Optional[float] = None,
    ):
        self.radius = radius
        self.index_of_refraction = index_of_refraction
        self.temperature = temperature

    def __eq__(self, other):
        return all(
            (
                self.radius == other.radius,
                self.index_of_refraction == other.index_of_refraction,
                self.temperature == other.temperature,
            )
        )

    def __hash__(self):
        return hash(
            (self.__class__, self.radius, self.index_of_refraction, self.temperature)
        )

    def __str__(self):
        return f"{self.__class__.__name__}(radius = {self.radius / u.um:.4g} um, index_of_refraction = {self.index_of_refraction})"

    def __repr__(self):
        return f"{self.__class__.__name__}(radius = {self.radius}, index_of_refraction = {self.index_of_refraction})"

    def index(self, wavelength):
        n = self.index_of_refraction(wavelength, self.temperature)
        return n

    def info(self) -> si.Info:
        info = si.Info(header=self.__class__.__name__)
        info.add_field("Radius", fmt.quantity(self.radius, fmt.LENGTH_UNITS))
        info.add_info(self.index_of_refraction.info())
        return info


@dataclass(eq=True, frozen=True)
class MicrosphereModeLocation:
    wavelength: float
    l: int
    radial_mode_number: int
    microsphere: Microsphere
    polarization: "mode.MicrosphereModePolarization"

    @property
    def frequency(self):
        return u.c / self.wavelength

    @property
    def omega(self):
        return u.twopi * self.frequency

    def __str__(self):
        return f"{self.__class__.__name__}(λ = {self.wavelength / u.nm:.6f} nm, l = {self.l}, n = {self.radial_mode_number}, polarization = {self.polarization})"


def wavelength_to_frequency(wavelength: FloatOrArray) -> FloatOrArray:
    return u.c / wavelength


def frequency_to_wavelength(frequency: FloatOrArray) -> FloatOrArray:
    return u.c / frequency


def shift_wavelength_by_frequency(
    wavelength: FloatOrArray, frequency_shift: float
) -> FloatOrArray:
    frequency = u.c / wavelength
    shifted_frequency = frequency + frequency_shift

    return u.c / shifted_frequency


def shift_wavelength_by_omega(
    wavelength: FloatOrArray, omega_shift: float
) -> FloatOrArray:
    frequency = u.c / wavelength
    shifted_frequency = frequency + (omega_shift / u.twopi)

    return u.c / shifted_frequency


@dataclass(order=True, frozen=True)
class WavelengthBound:
    lower: float
    upper: float

    def __contains__(self, item):
        return self.lower <= item <= self.upper

    def __str__(self):
        return f"{self.__class__.__name__}(lower = {self.lower / u.nm:.3f} nm | {wavelength_to_frequency(self.lower) / u.THz:.3f} THz, upper = {self.upper / u.nm:.3f} nm | {wavelength_to_frequency(self.upper) / u.THz:.3f} THz)"

    @classmethod
    def from_frequencies(cls, lower: float, upper: float) -> "WavelengthBound":
        wavelengths = frequency_to_wavelength(lower), frequency_to_wavelength(upper)

        return cls(lower=min(wavelengths), upper=max(wavelengths))

    def is_disjoint(self, other: "WavelengthBound"):
        return self.lower > other.upper or self.upper < other.lower


def merge_wavelength_bounds(bounds: Iterable[WavelengthBound]) -> List[WavelengthBound]:
    bounds = sorted(bounds)

    if len(bounds) <= 1:
        return bounds

    merged_bounds = [bounds[0]]
    for bound in bounds[1:]:
        if not bound.is_disjoint(merged_bounds[-1]):
            merged_bounds[-1] = _merge_bounds_pair(merged_bounds[-1], bound)
        else:
            merged_bounds.append(bound)

    return merged_bounds


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def _merge_bounds_pair(a: WavelengthBound, b: WavelengthBound) -> WavelengthBound:
    return WavelengthBound(lower=min(a.lower, b.lower), upper=max(a.upper, b.upper))


def sideband_bounds(
    *,
    center_wavelength: float,
    stokes_orders: int,
    antistokes_orders: int = 0,
    sideband_frequency: float,
    bandwidth_frequency: float,
) -> Tuple[WavelengthBound, ...]:
    pump_frequency = wavelength_to_frequency(center_wavelength)
    center_frequencies = (
        pump_frequency - (order * sideband_frequency)
        for order in range(-antistokes_orders, stokes_orders + 1)
    )

    lower_upper = (
        (center - bandwidth_frequency, center + bandwidth_frequency)
        for center in center_frequencies
    )

    bounds = [
        WavelengthBound.from_frequencies(lower, upper) for lower, upper in lower_upper
    ]

    return tuple(bounds)


def group_modes_by_sideband(modes, sidebands):
    sideband_to_modes = collections.defaultdict(set)
    for sideband in sidebands:
        for mode in modes:
            if mode.wavelength in sideband:
                sideband_to_modes[sideband].add(mode)

    return sideband_to_modes


def sideband_of_wavelength(wavelength, sidebands):
    for sideband in sidebands:
        if wavelength in sideband:
            return sideband


class LBound:
    __slots__ = ("min", "max")

    def __init__(self, min: int, max: int):
        self.min = int(min)
        self.max = int(max)

    def __contains__(self, item):
        return self.min <= item <= self.max

    def __repr__(self):
        return f"{self.__class__.__name__}(min = {self.min}, max = {self.max})"

    def __iter__(self) -> Iterator[int]:
        yield from range(self.min, self.max + 1)


def TE_P(wavelength: FloatOrArray, microsphere: Microsphere) -> FloatOrArray:
    return microsphere.index(wavelength)


def TM_P(wavelength: FloatOrArray, microsphere: Microsphere) -> FloatOrArray:
    return 1 / microsphere.index(wavelength)


P_SELECTOR = {
    mode.MicrosphereModePolarization.TRANSVERSE_ELECTRIC: TE_P,
    mode.MicrosphereModePolarization.TRANSVERSE_MAGNETIC: TM_P,
}


def P(
    wavelength: FloatOrArray,
    polarization: mode.MicrosphereModePolarization,
    microsphere: Microsphere,
) -> FloatOrArray:
    return P_SELECTOR[polarization](wavelength, microsphere)


def alpha(
    wavelength: FloatOrArray,
    polarization: mode.MicrosphereModePolarization,
    microsphere: Microsphere,
) -> FloatOrArray:
    n = microsphere.index(wavelength)
    p = P(wavelength, polarization, microsphere)
    return p / (n * np.sqrt((n ** 2) - 1))


def modal_equation(
    wavelength: FloatOrArray,
    l: int,
    polarization: mode.MicrosphereModePolarization,
    microsphere: Microsphere,
) -> FloatOrArray:
    """
    eq 3.70 in Balac2016

    Parameters
    ----------
    wavelength
        The free-space wavelength of the light.
    l
        The angular momentum number.
    polarization
        The mode polarization to consider.
    microsphere
        The microsphere to search for modes in.

    Returns
    -------
    val
        The value of the modal equation.
        The zeros of this equation are the free-space wavelengths of the resonator modes.
    """
    lp = l + 0.5
    lm = l - 0.5

    k_0_R = microsphere.radius * u.twopi / wavelength
    y = spc.yv(lm, k_0_R) / spc.yv(lp, k_0_R)

    k_R = microsphere.index(wavelength) * k_0_R
    p = P(wavelength=wavelength, polarization=polarization, microsphere=microsphere)
    j = p * spc.jv(lm, k_R) / spc.jv(lp, k_R)

    return y - j - (l * ((1 / k_0_R) - (p / k_R)))


def _l_bounds_from_wavelength(
    wavelength: float,
    microsphere: Microsphere,
    polarization: mode.MicrosphereModePolarization,
) -> LBound:
    n = microsphere.index(wavelength)
    p = P(wavelength=wavelength, polarization=polarization, microsphere=microsphere)
    delta_P = wavelength * p / (u.twopi * n * np.sqrt((n ** 2) - 1))
    min = u.twopi * (microsphere.radius + delta_P) / wavelength
    max = n * min

    min -= 0.5
    max -= 0.5

    return LBound(np.floor(min), np.ceil(max))


def l_bound_from_wavelength_bound(
    wavelength_bound: WavelengthBound,
    microsphere: Microsphere,
    polarization: mode.MicrosphereModePolarization,
) -> LBound:
    bound_1 = _l_bounds_from_wavelength(
        wavelength=wavelength_bound.lower,
        microsphere=microsphere,
        polarization=polarization,
    )
    bound_2 = _l_bounds_from_wavelength(
        wavelength=wavelength_bound.upper,
        microsphere=microsphere,
        polarization=polarization,
    )

    return LBound(min=min(bound_1.min, bound_2.min), max=max(bound_1.max, bound_2.max))


def find_mode_locations(
    wavelength_bounds: Iterable[WavelengthBound],
    microsphere: Microsphere,
    max_radial_mode_number=5,
) -> List[MicrosphereModeLocation]:
    """
    Find the locations (free-space wavelengths and polarizations) of
    all of the modes in the microsphere in the given wavelength bounds.
    """
    wavelength_bounds = merge_wavelength_bounds(wavelength_bounds)
    mode_locations: List[MicrosphereModeLocation] = []

    for polarization in mode.MicrosphereModePolarization:
        for wavelength_bound in wavelength_bounds:
            l_bound = l_bound_from_wavelength_bound(
                wavelength_bound=wavelength_bound,
                microsphere=microsphere,
                polarization=polarization,
            )

            for l in l_bound:
                # find where the two kinds of Bessels in the denominator
                # have asymptotes, which is where those Bessels have zeros

                # these are the ones with the in-medium wavenumber (including
                # index of refraction), so we need to do a little extra work
                j_asymptotes, y_asymptotes = find_bessel_zeros(
                    order=l, num_zeros=max_radial_mode_number
                )
                wavelength_over_index = (u.twopi * microsphere.radius) / j_asymptotes
                good_asymptotes = opt.root(
                    lambda wavelength: (wavelength / microsphere.index(wavelength))
                    - wavelength_over_index,
                    1000 * u.nm * np.ones_like(wavelength_over_index),
                ).x

                # the others are just a straight conversion from Bessel argument
                # to wavelength
                bad_asmyptotes = microsphere.radius * u.twopi / y_asymptotes

                # extra arguments for the modal equation
                args = (l, polarization, microsphere)

                roots = []

                # the first root is a special case
                first_zero = opt.brentq(
                    modal_equation,
                    good_asymptotes[0],
                    good_asymptotes[0] * 1.5,
                    args=args,
                )
                roots.append(first_zero)

                # search between each pair of good asymptotes for a root
                for (top, bottom) in pairwise(good_asymptotes):
                    # there won't be a root if there's a "bad asymptote" between
                    # the good asymptotes
                    if any(bottom < w < top for w in bad_asmyptotes):
                        continue

                    # look for good bounds for the root finder
                    center = (top + bottom) / 2
                    upper = (center + top) / 2
                    lower = (center + bottom) / 2
                    while not (modal_equation(upper, *args) < 0):
                        upper = (upper + top) / 2
                    while not (modal_equation(lower, *args) > 0):
                        lower = (lower + bottom) / 2

                    root: float = opt.brentq(modal_equation, lower, upper, args=args)
                    roots.append(root)

                new_locations = (
                    MicrosphereModeLocation(
                        wavelength=root,
                        l=l,
                        radial_mode_number=n,
                        microsphere=microsphere,
                        polarization=polarization,
                    )
                    for n, root in enumerate(roots, start=1)
                    if root in wavelength_bound
                )
                mode_locations.extend(new_locations)

    return sorted(mode_locations, key=lambda x: x.wavelength)


def _z_from_zeta(z, xi):
    return (2 / 3) * ((-xi) ** (3 / 2)) - np.sqrt((z ** 2) - 1) + np.arccos(1 / z)


def _jacobian_of_z_from_zeta(z, xi):
    return np.diag(
        -(z / np.sqrt((z ** 2) - 1)) + 1 / (z ** 2 * np.sqrt(1 - (1 / z ** 2)))
    )


def find_bessel_zeros(order: int, num_zeros: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the zeros of high-order spherical Bessel functions
    using a uniformly-accurate asymptotic approximation.

    See https://dlmf.nist.gov/10.21.viii (Bessel function zeros) for details.
    That section is technically for the non-spherical Bessel functions, but the
    zeros of those functions are in the same place as the zeros of Bessel functions
    displayed by half-integer order.

    Parameters
    ----------
    order
        The order of the Bessel function.
    num_zeros
        The number of zeros to find.

    Returns
    -------
    j_zeros, y_zeros
        The first ``num_zeros`` zeros of the Bessel functions of the first and second kind with the given ``order``.
    """
    order_plus_half = order + 0.5
    airy_zeros, *_ = spc.ai_zeros(num_zeros)
    bairy_zeros, *_ = spc.bi_zeros(num_zeros)

    all_zeros = np.concatenate((airy_zeros, bairy_zeros))

    zeta = (order_plus_half ** (-2 / 3)) * all_zeros

    z = opt.root(
        _z_from_zeta,
        1.1 * np.ones_like(zeta),
        args=(zeta,),
        jac=_jacobian_of_z_from_zeta,
        method="hybr",
        options=dict(col_deriv=True),
    ).x

    h = (4 * zeta / (1 - (z ** 2))) ** (1 / 4)

    p = 1j * ((z ** 2) - 1) ** (-1 / 2)
    U = (1 / 24) * ((3 * p) - 5 * (p ** 3))
    B_0 = np.real(1j * ((-zeta) ** (-1 / 2)) * U)

    zeros = (order_plus_half * z) + ((z * (h ** 2) * B_0) / (2 * order_plus_half))

    return zeros[:num_zeros], zeros[num_zeros:]
