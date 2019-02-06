import logging

import itertools
import functools

import numpy as np

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize = None)
def threej(l1, l2, l3, m1, m2, m3):
    r = _look_for_threej_shortcuts(l1, l2, l3, m1, m2, m3)
    if r is not None:
        return r

    return _threej_l_recursion(l1, l2, l3, m1, m2, m3)


def _look_for_threej_shortcuts(l1, l2, l3, m1, m2, m3):
    if not m1 + m2 + m3 == 0:
        return 0

    l_sum = l1 + l2 + l3
    if l_sum % 1 != 0:
        return 0

    if m1 == m2 == m3 == 0 and l_sum % 2 != 0:
        return 0

    if any((abs(m1) > l1, abs(m2) > l2, abs(m3) > l3)):
        return 0

    for (l1, l2, l3, m1, m2, m3), sign in _generate_equivalent_threej_symbols(l1, l2, l3, m1, m2, m3):
        if not abs(l1 - l2) <= l3 <= l1 + l2:  # triangle relations
            return 0

        if l1 == l2 and m1 == -m2 and l3 == m3 == 0:  # shortcut
            return (1 if (l1 - m1) % 2 == 0 else -1) / np.sqrt((2 * l1) + 1)

    return None


def _zip_symbol_args(a, b, c):
    return tuple(itertools.chain.from_iterable(zip(a, b, c)))


def _generate_equivalent_threej_symbols(l1, l2, l3, m1, m2, m3):
    yield (l1, l2, l3, m1, m2, m3), 1

    first, second, third = (
        (l1, m1),
        (l2, m2),
        (l3, m3),
    )

    # even permutations
    yield _zip_symbol_args(third, first, second), 1
    yield _zip_symbol_args(second, third, first), 1

    # odd permutations
    sign = 1 if (l1 + l2 + l3) % 2 == 0 else -1
    yield _zip_symbol_args(first, third, second), sign
    yield _zip_symbol_args(second, first, third), sign
    yield _zip_symbol_args(third, second, first), sign


def _threej_l_recursion(l1, l2, l3, m1, m2, m3):
    family = _threej_l_recursion_family(l2, l3, m2, m3)

    return family[l1]


@functools.lru_cache(maxsize = None)
def _threej_l_recursion_family(l2, l3, m2, m3):
    l_min = int(max(abs(l2 - l3), abs(m2 + m3)))
    l_max = int(l2 + l3)

    l_rng = np.zeros(l_max + 1, dtype = np.float64)
    l_rng[l_min:] = np.arange(l_min, l_max + 1, step = 1)

    X = np.empty_like(l_rng)
    Y = np.empty_like(l_rng)
    Z = np.empty_like(l_rng)

    X[l_min:] = l_rng[l_min:] * A(l_rng[l_min:] + 1, l2, l3, m2, m3)
    Y[l_min:] = B(l_rng[l_min:], l2, l3, m2, m3)
    Z[l_min:] = (l_rng[l_min:] + 1) * A(l_rng[l_min:], l2, l3, m2, m3)

    if l2 == l3 and m2 == m3 == 0:  # implies all three m are zero, special case
        return l_equal_and_all_m_zero_family_shortcut(l2, l3, l_max, Z, X)
    elif m2 == m3 == 0:
        unnormalized_psi = generate_unnormalized_psi_for_all_m_zero(X, Y, Z, l_max, l_min)
    else:
        unnormalized_psi = generate_unnormalized_psi(X, Y, Z, l_max, l_min)

    normalized_psi = generate_normalized_psi(unnormalized_psi, l2, l3, m2, m3, l_max)

    return normalized_psi


def A(l, l2, l3, m2, m3):
    l_sq = l ** 2
    first = l_sq - ((l2 - l3) ** 2)
    second = ((l2 + l3 + 1) ** 2) - (l ** 2)
    third = l_sq - ((m2 + m3) ** 2)

    return np.sqrt(first * second * third)


def B(l, l2, l3, m2, m3):
    pre = ((2 * l) + 1)
    first = (m2 + m3) * ((l2 * (l2 + 1)) - (l3 * (l3 + 1)))
    second = (m2 - m3) * l * (l + 1)

    return pre * (first - second)


# these seems to be stable, even going up, because the three-term recursion collapses to two terms

def l_equal_and_all_m_zero_family_shortcut(l2, l3, l_max, Z, X):
    """These come out normalized because we can get the first symbol in closed-form."""
    psi = {0: threej(l2, l3, 0, 0, 0, 0)}  # will hit shortcut
    for l in range(2, l_max + 1, 2):
        psi[l] = -Z[l - 1] * psi[l - 2] / X[l - 1]

    # don't need to store the odd-l members because they'll be hit by shortcuts in threej

    return psi


def generate_unnormalized_psi_for_all_m_zero(X, Y, Z, l_max, l_min):
    psi = {l_min: 1}
    for l in range(l_min + 2, l_max + 1, 2):  # even
        psi[l] = -Z[l - 1] * psi[l - 2] / X[l - 1]
    for l in range(l_min + 1, l_max + 1, 2):  # odd
        psi[l] = 0

    return psi


def generate_unnormalized_psi(X, Y, Z, l_max, l_min):
    l_forwards = l_min
    s = {}
    if Y[l_min] != 0:  # region 1 exists
        s = {l_min: -X[l_min] / Y[l_min]}
        l_mid = (l_min + l_max) / 2
        while l_forwards < l_mid:
            l_forwards += 1
            num = -X[l_forwards]
            denom = Y[l_forwards] + (Z[l_forwards] * s[l_forwards - 1])

            if denom == 0:
                l_forwards -= 1
                break

            s[l_forwards] = num / denom
            if abs(s[l_forwards]) >= 1:
                break

    l_backwards = l_max
    r = {}
    if Y[l_max] != 0:
        r = {l_max: -Z[l_max] / Y[l_max]}
        while l_backwards > l_forwards:
            l_backwards -= 1

            num = -Z[l_backwards]
            denom = (Y[l_backwards] + (X[l_backwards] * r[l_backwards + 1]))

            if denom == 0:
                l_backwards += 1
                break

            r[l_backwards] = num / denom

            if abs(r[l_backwards]) >= 1:
                break

    if len(s) > 1:  # enough values in forward recursion to start there
        # plan: recurse Psi_minus up from l_forward to l_backward
        Psi_minus = {l_forwards: 1, l_forwards - 1: s[l_forwards - 1]}
        # back-fill
        for l in range(l_forwards - 2, l_min - 1, -1):
            Psi_minus[l] = Psi_minus[l + 1] * s[l]

        # fill forward to l_backwards
        for l in range(l_forwards + 1, l_backwards + 1):
            Psi_minus[l] = -((Y[l - 1] * Psi_minus[l - 1]) + (Z[l - 1] * Psi_minus[l - 2])) / X[l - 1]

        # match up
        ratio = Psi_minus[l_backwards]
        Psi_plus = {l: ratio * v for l, v in Psi_minus.items()}

        for l in range(l_backwards + 1, l_max + 1):
            Psi_plus[l] = Psi_plus[l - 1] * r[l]

        return Psi_plus
    elif len(r) > 1:  # not enough values in forward recursion, start with backward recursion instead
        # plan: recurse Psi_plus down from l_backward to l_forward
        Psi_plus = {l_backwards: 1, l_backwards + 1: r[l_backwards + 1]}

        # fill forward
        for l in range(l_backwards + 2, l_max + 1):
            Psi_plus[l] = Psi_plus[l - 1] * r[l]

        # fill back to l_forwards
        for l in range(l_backwards - 1, l_forwards - 1, -1):
            Psi_plus[l] = - ((X[l + 1] * Psi_plus[l + 2]) + (Y[l + 1] * Psi_plus[l + 1])) / Z[l + 1]

        # match up
        ratio = Psi_plus[l_forwards]
        Psi_minus = {l: ratio * v for l, v in Psi_plus.items()}

        # no need for last loop like in first version
        # because we only get here if l_forwards is actually l_min

        return Psi_minus
    else:  # not enough symbols in either => symbol family is totally classical
        # plan: do the three-term from bottom to top
        psi = {l_min: 1, l_min - 1: 0}  # symbol is 0 outside bounds
        for l in range(l_min + 1, l_max + 1):
            psi[l] = -((Y[l - 1] * psi[l - 1]) + (Z[l - 1] * psi[l - 2])) / X[l - 1]

        return psi


def generate_normalized_psi(unnormalized_psi, l2, l3, m2, m3, l_max):
    normalization = np.sqrt(sum(((2 * l) + 1) * (f ** 2) for l, f in unnormalized_psi.items()))
    sgn = 1 if (l2 - l3 + m2 + m3) % 2 == 0 else -1
    is_negative = np.signbit(unnormalized_psi[l_max])  # need to do it this way to handle negative zero
    if is_negative:
        current_sign = -1
    else:
        current_sign = 1
    mult = (1 if current_sign == sgn else -1) / normalization

    return {l: mult * v for l, v in unnormalized_psi.items()}
