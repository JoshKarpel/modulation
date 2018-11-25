import logging

import functools
import collections

import numpy as np

import simulacra as si

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize = None)
def binomial_coefficient(n, k):
    if k < 0 or k > n:
        return 0
    if n <= 1 or k == 0 or k == n:
        return 1

    k = min(k, n - k)

    return binomial_coefficient(n - 1, k - 1) + binomial_coefficient(n - 1, k)


class PrimeFactorization(collections.Counter):
    @classmethod
    @functools.lru_cache(maxsize = None)
    def of_number(cls, n: int):
        factors = []

        test = 2
        while True:
            if n % test == 0:
                factors.append(test)
                n /= test
            else:
                break

        test = 3
        while n != 1:
            if n % test == 0:
                factors.append(test)
                n /= test
            else:
                test += 2

        return cls(factors)

    @classmethod
    @functools.lru_cache(maxsize = None)
    def of_factorial(cls, n: int):
        factors = cls()
        for factor in range(2, n + 1):
            factors *= cls.of_number(factor)

        return factors

    def factor_counts(self):
        yield from self.items()

    @si.utils.cached_property
    def number(self) -> int:
        num = 1
        for factor, count in self.factor_counts():
            num *= (factor ** count)

        return num

    def __add__(self, other):
        raise NotImplementedError

    def __sub__(self, other):
        raise NotImplementedError

    def __mul__(self, other):
        new = PrimeFactorization(self)
        if isinstance(other, int):
            new.update(PrimeFactorization.of_number(other))
        elif isinstance(other, PrimeFactorization):
            new.update(other)
        else:
            raise TypeError

        new.clean()

        return new

    def __truediv__(self, other):
        new = PrimeFactorization(self)
        if isinstance(other, int):
            new.subtract(PrimeFactorization.of_number(other))
        elif isinstance(other, PrimeFactorization):
            new.subtract(other)
        else:
            raise TypeError

        new.clean()

        return new

    def clean(self):
        for factor, count in tuple(self.factor_counts()):
            if count == 0:
                del self[factor]


@functools.lru_cache(maxsize = None)
def factored_triangle_coefficient_squared(x, y, z):
    if x + y + z != int(x + y + z):
        raise Exception

    factorization = PrimeFactorization.of_factorial(int(x + y - z))
    factorization *= PrimeFactorization.of_factorial(int(x - y + z))
    factorization *= PrimeFactorization.of_factorial(int(-x + y + z))
    factorization /= PrimeFactorization.of_factorial(int(x + y + z + 1))
    return factorization


def racah_w(a, b, c, d, e, f):
    alpha_1 = a + b + e
    alpha_2 = c + d + e
    alpha_3 = a + c + f
    alpha_4 = b + d + f
    beta_1 = a + b + c + d
    beta_2 = a + d + e + f
    beta_3 = b + c + e + f

    z_min = max(alpha_1, alpha_2, alpha_3, alpha_4)
    z_max = min(beta_1, beta_2, beta_3)
    w = 0
    for z in range(z_min, z_max + 1):
        sgn = 1 if (z + beta_1) % 2 == 0 else -1

        factorization = PrimeFactorization.of_factorial(z + 1)
        factorization /= PrimeFactorization.of_factorial(z - alpha_1)
        factorization /= PrimeFactorization.of_factorial(z - alpha_2)
        factorization /= PrimeFactorization.of_factorial(z - alpha_3)
        factorization /= PrimeFactorization.of_factorial(z - alpha_4)
        factorization /= PrimeFactorization.of_factorial(beta_1 - z)
        factorization /= PrimeFactorization.of_factorial(beta_2 - z)
        factorization /= PrimeFactorization.of_factorial(beta_3 - z)

        w += sgn * factorization.number

    norm_sq = factored_triangle_coefficient_squared(a, b, e) * factored_triangle_coefficient_squared(c, d, e) * factored_triangle_coefficient_squared(a, c, f) * factored_triangle_coefficient_squared(b, d, f)
    norm = np.sqrt(norm_sq.number)

    return norm * w


def sixj_via_racah(j1, j2, j3, j4, j5, j6):
    sgn = 1 if (j1 + j2 + j4 + j5) % 2 == 0 else -1
    return sgn * racah_w(j1, j2, j5, j4, j3, j6)


def threej_via_racah(l1, l2, l3, m1, m2, m3):
    if not m1 + m2 + m3 == 0:
        return 0

    if not abs(l1 - l2) <= l3 <= l1 + l2:
        return 0

    l_sum = l1 + l2 + l3
    if not l_sum == int(l_sum):
        return 0

    if m1 == m2 == m3 == 0 and l_sum % 2 != 0:
        return 0

    if any((abs(m1) > l1, abs(m2) > l2, abs(m3) > l3)):
        return 0

    if l1 == l2 and m1 == -m2 and l3 == m3 == 0:
        return (1 if (l1 - m1) % 2 == 0 else -1) / np.sqrt((2 * l1) + 1)

    f1 = int(l3 - l2 + m1)
    f2 = int(l3 - l1 - m2)
    f3 = int(l1 + l2 - l3)
    f4 = int(l1 - m1)
    f5 = int(l2 + m2)

    t_min = max(-f1, -f2, 0)
    t_max = min(f3, f4, f5)
    w = 0
    for t in range(t_min, t_max + 1):
        sign = 1 if t % 2 == 0 else -1

        denom = PrimeFactorization.of_factorial(t)
        denom *= PrimeFactorization.of_factorial(f1 + t)
        denom *= PrimeFactorization.of_factorial(f2 + t)
        denom *= PrimeFactorization.of_factorial(f3 - t)
        denom *= PrimeFactorization.of_factorial(f4 - t)
        denom *= PrimeFactorization.of_factorial(f5 - t)

        w += sign / denom.number

    prefactor_squared = factored_triangle_coefficient_squared(l1, l2, l3)
    prefactor_squared *= PrimeFactorization.of_factorial(int(l1 + m1))
    prefactor_squared *= PrimeFactorization.of_factorial(int(l1 - m1))
    prefactor_squared *= PrimeFactorization.of_factorial(int(l2 + m2))
    prefactor_squared *= PrimeFactorization.of_factorial(int(l2 - m2))
    prefactor_squared *= PrimeFactorization.of_factorial(int(l3 + m3))
    prefactor_squared *= PrimeFactorization.of_factorial(int(l3 - m3))
    prefactor = np.sqrt(float(prefactor_squared.number))
    overall_sign = 1 if (l1 - l2 - m3) % 2 == 0 else -1

    return overall_sign * prefactor * w
