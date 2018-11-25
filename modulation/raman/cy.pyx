cimport cython

import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def four_wave_polarization(
    np.complex128_t[::1] fields,
    np.complex128_t[::1] phase,
    np.complex128_t[:, :, :, :] polarization_sum_factors,
):
    cdef Py_ssize_t n_modes = fields.shape[0]
    cdef Py_ssize_t q, r, s, t
    cdef np.complex128_t acc, term_r, term_s, pre

    cdef np.complex128_t[::1] conj_fields = np.conj(fields)

    mode_polarization = np.zeros(n_modes, dtype = np.complex128)
    cdef np.complex128_t[::1] mode_polarization_view = mode_polarization

    for r in range(n_modes):
        term_r = fields[r]
        for s in range(n_modes):
            term_s = conj_fields[s]
            for t in range(n_modes):
                pre = term_r * term_s * fields[t]
                for q in range(n_modes):
                    mode_polarization_view[q] = mode_polarization_view[q] + (pre * polarization_sum_factors[q, r, s, t])

    cdef np.complex128_t[::1] conj_phase = np.conj(phase)
    for q in range(n_modes):
        mode_polarization_view[q] = mode_polarization_view[q] * conj_phase[q]

    return mode_polarization
