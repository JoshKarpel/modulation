import numpy as np

from modulation.resonators.microspheres import vsh


def test_inner_product():
    a = np.zeros((2, 2, 3))
    b = np.ones_like(a)

    a[0, 0] = np.array([1, 2, 3])
    a[0, 1] = np.array([4, 5, 6])
    a[1, 0] = np.array([7, 8, 9])
    a[1, 1] = np.array([10, 11, 12])

    e = np.einsum("ijk,ijk->ij", a, b)
    v = vsh.inner_product_of_vsh(a, b)

    assert (e == v).all()
    assert np.allclose(e, np.array([[6, 15], [24, 33]]))
    assert np.allclose(v, np.array([[6, 15], [24, 33]]))
