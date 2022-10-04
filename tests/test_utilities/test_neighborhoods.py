from ndautomata import neighbours
import numpy as np


def test_1dr1():
    expected = np.flip(np.arange(3).reshape((3,)) + 1)
    assert np.all(neighbours.regular(ndim=1, r=1) == expected)


def test_1dr2():
    expected = np.flip(np.arange(5).reshape((5,)) + 1)
    assert np.all(neighbours.regular(ndim=1, r=2) == expected)


def test_2dr1():
    expected = np.flip(np.arange(9).reshape((3, 3)) + 1)
    assert np.all(neighbours.regular(ndim=2, r=1) == expected)


def test_2dr2():
    expected = np.flip(np.arange(25).reshape((5, 5)) + 1)
    assert np.all(neighbours.regular(ndim=2, r=2) == expected)


def test_3dr1():
    expected = np.flip(np.arange(27).reshape((3, 3, 3)) + 1)
    assert np.all(neighbours.regular(ndim=3, r=1) == expected)


def test_3dr2():
    expected = np.flip(np.arange(125).reshape((5, 5, 5)) + 1)
    assert np.all(neighbours.regular(ndim=3, r=2) == expected)
