"""Module with tools for automaton definitions."""
from functools import reduce

import numpy as np


def regular(ndim, r):
    """Returns neighbours indexing for regular cell connections.
      _________________    _________________
     |     |     |     |  |     |     |     |
     |-1,-1|-1,0 |-1,+1|  |  9  |  8  |  7  |
     |-----|-----|-----|  |-----|-----|-----|
     | 0,-1| i,j | 0,+1|  |  6  |  5  |  4  |
     |-----|-----|-----|  |-----|-----|-----|
     |+1,-1|+1,0 |+1,+1|  |  3  |  2  |  1  |
     |_____|_____|_____|  |_____|_____|_____|

    :param ndim: Number of cell dimensions with neighbours
    :param r: Cell radius, distance a cell is considered a neighbour
    :return: Numpy array with neighbour indexing
    """
    if ndim < 1 or r < 1:
        raise ValueError
    shape = [1 + 2 * r] * ndim
    return orthogonal(shape)


def hexagonal(ndim, r):
    """Returns neighbour indexing for hexagonal cell connections.
      _________________    _________________
     |     |     |     |  |     |     |     |    / \ / \ / \
     |     |-1,0 |-1,+1|  |  0  |  7  |  6  |   |0,0|0,1|0,2|
     |-----|-----|-----|  |-----|-----|-----|    \ / \ / \ / \
     | 0,-1| i,j | 0,+1|  |  4  |  4  |  3  |     |1,0|1,1|1,2|
     |-----|-----|-----|  |-----|-----|-----|      \ / \ / \ / \
     |+1,-1|+1,0 |     |  |  2  |  1  |  0  |       |2,0|2,1|2,2|
     |_____|_____|_____|  |_____|_____|_____|        \ / \ / \ /

    :param ndim: Number of cell dimensions with neighbours
    :param r: Cell radius, distance a cell is considered a neighbour
    :return: Numpy array with neighbour indexing
    """  # noqa: W605
    if ndim > 2 or r > 1:  # TODO: Implement for more than 2 dimensions
        raise NotImplementedError
    if ndim < 1 or r < 1:
        raise ValueError
    neighbours = regular(ndim, r)
    neighbours[:] -= 1
    neighbours[0, 0] = 0
    neighbours[2, 2] = 0
    return neighbours


def orthogonal(size):
    """Returns neighbour indexing for orthogonal cell connections.
      _______________________    _______________________
     |     |     |     |     |  |     |     |     |     |
     |-1,-1|-1,0 | ... |-1,+n|  | y*x | ... | ... | ... |
     |-----|-----|-----|-----|  |-----|-----|-----|-----|
     | 0,-1| i,j | ... | 0,+n|  | ... | ... | ... | ... |
     |-----|-----|-----|-----|  |-----|-----|-----|-----|
     | ... | ... | ... | ... |  | 2*x | ... | x+2 | x+1 |
     |-----|-----|-----|-----|  |-----|-----|-----|-----|
     |+m,-1|+m,0 | ... |+m,+n|  |  x  | ... |  2  |  1  |
     |_____|_____|_____|_____|  |_____|_____|_____|_____|

    :param size: Shape of connections
    :return: Numpy array with neighbour indexing
    """
    n_neighbours = reduce((lambda x, y: x * y), size)
    return np.arange(n_neighbours, 0, -1).reshape(size)
