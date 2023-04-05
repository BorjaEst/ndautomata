"""Module to test automaton features and requirements.
Note order of bits in rules is from lowest to biggest.
"""
import cupy as np
from cupy.testing import assert_array_equal as assert_equal

from ndautomata import BaseAutomaton, initializers, neighbours


class TestElementaryD1R1:
    class Automaton(BaseAutomaton):
        neighbours = neighbours.regular(ndim=1, r=1)
        states = 10

    ic = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype="uint8")

    def test_cells_neighbours(self):
        rule = initializers.random(states=2, size=[2] * 3)
        nc = self.Automaton(self.ic, rule)
        assert_equal(nc.cell_neighbours(0), [1, 0, 9])
        assert_equal(nc.cell_neighbours(1), [2, 1, 0])
        assert_equal(nc.cell_neighbours(2), [3, 2, 1])
        assert_equal(nc.cell_neighbours(3), [4, 3, 2])
        assert_equal(nc.cell_neighbours(4), [5, 4, 3])
        assert_equal(nc.cell_neighbours(5), [6, 5, 4])
        assert_equal(nc.cell_neighbours(6), [7, 6, 5])
        assert_equal(nc.cell_neighbours(7), [8, 7, 6])
        assert_equal(nc.cell_neighbours(8), [9, 8, 7])
        assert_equal(nc.cell_neighbours(9), [0, 9, 8])


class TestOrthogonalW2L5:
    class Automaton(BaseAutomaton):
        neighbours = neighbours.orthogonal([2, 5])
        states = 10

    ic = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype="uint8")

    def test_cells_neighbours(self):
        rule = initializers.random(states=2, size=[2] * 10)
        nc = self.Automaton(self.ic, rule)
        assert_equal(nc.cell_neighbours(0, 0), [2, 1, 0, 4, 3, 7, 6, 5, 9, 8])
        assert_equal(nc.cell_neighbours(0, 1), [3, 2, 1, 0, 4, 8, 7, 6, 5, 9])
        assert_equal(nc.cell_neighbours(0, 2), [4, 3, 2, 1, 0, 9, 8, 7, 6, 5])
        assert_equal(nc.cell_neighbours(0, 3), [0, 4, 3, 2, 1, 5, 9, 8, 7, 6])
        assert_equal(nc.cell_neighbours(0, 4), [1, 0, 4, 3, 2, 6, 5, 9, 8, 7])
        assert_equal(nc.cell_neighbours(1, 0), [7, 6, 5, 9, 8, 2, 1, 0, 4, 3])
        assert_equal(nc.cell_neighbours(1, 1), [8, 7, 6, 5, 9, 3, 2, 1, 0, 4])
        assert_equal(nc.cell_neighbours(1, 2), [9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
        assert_equal(nc.cell_neighbours(1, 3), [5, 9, 8, 7, 6, 0, 4, 3, 2, 1])
        assert_equal(nc.cell_neighbours(1, 4), [6, 5, 9, 8, 7, 1, 0, 4, 3, 2])


class TestOrthogonalW2L2H2:
    class Automaton(BaseAutomaton):
        neighbours = neighbours.orthogonal([2, 2, 2])
        states = 10

    ic = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype="uint8")

    def test_cells_neighbours(self):
        rule = initializers.random(states=2, size=[2] * 8)
        nc = self.Automaton(self.ic, rule)
        assert_equal(nc.cell_neighbours(0, 0, 0), [0, 1, 2, 3, 4, 5, 6, 7])
        assert_equal(nc.cell_neighbours(0, 0, 1), [1, 0, 3, 2, 5, 4, 7, 6])
        assert_equal(nc.cell_neighbours(0, 1, 0), [2, 3, 0, 1, 6, 7, 4, 5])
        assert_equal(nc.cell_neighbours(0, 1, 1), [3, 2, 1, 0, 7, 6, 5, 4])
        assert_equal(nc.cell_neighbours(1, 0, 0), [4, 5, 6, 7, 0, 1, 2, 3])
        assert_equal(nc.cell_neighbours(1, 0, 1), [5, 4, 7, 6, 1, 0, 3, 2])
        assert_equal(nc.cell_neighbours(1, 1, 0), [6, 7, 4, 5, 2, 3, 0, 1])
        assert_equal(nc.cell_neighbours(1, 1, 1), [7, 6, 5, 4, 3, 2, 1, 0])
