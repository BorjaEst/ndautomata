"""A cellular automaton consists of a regular grid of cells, each in
one of a finite number of states. The grid can be in any finite number
of dimensions. For each cell, a set of cells called its neighborhood
is defined relative to the specified cell.

An initial state (time t = 0) is selected by assigning a state for
each cell. A new generation is created (advancing t by 1), according
to a rule function that determines the new state of each cell in terms
of the current state of the cell and the states of the cells in its
neighborhood.
"""
import copy
from abc import ABC
from math import ceil, floor

import numpy as np
from pydantic import PositiveInt
from scipy.ndimage import correlate

from ndautomata import neighbours


class BaseAutomaton(ABC):
    """Abstract class for automaton generation. You can generate your own
    automaton by subclassing this class defining your own attributes.

    Parameters
    ----------
    initial_configuration : (N,) ndarray
        Initial configuration for all the automaton cells.
        Cell values must be equal or lower than the number of states.
        `all(CA.configuration <= Automaton.states)`
    rule :  (N,) ndarray
        Indexing for next cell value following neighborhood states.
        Rule dimensions mut match the neighbours size.
        Dimension length must be equal or lower than the number of states.
        `Rule.shape ~= [Automaton.states] * Automaton.neighbours.size`

    Attributes
    ----------
    configuration : (N,) ndarray
        NdArray containing all current cell states for the cellular automaton.
    dimensions : PositiveInt
        Number of dimensions the cells are arranged in the cellular automaton.
    rule_constrain : PositiveInt
        Number of dimensions the rule requires.
    rule : (N,) ndarray
        Rule used to calculate next cell states in the cellular automaton.

    Methods
    ----------
    neighbour_indexes(self) : ndarray
        Calculates and returns the index values for each cell neighbours.
    cell_neighbours : (N,) ndarray
        Returns the values of the cell position neighbours as 1-dim array.

    Class Attributes
    ----------------
    neighbours :  (N,) ndarray
        Relative indexing for each cell in the cellular automaton.
    states : PositiveInt
        Amount of possible states a cell can take.
    """

    neighbours: np.ndarray
    states: PositiveInt

    def __init__(self, initial_configuration, rule):
        if initial_configuration.ndim != self.dimensions:
            raise ValueError("Initial configuration does not fit dimensions")
        if np.max(initial_configuration) >= self.states:
            raise ValueError("Initial configuration contains invalid states")
        self.configuration = copy.copy(initial_configuration)
        self.rule = rule
        self.__index = np.empty(initial_configuration.shape, dtype="uint")
        self._weights = np.array(self.states**self.neighbours, dtype="uint")
        self._weights //= self.states

    def neighbour_indexes(self):
        correlate(
            self.configuration,  # Automaton states and neighbours
            self._weights,  # Correlation with connection weights
            mode="wrap",  # ‘wrap’ (a b c d | a b c d | a b c d)
            output=self.__index,  # Output should be uint max
        )
        return self.__index

    def __next__(self):
        self.neighbour_indexes()  # Calculate neighbour indexes
        self.configuration[:] = self._rule[self.__index]
        return copy.deepcopy(self.configuration)

    @classmethod
    @property
    def dimensions(cls):
        return cls.neighbours.ndim

    @classmethod
    @property
    def rule_constrain(cls):
        return cls.neighbours.size

    @property
    def rule(self):
        rule_shape = [self.states] * self.rule_constrain
        return self._rule.reshape(rule_shape)

    @rule.setter
    def rule(self, value):
        if not isinstance(value, np.ndarray):
            raise TypeError("Expected ndarray for rule value")
        if len(value.shape) != self.rule_constrain:
            raise ValueError("Rule shape does not fit neighbours size")
        if np.max(value) >= self.states:
            raise ValueError("Rule contains invalid state values")
        self._rule = value.ravel()

    def cell_neighbours(self, *index):
        shape = self.neighbours.shape
        pads = [(floor(dim / 2), ceil(dim / 2)) for dim in shape]
        array = np.pad(self.configuration, pads, "wrap")
        views = np.lib.stride_tricks.sliding_window_view
        return views(array, shape)[tuple(index)].ravel()[::-1]


if __name__ == "__main__":
    from timeit import timeit

    from ndautomata import initializers

    class Automaton(BaseAutomaton):
        dimensions = 3
        neighbours = neighbours.regular(ndim=3, r=1)
        states = 2

    # Configuration
    connections = [Automaton.states] * Automaton.neighbours.size
    rule = initializers.random(states=Automaton.states, size=connections)
    ic = initializers.random(states=Automaton.states, size=[100, 100, 10])
    ca = Automaton(ic, rule)

    # Profiling
    time = timeit(lambda: [next(ca) for _ in range(100)], number=10)
    print(f"cc next timing: {time}")
