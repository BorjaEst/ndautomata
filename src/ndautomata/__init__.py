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
    rule : (N,) ndarray
        Rule used to calculate next cell states in the cellular automaton.

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
        assert initial_configuration.ndim == self.dimensions
        assert np.max(initial_configuration) < self.states
        self.configuration = copy.copy(initial_configuration)
        self.rule = rule
        self.__index = np.empty(initial_configuration.shape, dtype="uint")
        self._weights = np.array(self.states**self.neighbours, dtype="uint")
        self._weights //= self.states

    def __next__(self):
        correlate(
            self.configuration,  # Automaton states and neighbours
            self._weights,  # Correlation with connection weights
            output=self.__index,  # Output should be uint max
            mode="wrap",  # ‘wrap’ (a b c d | a b c d | a b c d)
        )
        self.configuration[:] = self._rule[self.__index]
        return copy.deepcopy(self.configuration)

    @classmethod
    @property
    def dimensions(cls):
        return cls.neighbours.ndim

    @property
    def rule(self):
        rule_shape = [self.states] * self.neighbours.size
        return self._rule.reshape(rule_shape)

    @rule.setter
    def rule(self, value):
        assert isinstance(value, np.ndarray)
        assert len(value.shape) == self.neighbours.size
        assert np.max(value) < self.states
        self._rule = value.ravel()


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
