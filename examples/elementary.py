"""In mathematics and computability theory, an elementary cellular automaton
is a one-dimensional cellular automaton where there are two possible states
(labeled 0 and 1) and the rule to determine the state of a cell in the next
generation depends only on the current state of the cell and its two immediate
neighbors.
"""
import ndautomata as ca
import matplotlib.pyplot as plt
import numpy as np
from ndautomata import initializers
from ndautomata import neighbours


# -------------------------------------------------------------------
# Automaton ---------------------------------------------------------
# Defined by dimensions, neighbours and states
class Automaton(ca.BaseAutomaton):
    neighbours = neighbours.regular(ndim=1, r=1)
    states = 2


# -------------------------------------------------------------------
# Rule --------------------------------------------------------------
# Requires the same shape as the automaton neighbours
# Dimensions size equals the automaton states
connections = [Automaton.states] * Automaton.neighbours.size
rule = np.unpackbits(np.array([110], dtype="uint8"), bitorder="little")
rule = rule.reshape(connections)


# -------------------------------------------------------------------
# Automaton instantiation -------------------------------------------
# Initialization requires states and size
ic = initializers.center(states=Automaton.states, size=[100])
automaton = Automaton(ic, rule)


# -------------------------------------------------------------------
# Evolution and plot ------------------------------------------------
# Iterate using next and list comprehensions
plt.rcParams["image.cmap"] = "binary"
plt.matshow([next(automaton) for _ in range(200)])
plt.show()
