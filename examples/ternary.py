"""One-dimensional cellular automaton where there are three possible states
(labeled 0, 1 and 2) and the rule to determine the state of a cell in the
next generation depends only on the current state of the cell and its two
immediate neighbors.
"""
import ndautomata as ca
import matplotlib.pyplot as plt
from ndautomata import initializers
from ndautomata import neighbours


# -------------------------------------------------------------------
# Automaton ---------------------------------------------------------
# Defined by dimensions, neighbours and states
class Automaton(ca.BaseAutomaton):
    neighbours = neighbours.regular(ndim=1, r=1)
    states = 3


# -------------------------------------------------------------------
# Rule --------------------------------------------------------------
# Requires the same shape as the automaton neighbours
# Dimensions size equals the automaton states
connections = [Automaton.states] * Automaton.neighbours.size
rule = initializers.random(states=Automaton.states, size=connections)


# -------------------------------------------------------------------
# Automaton instantiation -------------------------------------------
# Initialization requires states and size
ic = initializers.center(states=3, size=[100])
automaton = Automaton(ic, rule)


# -------------------------------------------------------------------
# Evolution and plot ------------------------------------------------
# Iterate using next and list comprehensions
plt.rcParams["image.cmap"] = "binary"
plt.matshow([next(automaton) for _ in range(200)])
plt.show()
