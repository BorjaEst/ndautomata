"""Automata as three-dimensional orthogonal grid of cubic cells, each of
which is in one of two possible states. Every cell interacts with its twenty
seven neighbours, which are the cells that are horizontally, vertically or
diagonally adjacent."""
import ndautomata as ca
import matplotlib.pyplot as plt
import numpy as np
from ndautomata import initializers
from ndautomata import neighbours


# -------------------------------------------------------------------
# Automaton ---------------------------------------------------------
# Defined by dimensions, neighbours and states
class Automaton(ca.BaseAutomaton):
    neighbours = neighbours.regular(ndim=3, r=1)
    states = 2


# -------------------------------------------------------------------
# Rule --------------------------------------------------------------
# Requires the same shape as the automaton neighbours
# Dimensions size equals the automaton states
connections = [Automaton.states] * Automaton.neighbours.size
rule = initializers.random(states=Automaton.states, size=connections)
rule[1::3] = 0  # Constrain for the rule


# -------------------------------------------------------------------
# Automaton instantiation -------------------------------------------
# Initialization requires states and size
ic = initializers.center(states=Automaton.states, size=[40, 40, 40])
automaton = Automaton(ic, rule)


# -------------------------------------------------------------------
# Evolution and plot ------------------------------------------------
# Iterate using next and mean using np.mean
plt.rcParams["image.cmap"] = "binary"
data = np.array([next(automaton) for _ in range(100)])
fig, axs = plt.subplots(1, automaton.dimensions)
for dim in range(automaton.dimensions):
    axis = tuple(d + 1 for d in range(automaton.dimensions) if d != dim)
    axs[dim].matshow(np.mean(data, axis=axis))
plt.show()
