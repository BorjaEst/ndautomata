"""In this example, the automaton neighbours are a orthogonal rectangle with
the following features:
 - One neighbours dimension is longer than the other
 - One neighbours dimension is pair (no symmetric)
"""
import ndautomata as ca
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from ndautomata import initializers
from ndautomata import neighbours


# -------------------------------------------------------------------
# Automaton ---------------------------------------------------------
# Defined by dimensions, neighbours and states
class Automaton(ca.BaseAutomaton):
    neighbours = neighbours.orthogonal([5, 2])
    states = 2


# -------------------------------------------------------------------
# Rule --------------------------------------------------------------
# Requires the same shape as the automaton neighbours
# Dimensions size equals the automaton states
connections = [Automaton.states] * Automaton.neighbours.size
rule = initializers.random(states=Automaton.states, size=connections)


# -------------------------------------------------------------------
# Automaton instantiation -------------------------------------------
# Initialization requires states and size
ic = initializers.center(states=Automaton.states, size=[100, 100])
automaton = Automaton(ic, rule)


# -------------------------------------------------------------------
# Evolution and plot ------------------------------------------------
# Iterate using next and list comprehensions
plt.rcParams["image.cmap"] = "binary"
fig = plt.figure()
plt.matshow(ic, fignum=0)
updt = lambda *args: plt.matshow(next(automaton), fignum=0)  # noqa: E731
anim = animation.FuncAnimation(fig, updt, frames=200, repeat=False)
plt.show()
