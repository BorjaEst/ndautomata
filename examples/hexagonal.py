"""Automata based on bidimensional grid of square cells, each of which is in
one of two possible states. Every cell interacts only with six neighbours,
which are the cells that are horizontally, vertically, or half diagonally
adjacent.                         _________________
         / \ / \ / \             |     |     |     |
        |0,0|0,1|0,2|            |     |-1,0 |-1,+1|
         \ / \ / \ / \           |-----|-----|-----|
          |1,0|1,1|1,2|          | 0,-1| i,j | 0,+1|
           \ / \ / \ / \         |-----|-----|-----|
            |2,0|2,1|2,2|        |+1,-1|+1,0 |     |
             \ / \ / \ /         |_____|_____|_____|
"""  # noqa: W605
import ndautomata as ca
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from ndautomata import initializers
from ndautomata import neighbours


# -------------------------------------------------------------------
# Automaton ---------------------------------------------------------
# Defined by dimensions, neighbours and states
class Automaton(ca.BaseAutomaton):
    neighbours = neighbours.hexagonal(ndim=2, r=1)
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
ic = initializers.center(states=2, size=[100, 100])
automaton = Automaton(ic, rule)


# -------------------------------------------------------------------
# Evolution and plot ------------------------------------------------
# Iterate using next and list comprehensions
# TODO: improve using matplotlib.collections.RegularPolyCollection
plt.rcParams["image.cmap"] = "binary"
fig = plt.figure()
plt.matshow(ic, fignum=0)
updt = lambda *args: plt.matshow(next(automaton), fignum=0)  # noqa: E731
anim = animation.FuncAnimation(fig, updt, frames=200, repeat=False)
plt.show()
