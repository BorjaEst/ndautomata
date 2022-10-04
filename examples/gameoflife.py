"""The Game of Life, also known simply as Life, is a cellular automaton
devised by the British mathematician John Horton Conway in 1970. It is a
zero-player game, meaning that its evolution is determined by its initial
state, requiring no further input. One interacts with the Game of Life by
creating an initial configuration and observing how it evolves. It is Turing
complete and can simulate a universal constructor.
"""
import ndautomata as ca
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from ndautomata import initializers
from ndautomata import neighbours


# -------------------------------------------------------------------
# Automaton ---------------------------------------------------------
# Defined by dimensions, neighbours and states
class Automaton(ca.BaseAutomaton):
    neighbours = neighbours.regular(ndim=2, r=1)
    states = 2


# -------------------------------------------------------------------
# Rule --------------------------------------------------------------
# Requires the same shape as the automaton neighbours
# Dimensions size equals the automaton states
connections = [Automaton.states] * Automaton.neighbours.size
rule = initializers.zeros(states=Automaton.states, size=connections)
for index in np.ndindex(rule.shape):
    match index[4]:  # State
        # Any alive cell that is touching less than two alive neighbours dies
        case 1 if sum(index) <= 2:
            rule[index] = 0  # dies
        # Any alive cell touching four or more alive neighbours dies
        case 1 if sum(index) >= 5:
            rule[index] = 0  # dies
        # Any dead cell touching exactly three alive neighbours becomes alive
        case 0 if sum(index) == 3:
            rule[index] = 1  # lives
        # Any alive cell touching two or three alive neighbours does nothing
        case state:
            rule[index] = state


# -------------------------------------------------------------------
# Automaton instantiation -------------------------------------------
# Initialization requires states and size
ic = initializers.random(states=2, size=[100, 100])
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
