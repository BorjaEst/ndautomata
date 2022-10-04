"""Module to test neural automaton evolution operations"""
from ndautomata import BaseAutomaton, neighbours, initializers
from pytest import fixture


@fixture(scope="session", params=[1, 2])
def ndim(request):
    return request.param


@fixture(scope="session", params=[1])
def radius(request):
    return request.param


@fixture(scope="session", params=[2, 3])
def nstates(request):
    return request.param


@fixture(scope="session")
def automaton_class(ndim, radius, nstates):
    class Automaton(BaseAutomaton):
        neighbours = neighbours.regular(ndim, radius)
        states = nstates

    return Automaton


@fixture(scope="session", params=[10, 20])
def shape(request, ndim):
    return [request.param] * ndim


@fixture(scope="session", params=[initializers.random])
def rule(request, nstates, automaton_class):
    connections = [nstates] * automaton_class.neighbours.size
    return request.param(states=nstates, size=connections)
