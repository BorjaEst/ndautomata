"""Module to test automaton features and requirements"""
import copy

import numpy as np
from ndautomata import initializers
from pytest import fixture, mark


# Module fixtures ---------------------------------------------------
@fixture(scope="class", params=[initializers.random])
def initial_configuration(request, nstates, shape):
    return request.param(nstates, shape)


@fixture(scope="class")
def ic(initial_configuration):
    return initial_configuration


@fixture(scope="class")
def automaton(automaton_class, initial_configuration, rule):
    return automaton_class(initial_configuration, rule)


# Requirements ------------------------------------------------------
class AttrRequirements:
    def test_attr_rule(self, automaton):
        assert hasattr(automaton, "rule")
        assert isinstance(automaton.rule, np.ndarray)

    def test_attr_configuration(self, automaton):
        assert hasattr(automaton, "configuration")
        assert isinstance(automaton.configuration, np.ndarray)

    def test_method_next(self, automaton):
        assert hasattr(automaton, "__next__")
        assert callable(automaton.__next__)

    def test_ic_memory(self, automaton, ic):
        assert not np.shares_memory(automaton.configuration, ic)


class NextRequirements:
    def test_returns_ndarray(self, configuration):
        assert isinstance(configuration, np.ndarray)

    def test_keep_rule(self, original, rule):
        assert np.all(original.rule == rule)

    @mark.parametrize("rule", [initializers.zeros], indirect=True)
    def test_indexing_rule0(self, configuration):
        assert np.all(configuration == 0)

    @mark.parametrize("rule", [initializers.ones], indirect=True)
    def test_indexing_rule1(self, configuration, nstates):
        assert np.all(configuration == nstates - 1)

    def test_next_copies(self, automaton, configuration):
        assert automaton.configuration is not configuration
        assert np.all(automaton.configuration == configuration)


# Parametrization ---------------------------------------------------
class TestAutomaton(AttrRequirements, NextRequirements):
    @fixture(scope="function")
    def original(self, automaton):
        return copy.deepcopy(automaton)

    @fixture(scope="function")
    def configuration(self, automaton, original):
        return next(automaton)
