from ndautomata import BaseAutomaton
from pytest import fixture


@fixture(scope="module")
def example():
    import examples.gameoflife as example

    return example


def test_automaton(example):
    assert isinstance(example.automaton, BaseAutomaton)