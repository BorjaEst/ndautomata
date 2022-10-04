"""Module to test automaton features and requirements.
Note order of bits in rules is from lowest to biggest.
"""
import numpy as np
from ndautomata import BaseAutomaton, neighbours, initializers
from pytest import mark


def srule(buffer):
    array = np.frombuffer(buffer, dtype="uint8")[::-1]
    return np.unpackbits(array, bitorder="little")


class TestElementaryR1S2:
    class Automaton(BaseAutomaton):
        neighbours = neighbours.regular(ndim=1, r=1)
        states = 2

    ic = np.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 0], dtype="uint8")

    @mark.parametrize("repeat", range(10))
    def test_rule_indexes(self, repeat):
        rule = initializers.random(states=2, size=[2] * 3)
        nc = next(self.Automaton(self.ic, rule))
        assert nc[0] == rule[0, 0, 1]
        assert nc[1] == rule[0, 1, 0]
        assert nc[2] == rule[1, 0, 0]
        assert nc[3] == rule[0, 0, 1]
        assert nc[4] == rule[0, 1, 1]
        assert nc[5] == rule[1, 1, 0]
        assert nc[6] == rule[1, 0, 1]
        assert nc[7] == rule[0, 1, 0]
        assert nc[8] == rule[1, 0, 0]
        assert nc[9] == rule[0, 0, 0]

    def test_rule_0b00000000(self):  # 000
        rule = srule(b"\x00").reshape(2, 2, 2)
        nc = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype="uint8")
        assert np.all(nc == next(self.Automaton(self.ic, rule)))

    def test_rule_0b00000001(self):  # 001
        rule = srule(b"\x01").reshape(2, 2, 2)
        nc = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype="uint8")
        assert np.all(nc == next(self.Automaton(self.ic, rule)))

    def test_rule_0b00100000(self):  # 032
        rule = srule(b"\x20").reshape(2, 2, 2)
        nc = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype="uint8")
        assert np.all(nc == next(self.Automaton(self.ic, rule)))

    def test_rule_0b01101110(self):  # 110
        rule = srule(b"\x6e").reshape(2, 2, 2)
        nc = np.array([1, 1, 0, 1, 1, 1, 1, 1, 0, 0], dtype="uint8")
        assert np.all(nc == next(self.Automaton(self.ic, rule)))

    def test_rule_0b01111101(self):  # 125
        rule = srule(b"\x7d").reshape(2, 2, 2)
        nc = np.array([0, 1, 1, 0, 1, 1, 1, 1, 1, 1], dtype="uint8")
        assert np.all(nc == next(self.Automaton(self.ic, rule)))

    def test_rule_0b11010111(self):  # 215
        rule = srule(b"\xd7").reshape(2, 2, 2)
        nc = np.array([1, 1, 1, 1, 0, 1, 0, 1, 1, 1], dtype="uint8")
        assert np.all(nc == next(self.Automaton(self.ic, rule)))

    def test_rule_0b11111110(self):  # 254
        rule = srule(b"\xfe").reshape(2, 2, 2)
        nc = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0], dtype="uint8")
        assert np.all(nc == next(self.Automaton(self.ic, rule)))

    def test_rule_0b11111111(self):  # 255
        rule = srule(b"\xff").reshape(2, 2, 2)
        nc = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype="uint8")
        assert np.all(nc == next(self.Automaton(self.ic, rule)))


class TestElementaryR2S2:
    class Automaton(BaseAutomaton):
        neighbours = neighbours.regular(ndim=1, r=2)
        states = 2

    ic = np.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 0], dtype="uint8")

    @mark.parametrize("repeat", range(10))
    def test_rule_indexes(self, repeat):
        rule = initializers.random(states=2, size=[2] * 5)
        nc = next(self.Automaton(self.ic, rule))
        assert nc[0] == rule[0, 0, 0, 1, 0]
        assert nc[1] == rule[0, 0, 1, 0, 0]
        assert nc[2] == rule[0, 1, 0, 0, 1]
        assert nc[3] == rule[1, 0, 0, 1, 1]
        assert nc[4] == rule[0, 0, 1, 1, 0]
        assert nc[5] == rule[0, 1, 1, 0, 1]
        assert nc[6] == rule[1, 1, 0, 1, 0]
        assert nc[7] == rule[1, 0, 1, 0, 0]
        assert nc[8] == rule[0, 1, 0, 0, 0]
        assert nc[9] == rule[1, 0, 0, 0, 1]

    def test_rule_0x00000000(self):
        rule = srule(b"\x00\x00\x00\x00").reshape([2] * 5)
        nc = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype="uint8")
        assert np.all(nc == next(self.Automaton(self.ic, rule)))

    def test_rule_0x00000001(self):
        rule = srule(b"\x00\x00\x00\x01").reshape([2] * 5)
        nc = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype="uint8")
        assert np.all(nc == next(self.Automaton(self.ic, rule)))

    def test_rule_0x00000004(self):
        rule = srule(b"\x00\x00\x00\x04").reshape([2] * 5)
        nc = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype="uint8")
        assert np.all(nc == next(self.Automaton(self.ic, rule)))

    def test_rule_0x00000200(self):
        rule = srule(b"\x00\x00\x02\x00").reshape([2] * 5)
        nc = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype="uint8")
        assert np.all(nc == next(self.Automaton(self.ic, rule)))

    def test_rule_0x00000204(self):
        rule = srule(b"\x00\x00\x02\x04").reshape([2] * 5)
        nc = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype="uint8")
        assert np.all(nc == next(self.Automaton(self.ic, rule)))

    def test_rule_0x00080000(self):
        rule = srule(b"\x00\x08\x00\x00").reshape([2] * 5)
        nc = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype="uint8")
        assert np.all(nc == next(self.Automaton(self.ic, rule)))

    def test_rule_0x00080204(self):
        rule = srule(b"\x00\x08\x02\x04").reshape([2] * 5)
        nc = np.array([1, 0, 1, 1, 0, 0, 0, 0, 0, 0], dtype="uint8")
        assert np.all(nc == next(self.Automaton(self.ic, rule)))

    def test_rule_0xffffffff(self):
        rule = srule(b"\xff\xff\xff\xff").reshape([2] * 5)
        nc = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype="uint8")
        assert np.all(nc == next(self.Automaton(self.ic, rule)))


class TestElementaryR2S3:
    class Automaton(BaseAutomaton):
        neighbours = neighbours.regular(ndim=1, r=2)
        states = 3

    ic = np.array([0, 1, 2, 0, 1, 1, 0, 1, 2, 0], dtype="uint8")

    @mark.parametrize("repeat", range(10))
    def test_rule_indexes(self, repeat):
        rule = initializers.random(states=3, size=[3] * 5)
        nc = next(self.Automaton(self.ic, rule))
        assert nc[0] == rule[2, 0, 0, 1, 2]
        assert nc[1] == rule[0, 0, 1, 2, 0]
        assert nc[2] == rule[0, 1, 2, 0, 1]
        assert nc[3] == rule[1, 2, 0, 1, 1]
        assert nc[4] == rule[2, 0, 1, 1, 0]
        assert nc[5] == rule[0, 1, 1, 0, 1]
        assert nc[6] == rule[1, 1, 0, 1, 2]
        assert nc[7] == rule[1, 0, 1, 2, 0]
        assert nc[8] == rule[0, 1, 2, 0, 0]
        assert nc[9] == rule[1, 2, 0, 0, 1]


class TestOrthogonalR1S2:
    class Automaton(BaseAutomaton):
        neighbours = neighbours.regular(ndim=2, r=1)
        states = 2

    ic = np.array([[0, 1, 0, 0, 1, 1, 0, 1, 0, 0]] * 6, dtype="uint8")

    @mark.parametrize("repeat", range(10))
    def test_rule_indexes(self, repeat):
        rule = initializers.random(states=2, size=[2] * 9)
        nc = next(self.Automaton(self.ic, rule))
        assert nc[0][0] == rule[0, 0, 1, 0, 0, 1, 0, 0, 1]
        assert nc[0][1] == rule[0, 1, 0, 0, 1, 0, 0, 1, 0]
        assert nc[0][2] == rule[1, 0, 0, 1, 0, 0, 1, 0, 0]
        assert nc[0][3] == rule[0, 0, 1, 0, 0, 1, 0, 0, 1]
        assert nc[0][4] == rule[0, 1, 1, 0, 1, 1, 0, 1, 1]
        assert nc[0][5] == rule[1, 1, 0, 1, 1, 0, 1, 1, 0]
        assert nc[0][6] == rule[1, 0, 1, 1, 0, 1, 1, 0, 1]
        assert nc[0][7] == rule[0, 1, 0, 0, 1, 0, 0, 1, 0]
        assert nc[0][8] == rule[1, 0, 0, 1, 0, 0, 1, 0, 0]
        assert nc[0][9] == rule[0, 0, 0, 0, 0, 0, 0, 0, 0]

    def test_rule_0_0(self):
        rule = np.array([0] * 2 ** 9).reshape([2] * 9)
        nc = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * 6, dtype="uint8")
        assert np.all(nc == next(self.Automaton(self.ic, rule)))

    def test_rule_0_1(self):
        rule = np.array([1] + [0] * (2 ** 9 - 1)).reshape([2] * 9)
        nc = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]] * 6, dtype="uint8")
        assert np.all(nc == next(self.Automaton(self.ic, rule)))

    def test_rule_1_1(self):
        rule = np.array([1] * 2 ** 9).reshape([2] * 9)
        nc = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]] * 6, dtype="uint8")
        assert np.all(nc == next(self.Automaton(self.ic, rule)))
