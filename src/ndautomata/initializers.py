"""Module for initializer functions."""
import numpy as np


def zeros(states, size):
    """Returns an array where all values are zero.
    :param states: Number of possible states, not used
    :param size: Shape of the return array
    :return: numpy ndarray
    """
    return np.zeros(size, dtype="uint8")


def ones(states, size):
    """Returns an array where all values are one.
    :param states: Number of possible states, not used
    :param size: Shape of the return array
    :return: numpy ndarray
    """
    return (states - 1) * np.ones(size, dtype="uint8")


def center(states, size):
    """Returns an array where all values are zero except the center.
    :param states: Number of possible states, not used
    :param size: Shape of the return array
    :return: numpy ndarray
    """
    middle = tuple(int(d / 2.0) for d in size)
    values = zeros(states, size)
    values[middle] = states - 1
    return values


def border(states, size):
    """Returns an array where all values are zero except the border.
    :param states: Number of possible states, not used
    :param size: Shape of the return array
    :return: numpy ndarray
    """
    border = tuple(0 for _ in size)
    values = zeros(states, size)
    values[border] = states - 1
    return values


def random(states, size):
    """Returns an array where all values are random.
    :param states: Number of possible states, not used
    :param size: Shape of the return array
    :return: numpy ndarray
    """
    return np.random.randint(states, size=size, dtype="uint8")
