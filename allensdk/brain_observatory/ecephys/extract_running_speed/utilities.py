
import numpy as np


DEGREES_TO_RADIANS = np.pi / 180.0


def degrees_to_radians(degrees):
    return np.array(degrees) * DEGREES_TO_RADIANS


def angular_to_linear_velocity(angular_velocity, radius):
    return np.multiply(angular_velocity, radius)