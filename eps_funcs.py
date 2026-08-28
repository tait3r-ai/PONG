import math


def norm1(input):
    """Gentle decay multiplier, always < 1. Used when performance is fine."""
    y = 0.975 * math.exp(-0.0010075 * input)
    return y


def norm2(input):
    """Linear multiplier that can exceed 1 for very negative input, giving
    an exploration boost when the smoothed reward is poor."""
    y = -0.01 * input + 0.9
    return y