import numpy as np
from numpy import pi
from scipy.special import gamma

def M13a(n1):
    """ Common part of the 13-loop matrices """
    return np.tan(n1 * pi) / (14. * (-3 + n1) * (-2 + n1) * (-1 + n1) * n1 * pi)

def M22a(n1, n2):
    """ Common part of the 22-loop matrices """
    return (gamma(1.5 - n1) * gamma(1.5 - n2) * gamma(-1.5 + n1 + n2)) / (8. * pi**1.5 * gamma(n1) * gamma(3 - n1 - n2) * gamma(n2))