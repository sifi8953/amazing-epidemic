import numpy as np


# E -> I (alpha*E)
# S -> E (beta*I/N*S)
# I -> R (gamma*I)


def prop(X, coeff):
    S, E, I, R = X
    N = np.sum(X)
    return coeff * np.array([E, I/N*S, I])


def stoch():
    return np.array([
        # S  E  I  R
        [0, -1, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, -1, 1],
    ])


def coeff(alpha, beta, gamma):
    return np.array([alpha, beta, gamma])


def X0(N, I0):
    return np.array([N - I0, 0, I0, 0])
