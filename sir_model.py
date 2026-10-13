import numpy as np


# S -> I (beta*I/N*S)
# I -> R (gamma*I)


def prop(X, coeff):
    S, I, R = X
    N = S + I + R
    return coeff * np.array([I/N*S, I])


def stoch():
    return np.array([
        # S  I  R
        [-1, 1, 0],
        [0, -1, 1],
    ])


def coeff(beta, gamma):
    return np.array([beta, gamma])


def X0(N, I0):
    return np.array([N - I0, I0, 0])


LABELS = ["Susceptible", "Infectious", "Recovered"]
