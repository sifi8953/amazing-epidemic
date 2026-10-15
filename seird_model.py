import numpy as np


# E -> I (alpha*E)
# S -> E (beta*I/N*S)
# I -> R (gamma*I)
# I -> D (gamma*I)


def prop(X, coeff):
    S, E, I, R, D = X
    N = np.sum(X)  # number of dead count toward population
    return coeff * np.array([E, I/N*S, I, I])


def stoch():
    return np.array([
        # S  E  I  R  D
        [0, -1, 1, 0, 0],
        [-1, 1, 0, 0, 0],
        [0, 0, -1, 1, 0],
        [0, 0, -1, 0, 1],
    ])


def coeff(alpha, beta, gamma, mu):
    return np.array([alpha, beta, gamma, mu])


def X0(N, I0):
    return np.array([N - I0, 0, I0, 0, 0])


LABELS = ["Susceptible", "Exposed", "Infectious", "Recovered", "Deceased"]
