import numpy as np


# E -> I (alpha*E)
# S -> E (beta*I/N*S)
# I -> R (gamma*I)
# I -> D (mu*I)
# S -> V (v)


def prop(X, coeff):
    S, E, I, R, D, V = X
    N = S + E + I + R + V
    return coeff * np.array([E, I/N*S, I, I, 1])


def stoch():
    return np.array([
        # S  E  I  R  D  V
        [0, -1, 1, 0, 0, 0],
        [-1, 1, 0, 0, 0, 0],
        [0, 0, -1, 1, 0, 0],
        [0, 0, -1, 0, 1, 0],
        [-1, 0, 0, 0, 0, 1],
    ])


def coeff(alpha, beta, gamma, mu, v):
    return np.array([alpha, beta, gamma, mu, v])


def X0(N, I0):
    return np.array([N - I0, 0, I0, 0, 0, 0])


LABELS = ["Susceptible", "Exposed", "Infectious", "Recovered", "Deceased", "Vaccinated"]
