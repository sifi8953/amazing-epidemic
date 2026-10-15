import numpy as np

# coeff = [alpha, beta, gamma, mu, omega, sigma_1, sigma_2, v_1, v_2]

# C = beta*I/N

# E -> I (alpha*E)
# S -> E (C*S)
# I -> R (gamma*I)
# I -> D (mu*I)
# R -> E (omega*C*R)
# V_1 -> E ((1-sigma_1)*C*V_1)
# V_2 -> E ((1-sigma_2)*C*V_2)
# S -> V_1 (v_1)
# V_1 -> V_2 (v_2*V_1)


def prop(X, coeff):
    S, E, I, R, D, V_1, V_2 = X
    N = np.sum(X)  # number of dead count toward population
    ALPHA, BETA, GAMMA, MU, OMEGA, SIGMA_1, SIGMA_2, v_1, v_2 = coeff
    C = BETA * I / N
    return np.array([ALPHA * E, C * S, GAMMA * I, MU * I, OMEGA * C * R,
                     (1 - SIGMA_1) * C * V_1, (1 - SIGMA_2) * C * V_2, v_1, v_2 * V_1])


def stoch():
    return np.array([
        # S  E  I  R  D  V_1 V_2
        [0, -1, 1, 0, 0, 0, 0],
        [-1, 1, 0, 0, 0, 0, 0],
        [0, 0, -1, 1, 0, 0, 0],
        [0, 0, -1, 0, 1, 0, 0],
        [0, 1, 0, -1, 0, 0, 0],
        [0, 1, 0, 0, 0, -1, 0],
        [0, 1, 0, 0, 0, 0, -1],
        [-1, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, -1, 1],
    ])


def coeff(alpha, beta, gamma, mu, omega, sigma_1, sigma_2, v_1, v_2):
    return np.array([alpha, beta, gamma, mu, omega, sigma_1, sigma_2, v_1, v_2])


def X0(N, I0):
    return np.array([N - I0, 0, I0, 0, 0, 0, 0])


LABELS = ["Susceptible", "Exposed", "Infectious", "Recovered", "Deceased", "Vaccinated 1 dose", "Vaccinated 2 doses"]
