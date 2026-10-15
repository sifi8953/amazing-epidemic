from gillespie import SSA
from rk4 import solve_ivp
from vaccine_two_dose_model import prop, stoch, coeff, X0, LABELS

import numpy as np
import matplotlib.pyplot as plt

h = 0.1

N = 1000
I0 = 5

ALPHA = 0.5
BETA = 0.3
GAMMA = 1/7
MU = 1/70
OMEGA = 0 #Ratio not immune after infection 
SIGMA_2 = 0.6 #Full vaccine efficiency
SIGMA_1 = 0.94 * SIGMA_2 #One dose vaccine efficiency
V_1 = 0
V_2 = 0

SIMULATIONS = 1

tspan = [0, 120]

x0 = X0(N, I0)
c = coeff(ALPHA, BETA, GAMMA, MU, OMEGA, SIGMA_1, SIGMA_2, V_1, V_2)


def ODE_from_stoch(prop, stoch, coeff):
    # avoid having to define all terms twice by using
    # the stochastic model terms to reconstruct the ODE
    sMat = stoch()
    return lambda t, X: prop(X, coeff) @ sMat


def plot_sol(sol, labels):
    t, y = sol
    y = np.transpose(y)
    for points, label in zip(y, labels, strict=True):
        plt.plot(t, points, label=label)
    plt.xlabel("time")
    plt.ylabel("population")
    plt.legend()
    plt.grid()


f = ODE_from_stoch(prop, stoch, c)
sol = solve_ivp(f, tspan, x0, h)
plt.figure(1)
plot_sol(sol, LABELS)
plt.savefig("ode.png")
for i in range(SIMULATIONS):
    sol = SSA(prop, stoch, x0, tspan, c)
    plt.figure(2)
    plot_sol(sol, LABELS)

plt.savefig("ssa.png")
plt.show()
