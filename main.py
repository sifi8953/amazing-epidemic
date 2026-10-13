from gillespie import SSA
from rk4 import solve_ivp
from vaccine_model import prop, stoch, coeff, X0, LABELS

import numpy as np
import matplotlib.pyplot as plt

h = 0.1

N = 1000
I0 = 5

ALPHA = 0.5
BETA = 0.3
GAMMA = 1/7
MU = 1/70
V = 2

tspan = [0, 120]

x0 = X0(N, I0)
c = coeff(ALPHA, BETA, GAMMA, MU, V)


def ODE_from_stoch(prop, stoch, coeff):
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

sol = SSA(prop, stoch, x0, tspan, c)
plt.figure(2)
plot_sol(sol, LABELS)
plt.savefig("ssa.png")

plt.show()
