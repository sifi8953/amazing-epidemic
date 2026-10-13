from gillespie import SSA
from rk4 import solve_ivp
from seir_model import prop, stoch, coeff, X0

import numpy as np
import matplotlib.pyplot as plt

h = 0.1

N = 1000
I0 = 5

ALPHA = 0.3
BETA = 0.3
GAMMA = 1/7

tspan = [0, 120]

x0 = X0(N, I0)
c = coeff(ALPHA, BETA, GAMMA)


def ODE_from_stoch(prop, stoch, coeff):
    sMat = stoch()
    return lambda t, X: prop(X, coeff) @ sMat


def plot(t, S, I, R):
    plt.plot(t, S, label="Susceptible")
    plt.plot(t, E, label="Exposed")
    plt.plot(t, I, label="Infectious")
    plt.plot(t, R, label="Resistant")
    plt.xlabel("time")
    plt.ylabel("population")
    plt.legend()
    plt.grid()


f = ODE_from_stoch(prop, stoch, c)
t, y = solve_ivp(f, tspan, x0, h)
S, E, I, R = np.transpose(y)
plt.figure(1)
plot(t, S, I, R)


t, y = SSA(prop, stoch, x0, tspan, c)
S, E, I, R = np.transpose(y)
plt.figure(2)
plot(t, S, I, R)

plt.show()
