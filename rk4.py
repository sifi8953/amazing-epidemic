import numpy as np


def solve_ivp(f, t_span, y0, h):
    steps = round((t_span[1] - t_span[0]) / h)
    t = np.linspace(t_span[0], t_span[1], steps + 1)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0

    for i in range(steps):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h/2, y[i] + (h/2)*k1)
        k3 = f(t[i] + h/2, y[i] + (h/2)*k2)
        k4 = f(t[i+1], y[i] + h*k3)
        y[i+1] = y[i] + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

    return t, y
