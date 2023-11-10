import numpy as np
import matplotlib.pyplot as plt

import random
from datetime import datetime
import os
import sys

random.seed(datetime.now().timestamp())
_r = random.random()
dirname = f"output_crank_nicolson_{_r}"

os.makedirs(dirname)

_t = sys.stdout
sys.stdout = open(f"{dirname}/output.txt", "w")

def thomas_algo(lower_diag, main_diag, upper_diag, sol):
    n = main_diag.size

    if lower_diag.size < n:
        lower_diag = np.concatenate(([0], lower_diag))

    e = np.empty(n)
    f = np.empty(n)
    x = np.empty(n)

    e[0] = upper_diag[0] / main_diag[0]
    f[0] = sol[0] / main_diag[0]

    for i in range(1, n-1):
        denom = main_diag[i] - lower_diag[i] * e[i-1]
        e[i] = upper_diag[i] / denom
        f[i] = (sol[i] - lower_diag[i] * f[i-1]) / denom

    x[n-1] = (sol[n-1] - lower_diag[n-1] * f[n-2]) / (main_diag[n-1] - lower_diag[n-1] * e[n-2])

    for i in range(n-2, -1, -1):
        x[i] = f[i] - e[i] * x[i+1]

    return x


def get_temps(
        l = 0.5,
        points = 10,
        time_intervals = 10,
        initial_temp = 20,
        left_bound = 60,
        right_bound = 100
):
    rod_temps = np.empty((time_intervals, points))
    rod_temps.fill(initial_temp)
    rod_temps[:, 0] = left_bound
    rod_temps[:, -1] = right_bound

    interior_points = points - 2
    lower_diag = np.empty(interior_points-1)
    lower_diag.fill(-1*l)
    main_diag = np.empty(interior_points)
    main_diag.fill(2 * (1 + l))
    upper_diag = np.empty(interior_points-1)
    upper_diag.fill(-1*l)

    for t in range(1, time_intervals):
        sol = np.empty(interior_points)
        for i in range(interior_points):
            sol[i] = l * (rod_temps[t-1][i] + rod_temps[t-1][i+2]) + 2*(1 - l) * rod_temps[t-1][i+1]
        sol[0] += l * left_bound
        sol[-1] += l * right_bound

        x = np.concatenate(([left_bound], thomas_algo(lower_diag, main_diag, upper_diag, sol), [right_bound]))
        rod_temps[t] = x

    return rod_temps


def plot_figure(rod_temps, l):
    plt.figure(dpi=100)
    for t in range(rod_temps.shape[0]):
        plt.plot(np.arange(rod_temps.shape[1]), rod_temps[t], label=f"t = {t}s")
    plt.title(f"lambda = {l:.2f}")
    plt.savefig(f"{dirname}/output_crank_nicolson_lambda={l:.2f}.png")


lambdas = np.linspace(.1, .5, 6)

for l in lambdas:
    rod_temps = np.round(get_temps(l), 2)
    print(rod_temps, end="\n\n")
    plot_figure(rod_temps, l)

sys.stdout = _t
print(f"output saved to {dirname} folder")