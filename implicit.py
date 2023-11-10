import numpy as np
import matplotlib.pyplot as plt

import random
from datetime import datetime
import os
import sys

random.seed(datetime.now().timestamp())
_r = random.random()
dirname = f"output_implicit_{_r}"
os.makedirs(dirname)

_t = sys.stdout
sys.stdout = open(f"{dirname}/output.txt", "w")


def thomas_algo(lower_diag, main_diag, upper_diag, sol):
    # number of equations
    n = main_diag.size

    if lower_diag.size < n:
        # pad
        lower_diag = np.concatenate(([0], lower_diag))

    # print(lower_diag)
    # print(main_diag)
    # print(upper_diag)

    e = np.zeros(n)
    f = np.zeros(n)
    x = np.zeros(n)

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

# def make_tridiag_matrix(lower_diag, main_diag, upper_diag):
#     m = np.empty((main_diag.size, main_diag.size))
#     m[0][0] = main_diag[0]
#     m[0][1] = upper_diag[0]
#
#

def get_temps(
        l = .5,
        points = 10,
        time_intervals = 10,
        initial_temp = 20,
        left_bound = 60,
        right_bound = 100
):
    interior_points = points - 2

    rod_temps = np.empty((time_intervals, points))

    rod_temps.fill(initial_temp)
    rod_temps[:, 0] = left_bound
    rod_temps[:, -1] = right_bound

    lower_diag = np.empty(interior_points - 1)
    lower_diag.fill(-1*l)
    main_diag = np.empty(interior_points)
    main_diag.fill(1 + 2*l)
    upper_diag = np.empty(interior_points - 1)
    upper_diag.fill(-1*l)

    # make_tridiagona_matrix(lower_diag, main_diag, upper_diag)

    for t in range(1, time_intervals):
        sol = np.delete(np.copy(rod_temps[t-1]), [0, -1])
        sol[0] += l * left_bound
        sol[-1] += l * right_bound
        x = thomas_algo(lower_diag, main_diag, upper_diag, sol)
        # print("--", x)
        x = np.concatenate(([left_bound], x, [right_bound]))
        rod_temps[t] = x

    return rod_temps


def plot_figure(rod_temps, l):
    plt.figure(dpi=100)
    for t in range(rod_temps.shape[0]):
        plt.plot(np.arange(rod_temps.shape[1]), rod_temps[t], label = f"t = {t}s")
    plt.title(f"lambda = {l}")
    plt.legend()
    plt.savefig(f"{dirname}/output_implicit_lambda={l:.2f}.png")


lambdas = np.round(np.linspace(.1, .5, 6), 3)

for l in lambdas:
    rod_temps = np.round(get_temps(l), 2)
    print(rod_temps, end="\n\n")
    plot_figure(rod_temps, l)

sys.stdout.close()
sys.stdout = _t
print(f"files saved to {dirname}")