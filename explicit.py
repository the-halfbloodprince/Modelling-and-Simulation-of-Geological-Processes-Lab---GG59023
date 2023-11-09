import numpy as np
import matplotlib.pyplot as plt

# imports for generating random numbers to store multiple results in separate folders
import random
from datetime import datetime
import os

import sys

# generate new dir
random.seed(datetime.now().timestamp())
_r = random.random()
dirname = f"output_explicit_{_r}"
os.makedirs(dirname)

_t = sys.stdout
sys.stdout = open(f"{dirname}/output.txt", "w")

# ====== MAIN CODE STARTS ======
def explicit_calculation(l, a, b, c):
    return b + l * (a - 2 * b + c)

def get_temps(
        l=.5,
        points=10,
        time_intervals=10,
        initial_temp=20,
        left_bound=60,
        right_bound=100
):
    rod_temps = np.empty((time_intervals, points))

    # initial temperatures
    rod_temps[0].fill(initial_temp)

    # boundary conditions
    rod_temps[:, 0] = left_bound
    rod_temps[:, -1] = right_bound

    # fill in the next time steps
    for t in range(1, time_intervals):
        # calculate all the interior points
        for x in range(1, points-1):
            rod_temps[t, x] = explicit_calculation(l, rod_temps[t-1][x-1], rod_temps[t-1][x], rod_temps[t-1][x+1])

    return rod_temps


def plot_figure(rod_temps, l):
    # plotting graph
    plt.figure(dpi=100)
    for t in range(rod_temps.shape[0]):
        plt.plot(np.arange(rod_temps.shape[1]), rod_temps[t], label=f"t = {t}s")
    plt.title(label=f"lambda = {l}")
    plt.legend()
    # plt.show()
    plt.savefig(f"{dirname}/explicit_lambda={l:.2f}.png")


lambdas = np.round(np.linspace(.1, .5, 6), 3)

print("lambdas: ", lambdas, end="\n\n")

for l in lambdas:
    rod_temps = np.round(get_temps(l), 2)
    print(rod_temps, end="\n\n")
    plot_figure(rod_temps, l)

# ======   MAIN CODE END  ======

sys.stdout.close()
sys.stdout = _t

print(f"files saved to /{dirname}")