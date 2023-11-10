import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# leibmann

def get_temps(
        l = 0.5,
        x_points = 7,
        y_points = 7,
        time_intervals = 10,
        initial_temp = 40,
        bound_temps = [150, 250, 100, 200],
        max_err = 1e-9
):
    # rod_temps = np.empty((time_intervals, y_points, x_points))
    # rod_temps.fill(initial_temp)
    grid = np.empty((y_points, x_points))
    grid.fill(initial_temp)
    grid[0, :] = bound_temps[0]
    grid[:, -1] = bound_temps[3]
    grid[-1, :] = bound_temps[1]
    grid[:, 0] = bound_temps[2]

    print(grid)

    grid_old = np.copy(grid)

    err = 1e9
    max_iterations = 200
    iterations_completed = 0

    while err >= max_err or iterations_completed <= max_iterations:
        for y in range(1, y_points - 1):
            for x in range(1, x_points - 1):
                grid[y][x] = (grid_old[y-1][x] + grid_old[y][x+1] + grid_old[y+1][x] + grid_old[y][x-1])/4
        err = np.sqrt(np.sum((grid - grid_old) ** 2) / (x_points-2)*(y_points-2))
        grid_old = np.copy(grid)
        iterations_completed += 1

    print("iterations completed: ", iterations_completed)
    print(np.round(grid, 2))

    return grid


# contour plot of grid
grid = get_temps()
plt.figure(dpi=100)
plt.contourf(grid, cmap=cm.hot)
plt.show()