# imports
import numpy as np
import matplotlib.pyplot as plt

# 1D array
a = np.array([-10, 10])
b = np.array([1, 2, 3])

# 2D array (matrix)
m = np.array([[1, 2, 3], [4, 5, 6], [7, 11, 9]])
print(m)
print(m.shape)

# det
print(np.linalg.det(m))

# transpose
m = m.T
print(m)
print(m.shape)

# inverse
mi = np.linalg.inv(m)
print(mi)

# multiply
mm = m @ mi

mm = np.round(mm)

print(mm)

# plt.plot(a, np.sin(a))
# plt.show()