import numpy as np

a = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]

b = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]

print(np.c_[a, b])

print(np.r_[a, b])


