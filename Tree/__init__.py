import numpy as np

list = []

a = [[1,2,3],
     [4,5,6],
     [7,8,9],
     [1,2,3],
     [4,5,6]]

ids = [0, 2]

# c = np.take(a, ids)
# print(c)

b = [[1,1,1],
     [1,1,1],
     [1,1,1],
     [1,1,1],
     [1,1,1]]

list.append(a)
list.append(b)
list.append(a)
list.append(b)
print(list)

# sw = np.swapaxes(list, 0, 1)
m = np.mean(list, axis=0)
print(m)