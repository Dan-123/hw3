import numpy as np

file = np.fromfile("air.txt", sep=" ")
np.asanyarray(file)
print(file)
print(file[0])