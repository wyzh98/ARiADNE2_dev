import torch
import numpy as np

a = np.array([1, 2, 3])
b = torch.FloatTensor(a)

print(b)
a[0] = 4
print(b)
print(a)