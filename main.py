import greymind as gm
import random
import numpy as np

data = [[float(random.randint(1,10)) for i in range(3)] for v in range(3)]
data = np.array(data)
print(data)
gm.norm(data)
print(data)
