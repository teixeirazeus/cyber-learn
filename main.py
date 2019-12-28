import cyberlearn as cyberl
import random
import numpy as np

#data = [[float(random.randint(1,10)) for i in range(3)] for v in range(3)]
data = [[1,1,2],
        [2,2,4],
        [3,3,6]]
data = [[float(n) for n in v]for v in data]
x = [[float(v[n]) for n in range(len(v)-1)]for v in data]
y = [[float(v[-1])]for v in data]
x = np.array(x)
y = np.array(y).flatten()

lr = gm.LinearRegression()
lr.train(x, y, 0.001, 1000)
p = lr.predict(np.array([[9,9],[10,10]]))
print("Preveu", p)