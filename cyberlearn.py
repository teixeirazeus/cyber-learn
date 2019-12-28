import numpy as np

def apply_norm(data, maxNmin):
    """Aplica normalizaçao dado um vetor de minimos e maximos"""
    for atribute, mm in enumerate(maxNmin):
        min, max = mm
        for i in range(len(data)):
            data[i][atribute] = (data[i][atribute] - min) / (max - min)


def norm(data):
    """Normaliza uma lista de vetores."""
    maxNmin = []
    for atribute in range(len(data[0])):
        # get min and max
        min, max = np.inf, -np.inf
        for vector in data:
            if vector[atribute] < min: min = vector[atribute]
            if vector[atribute] > max: max = vector[atribute]
        maxNmin.append([min,max])
        # update
    apply_norm(data, maxNmin)
    return maxNmin

class LinearRegression:
    def __init__(self):
        self.theta = []
    def cost(self, x, y, theta):
        m = y.shape[0] # number of samples
        h = x.dot(theta) - y;
        return (1/(2*m)) * np.sum(h**2)

    def train(self, x, y, alpha, iterations):
        # alpha is the learning rate
        if len(self.theta) == 0: self.theta = np.array([0.0 for atribute in x[0]]).T
        m = y.shape[0]  # number of samples
        history = []
        for i in range(iterations):
            # e = sum(((X * theta) - y). * X);
            #e = np.sum((x.dot(self.theta)-y).multiply(x))
            e = np.sum(x.T*(x.dot(self.theta) -y))
            print("Erro:", e)
            #theta = theta - (e * (alpha / m))
            self.theta = self.theta - (e*(alpha/m))
            history.append(self.theta)
        return self.theta, history

    def predict(self, x):
        return x.dot(self.theta)




