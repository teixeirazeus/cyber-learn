import numpy as np


def norm(entrada):
    """Normaliza uma lista de vetores."""
    maxNmin = []
    for campo in range(len(dados[0])):
        # get min and max
        min, max = 0,0
        for vector in dados:
            if vector[atribute] < min: min = vector[atribute]
            if vector[atribute] > max: max = vector[atribute]
        # update
        for i in range(len(dados)):
            dados[i][campo] = (dados[i][campo]-min)/(max-min)
        maxNmin.append([min,max])
    return maxNmin

class LinearRegression:
    self.theta = []

    def cost(self, x, y, theta):
        m = y.shape[0] # number of samples
        h = x.dot(theta) - y;
        return (1/(2*m)) * np.sum(h**2)

    def train(self, x, y, alpha, iterations):
        # alpha is the learning rate
        if len(self.theta) == 0: self.theta = np.array([0 for atribute in x]).T
        m = y.shape[0]  # number of samples
        history = []
        for i in range(iterations):
            # e = sum(((X * theta) - y). * X);
            e = np.sum((x.dot(theta)-y).multiply(x))
            #theta = theta - (e * (alpha / m))
            self.theta = self.theta - (e*(alpha/m))
            history.append(self.theta)
        return self.theta, history




