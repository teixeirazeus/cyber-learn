import numpy as np


def normalizador(entrada):
    """Normaliza uma lista de vetores."""
    dados = copy.deepcopy(entrada)
    for campo in range(len(dados[0])):
        maximo = max([ponto[campo] for ponto in dados])
        minimo = min([ponto[campo] for ponto in dados])
        for i in range(len(dados)):
            dados[i][campo] = (dados[i][campo]-minimo)/(maximo-minimo)
    return dados

class LinearRegression:
    self.minNmax_x = []
    self.minNmax_y = []
    self.theta = []
    def fit(self, x, y):
        for atribute in range(len(x[0])):
            # get min and max
            min_x, max_x = 0,0
            for vector in x:
                if vector[atribute] < min_x: min_x = vector[atribute]
                if vector[atribute] > max_x: max_x = vector[atribute]

            # mean normalization
            for v in range(len(x)):
                v[atribute] = (v[atribute]-min_x)/(max_x-min_x)

            self.minNmax_x.append([min_x, max_x])  # save for predict

        for atribute in range(len(y[0])):
            # get min and max
            min_y, max_y = 0,0
            for vector in y:
                if vector[atribute] < min_y: min_y = vector[atribute]
                if vector[atribute] > max_y: max_y = vector[atribute]

            # mean normalization
            for v in range(len(y)):
                v[atribute] = (v[atribute]-min_y)/(max_x-min_y)

            self.minNmax_y.append([min_x, max_x])  # save for predict

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




