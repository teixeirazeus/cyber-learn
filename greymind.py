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

            minNmax_x.append([min_x, max_x])  # save for predict