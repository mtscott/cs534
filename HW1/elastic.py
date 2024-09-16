import numpy as np

def loss(x, y, beta, el, alpha):
    term1 = 0.5 * np.norm(y - x@beta,2)**2
    term2 = alpha * np.norm(beta,2)**2 + (1-alpha) * np.norm(beta,1)
    return term1 + el * term2

def grad_step(x, y, beta, el, alpha, eta):
    return 0


class ElasticNet:
    def __init__(self, el, alpha, eta, batch, epoch):
        # do something
        return

    def coef(self):
        return 0

    def train(self, x, y):
        return 0

    def predict(self, x):
        return 0
