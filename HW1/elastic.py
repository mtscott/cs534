import numpy as np
import numpy.linalg as npla

def loss(x, y, beta, el, alpha):
    term1 = 0.5 * npla.norm(y - x@beta,2)**2
    term2 = alpha * npla.norm(beta,2)**2 + (1-alpha) * npla.norm(beta,1)
    return term1 + el * term2

def grad_step(x, y, beta, el, alpha, eta):
    term1 = 2 * (x @ beta - y) @ x
    term2 = 2 * el * alpha * beta
    # using vectorized code, we can rewrite the prox. gradient
    term3 = el * (1 -  alpha) * np.sign(beta) * np.maximum(np.zeros(len(beta)), np.abs(beta) - el * eta)
    return term1 + term2 + term3


class ElasticNet:
    def __init__(self, el, alpha, eta, batch, epoch):
        self.el = el        # lambda
        self.alpha = alpha
        self.eta = eta
        self.batch = batch
        self.epoch = epoch

    def coef(self):
        return 0

    def train(self, x, y):
        # Example dictionary
        errdict = {'a': 1, 'b': 2, 'c': 3}

        # Append a new key-value pair
        errdict['d'] = 4

        print(errdict)
        return errdict

    def predict(self, x):
        return 0
