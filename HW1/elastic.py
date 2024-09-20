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
        self.beta = None    # Secret tool that will help us later

    def coef(self):
        return self.beta

    def train(self, x, y):
        n, p = np.shape(x)
        rng = np.random.default_rng()

        # combine x and y and shuffle so labels stay with data
        xy = np.c_[np.array(x),np.array(y)]

        # Example dictionary
        errdict = {}

        # Initial Guess is M-P pseudoinverse 
        #U, D, V = npla.svd(x)
        # dinv = D / (D**2 + self.alpha)
        #y = V.T @dinv @ U.T
        beta = npla.pinv(x)@y

        for i in range(self.epoch + 1):
            # Shuffle Data 
            rng.shuffle(xy)

            for start in range(0, n+1, self.batch):

                stop = start + self.batch 
                # Batch Data
                xmini = xy[start:stop, :-1]
                ymini = xy[start:stop, -1:]

                beta -= self.eta * grad_step(xmini,ymini,beta, self.el,self.alpha,self.eta) / n

            errdict[i] = loss(x,y, beta, self.el, self.alpha)

        self.beta = beta

        return errdict

    def predict(self, x):
        #return x
        return x @ self.beta 
