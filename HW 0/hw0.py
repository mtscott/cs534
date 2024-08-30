import numpy as np
rng = np.random.default_rng(12345)

def draw_samples(n):
    """
    We have a trinary random variabel X = {-1,0,1}, such that
            P(X = 1) = 0.25
            P(X = 0) = 0.45
            P(X = -1) = 0.30

    draw_samples takes n samples from X and outputs them.

    input:  n - number of draws from sample X

    output: out - an np array of n samples from distribution X
    """
    out = np.zeros(n)
    for i in range(n):
        rand = np.random.uniform()
        if rand < 0.25:
            out[i] = 1.0
        elif rand < 0.70:
            out[i] = 0.0
        else:
            out[i] = -1.0
    return out

def sum_squares(arr):
    """
    Given a np array, arr, we compute the sum of the squares of this array,
    i.e. [1,2,3,] -> 1^2 + 2^2 + 3^2 = 14

    Mathematically, this is equivalent to the 2-norm (Euclidean norm) squared.
    """
    return np.linalg.norm(arr) ** 2

def troublemakers(n):
    """
    troublemakers computes the volumes inswide the cups after n cycles of mixing

    input:  n - number of cycles of mixing and pouring

    output: vols - an np array, [vol of liquid in cup A, col of liquid in cup S]
    """
    # This is computed by assigning s_int = 0.35 a + s, then substituting s_int back into cup A and cup S
    A = np.array([[0.72, 0.2], [0.28, 0.8]])

    vols = np.linalg.matrix_power(A, n) @ np.ones((2,1))
    
    return vols