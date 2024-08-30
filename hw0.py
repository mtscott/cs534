import numpy as np
rng = np.random.default_rng(12345)

def draw_samples(n):
    """
    
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
    
    """
    return np.linalg.norm(arr) ** 2

def troublemakers(n):
    """
    
    """
    A = np.array([[0.72, 0.2], [0.28, 0.8]])
    vols = np.linalg.matrix_power(A, n) @ np.ones((2,1))
    
    return vols