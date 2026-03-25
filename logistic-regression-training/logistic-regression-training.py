import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    p, n = X.shape
    w = np.zeros(n)
    b = 0
    for i in range (steps):
        L_w = 1/n * (X.T @ (_sigmoid((X @ w + b)) - y))
        L_b = 1/n * np.sum(_sigmoid((X @ w + b)) - y) 
        w = w - lr * L_w
        b = b - lr * L_b
    # Write code here
    return (w, b)