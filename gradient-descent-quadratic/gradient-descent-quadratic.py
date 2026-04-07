def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    # Write code here
    x = 0
    for i in range(steps):
        x = x0 - lr * (2 * a * x + b)
        x0 = x
    return x