import numpy as np

def r2_score(y_true: list, y_pred: list) -> float:
    """Return the coefficient of determination."""
    # Write code here
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    SS_res = np.sum((y_true - y_pred) ** 2)
    SS_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if SS_res ==  0: return 1.0
    if SS_tot == 0: return 0.0
    R_2 = 1 - (SS_res / SS_tot)
    return float(R_2)