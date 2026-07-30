def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    # Write code here
    n = len(y_true)
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(n):
        if y_pred[i] != y_true[i]:
            FP += 1
        if y_pred[i] == y_true[i] :
            TP += 1
    return 2 * (TP/ ((2*TP) + 2*FP ))