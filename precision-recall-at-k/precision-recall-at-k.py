def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Write code here
    curr = recommended[:k]
    x = 0
    for i in range(len(curr)):
        if curr[i] in relevant:
            x += 1
    return [x/k, x/len(relevant)]