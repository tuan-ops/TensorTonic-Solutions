def perplexity(prob_distributions, actual_tokens):
    """
    Compute the perplexity of a token sequence given predicted distributions.
    """
    # Write code here
    prob = np.asarray(prob_distributions)
    actual = np.asarray(actual_tokens)
    n = len(actual_tokens)
    re = 0
    for i in range(n):
        re += np.log(prob[i, actual[i]])
    H = -1/n * (re)
    return np.exp(H)