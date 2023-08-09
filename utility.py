import numpy as np

def standardform_to_slackform(standardform_lp):
    (A, b, c) = standardform_lp
    m = len(b)
    n = len(c)
    N = np.arange(1, n+1)
    B = np.arange(n+1, n+m+1)
    v = 0
    return (B, N, A, b, c, v)