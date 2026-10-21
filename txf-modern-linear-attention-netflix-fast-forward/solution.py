import numpy as np


def feature_map(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.maximum(x, 0.0) + 1.0


def linear_attention_row_sums(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    # K' = phi(K)
    Kp = feature_map(K)

    # Z = sum_j K'_j, shape (d,)
    Z = Kp.sum(axis=0)

    # We only need row sums of O, so instead of forming M = K'^T @ V
    # and then Q' @ M, use:
    #   sum_c O_i,c = (Q'_i @ M @ 1) / (Q'_i @ Z)
    #               = (Q'_i @ (K'^T @ (V @ 1))) / (Q'_i @ Z)
    #
    # vsum = V @ 1, shape (N,)
    vsum = V.sum(axis=1)

    # w = K'^T @ vsum = M @ 1, shape (d,)
    w = Kp.T @ vsum

    # Free K' before allocating Q'.
    del Kp

    # Q' = phi(Q)
    Qp = feature_map(Q)

    numerator = Qp @ w
    denominator = Qp @ Z

    return numerator / denominator
