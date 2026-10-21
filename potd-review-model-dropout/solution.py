import numpy as np


def inverted_dropout(x: np.ndarray, m: np.ndarray, p_keep: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    return (x * m) / p_keep
