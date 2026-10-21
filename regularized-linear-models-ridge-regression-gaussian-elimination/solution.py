import numpy as np


def gaussian_elimination_solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).copy()
    b = np.asarray(b, dtype=np.float64).copy()

    n = a.shape[0]

    for k in range(n):
        pivot_row = k + int(np.argmax(np.abs(a[k:, k])))
        if pivot_row != k:
            a[[k, pivot_row], :] = a[[pivot_row, k], :]
            b[[k, pivot_row]] = b[[pivot_row, k]]

        pivot = a[k, k]

        for i in range(k + 1, n):
            factor = a[i, k] / pivot
            a[i, k:] -= factor * a[k, k:]
            b[i] -= factor * b[k]

    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(a[i, i + 1:], x[i + 1:])) / a[i, i]

    return x


def ridge_regression_predict(
    input: np.ndarray, target: np.ndarray, lam: float, queries: np.ndarray
) -> np.ndarray:
    X = np.asarray(input, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    Q = np.asarray(queries, dtype=np.float64)

    n, d = X.shape

    Xb = np.hstack([X, np.ones((n, 1), dtype=np.float64)])

    A = Xb.T @ Xb + lam * np.eye(d + 1, dtype=np.float64)
    b = Xb.T @ y

    theta = gaussian_elimination_solve(A, b)

    m = Q.shape[0]
    Qb = np.hstack([Q, np.ones((m, 1), dtype=np.float64)])

    return Qb @ theta
