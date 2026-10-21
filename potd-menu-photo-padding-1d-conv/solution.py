import numpy as np


def conv1d(x: np.ndarray, kernel: np.ndarray, mode: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)

    if mode == "same":
        pad = kernel.shape[0] // 2
        x = np.pad(x, (pad, pad), mode="constant", constant_values=0.0)
    elif mode != "valid":
        raise ValueError("mode must be 'valid' or 'same'")
        
    return np.correlate(x, kernel, mode="valid")
