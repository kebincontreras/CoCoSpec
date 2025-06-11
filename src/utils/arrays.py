import numpy as np


def normalize(
        array: np.ndarray,
) -> np.ndarray:
    max_vals = array.max(axis=1, keepdims=True)
    array_normalized = array / max_vals
    return array_normalized


def min_max(
        array: np.ndarray,
) -> np.ndarray:
    min_vals = array.min(axis=1, keepdims=True)
    max_vals = array.max(axis=1, keepdims=True)
    array_min_maxed = (array - min_vals) / (max_vals - min_vals)
    return array_min_maxed
