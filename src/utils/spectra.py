import numpy as np


def correct_flat_and_dark(
        image: np.ndarray,
        flat_field: np.ndarray,
        dark_field: np.ndarray,
) -> np.ndarray:
    image_corrected = (image - dark_field) / (flat_field - dark_field)
    return image_corrected


def correct_reference_spectrum(
        image: np.ndarray,
        spectrum: np.ndarray,
) -> np.ndarray:
    image_corrected = image / spectrum[None, None]
    return image_corrected
