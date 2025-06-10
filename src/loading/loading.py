"""Module that takes a file path and loads an object."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import spectral
import tifffile as tiff
from PIL import Image

from src.const.enums import CamerasEnum, ImageFormatsEnum, CocoaConditionsEnum


def load_from_json(filepath: Path) -> dict | list[dict]:
    with open(file=f"{filepath}", mode="r") as file_opened:
        content = json.load(file_opened)
    return content


def get_item(items_list: dict, key: str, value: Any) -> Any:
    for item in items_list:
        if item[key] == value:
            return item
    raise ValueError(f"Item not found in list.")


def load_wavelengths(
        filepath: Path,
        wavelength_min: float = None,
        wavelength_max: float = None,
) -> np.ndarray:
    wavelengths = pd.read_csv(filepath)["wavelengths (nm)"].to_numpy()
    wavelengths = crop_wavelengths(wavelengths=wavelengths, wavelength_min=wavelength_min,
                                   wavelength_max=wavelength_max)
    return wavelengths


def load_image(
        filepath: Path,
        image_format: ImageFormatsEnum,
) -> np.ndarray:
    if image_format == ImageFormatsEnum.JPG:
        image = Image.open(fp=f"{filepath}").convert(mode='RGB')
        image_arr = np.array(image)[:, :, ::-1]

    elif image_format in [ImageFormatsEnum.ENVI, ImageFormatsEnum.HDR]:
        image = spectral.open_image(file=f"{filepath}")
        image_arr = image.load()

    elif image_format in [ImageFormatsEnum.NPY, ImageFormatsEnum.NUMPY]:
        image_arr = np.load(file=f"{filepath}")

    elif image_format in [ImageFormatsEnum.TIFF, ImageFormatsEnum.TIF]:
        image_arr = tiff.imread(f"{filepath}")

    else:
        raise ValueError(f"Image format '{image_format}' is not supported yet.")

    return np.asarray(image_arr)


def crop_wavelengths(
        wavelengths: np.ndarray,
        wavelength_min: float = None,
        wavelength_max: float = None,
) -> np.ndarray:
    mask = np.ones_like(wavelengths, dtype=bool)
    if wavelength_min is not None:
        mask &= wavelengths >= wavelength_min
    if wavelength_max is not None:
        mask &= wavelengths <= wavelength_max
    return wavelengths[mask]
