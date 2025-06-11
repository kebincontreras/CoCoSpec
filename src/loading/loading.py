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


def load_campaign_metadata():
    camera_to_metadata_mapper = {
        CamerasEnum.EOS_M50: {
            "format": ImageFormatsEnum.JPG,
            "cocoa_conditions": [CocoaConditionsEnum.OPEN]
        },
        CamerasEnum.SPECIM_IQ: {
            "format": ImageFormatsEnum.ENVI,
            "cocoa_conditions": [CocoaConditionsEnum.OPEN, CocoaConditionsEnum.CLOSED]
        },
        CamerasEnum.TOUCAN: {
            "format": ImageFormatsEnum.NPY,
            "cocoa_conditions": [CocoaConditionsEnum.OPEN, CocoaConditionsEnum.CLOSED]
        },
        CamerasEnum.ULTRIS_SR5: {
            "format": ImageFormatsEnum.TIFF,
            "cocoa_conditions": [CocoaConditionsEnum.OPEN, CocoaConditionsEnum.CLOSED]
        },
    }
    return camera_to_metadata_mapper


def load_yolo_annotations_fixed(annotation_path, img_width, img_height):
    boxes = []
    try:
        with open(annotation_path, "r") as file:
            lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                label = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])
                x_min = int((x_center - width / 2) * img_width)
                y_min = int((y_center - height / 2) * img_height)
                x_max = int((x_center + width / 2) * img_width)
                y_max = int((y_center + height / 2) * img_height)
                boxes.append((x_min, y_min, x_max, y_max, label))
            except ValueError:
                continue
    except FileNotFoundError:
        print(f"❌ Annotation file not found -> {annotation_path}")
    return boxes


def extract_spectral_signatures(hsi_data, boxes, num_bands):
    signatures = {0: [], 1: [], 2: []}
    for (x_min, y_min, x_max, y_max, label) in boxes:
        label = int(label)
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(hsi_data.shape[1], x_max), min(hsi_data.shape[0], y_max)
        if x_min < x_max and y_min < y_max:
            region = hsi_data[y_min:y_max, x_min:x_max, :]
            mean_spectrum = np.mean(region, axis=(0, 1))
            if mean_spectrum.shape[0] == num_bands:
                signatures[label].append(mean_spectrum)
    for label in signatures:
        signatures[label] = np.mean(signatures[label], axis=0) if signatures[label] else np.zeros(num_bands)
    return signatures


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


def load_pixels_info() -> dict:
    pixels_info = {
        19: {
            "eos_m50": {
                "reference_spectra": None,
                "selected_positions": [
                    [2292, 3554],
                ],
            },
            "specim_iq": {
                "reference_spectra": [
                    [288, 76],
                ],
                "selected_positions": [
                    [207, 216],
                ],
            },
            "toucan": {
                "reference_spectra": [
                    [197, 1357],
                ],
                "selected_positions": [
                    [1162, 1346],
                ],
            },
            "ultris_sr5": {
                "reference_spectra": [
                    [2, 104],
                ],
                "selected_positions": [
                    [162, 108],
                ],
            },
        }
    }
    return pixels_info
