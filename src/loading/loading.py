import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
import spectral
import tifffile as tiff

from src.const.enums import CamerasEnum, ImageFormatsEnum
from src.const.paths import res_dir


def get_item(items_list: dict, key: str, value: Any) -> Any:
    for item in items_list:
        if item[key] == value:
            return item
    raise ValueError(f"Item not found in list.")


def load_from_json(filepath: Path) -> dict:
    with open(file=f"{filepath}", mode="r") as file_opened:
        content = json.load(file_opened)
    return content


def load_wavelengths(
        camera_name: CamerasEnum,
) -> np.ndarray:
    wavelengths_dir = res_dir() / "wavelengths"
    wl_filename = f"{camera_name.value.lower()}.csv"
    wl_filepath = wavelengths_dir / wl_filename
    wavelengths = pd.read_csv(wl_filepath)["wavelengths (nm)"].to_numpy()
    return wavelengths


def load_image(
        filepath: Path,
        image_format: ImageFormatsEnum,
) -> np.ndarray:
    if image_format == ImageFormatsEnum.JPG:
        image = Image.open(fp=f"{filepath}")
        image_arr = np.array(image)

    elif image_format in [ImageFormatsEnum.ENVI, ImageFormatsEnum.HDR]:
        image = spectral.open_image(file=f"{filepath}")
        image_arr = image.load()

    elif image_format in [ImageFormatsEnum.NPY, ImageFormatsEnum.NUMPY]:
        image_arr = np.load(file=f"{filepath}")

    elif image_format in [ImageFormatsEnum.TIFF, ImageFormatsEnum.TIF]:
        image_arr = tiff.imread(f"{filepath}")

    else:
        raise ValueError(f"Image format '{image_format}' is not supported yet.")

    return image_arr


def main():
    wavelengths = load_wavelengths(camera_name=CamerasEnum.EOS_M50)
    print(wavelengths)


if __name__ == "__main__":
    main()
