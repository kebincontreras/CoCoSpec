from pathlib import Path
from typing import Self

import numpy as np
from pydantic import BaseModel

from src.const.enums import CamerasEnum, ImageFormatsEnum, file_extension, CocoaConditionsEnum
from src.const.paths import wavelengths_dir, flat_field_filepath, flat_fields_dir, dark_field_filepath, dark_fields_dir
from src.loading.loading import get_item, load_wavelengths, load_image, load_cameras_metadata


class CameraInfo(BaseModel):
    name: CamerasEnum
    default_bands: tuple[int, int, int]
    image_shape: tuple[int, int, int]  # [rows | height, columns | width, channels | bands]
    image_format: ImageFormatsEnum
    cocoa_conditions: list[CocoaConditionsEnum]
    bit_depth: int | None = None
    permutation: tuple = (0, 1, 2)  # must transpose to (rows, columns, channels)
    wavelengths_unit: str = "nm"
    flat_field_filename: str | None = None
    flat_field_format: ImageFormatsEnum | None = None
    dark_field_filename: str | None = None
    dark_field_format: ImageFormatsEnum | None = None

    @property
    def _wavelengths_filepath(self) -> Path:
        wl_filename = f"{self.name.value.lower()}.csv"
        wl_filepath = wavelengths_dir() / wl_filename
        return wl_filepath

    @property
    def _flat_field_filepath(self) -> Path:
        if self.flat_field_filename is None:
            filepath = flat_field_filepath(
                camera_name=self.name,
                extension=file_extension(image_format=self.flat_field_format),
            )
        else:
            filepath = flat_fields_dir() / self.flat_field_filename
        return filepath

    @property
    def _dark_field_filepath(self) -> Path:
        if self.dark_field_filename is None:
            filepath = dark_field_filepath(
                camera_name=self.name,
                extension=file_extension(image_format=self.dark_field_format),
            )
        else:
            filepath = dark_fields_dir() / self.dark_field_filename
        return filepath

    @classmethod
    def from_name(cls, camera_name: CamerasEnum) -> Self:
        cameras_metadata = load_cameras_metadata()
        camera_info = get_item(items_list=cameras_metadata, key="name", value=camera_name)
        return cls(**camera_info)

    def load_wavelengths(
            self,
            wavelength_min: float = None,
            wavelength_max: float = None,
    ) -> np.ndarray:
        wl_filepath = self._wavelengths_filepath
        return load_wavelengths(filepath=wl_filepath, wavelength_min=wavelength_min, wavelength_max=wavelength_max)

    def load_flat_field(self) -> np.ndarray:
        if self.flat_field_format is None and self.flat_field_filename is None:
            norm_val = 1.
            image = np.array(norm_val)
        else:
            image = load_image(
                filepath=self._flat_field_filepath,
                image_format=self.flat_field_format,
            )
        return image

    def load_dark_field(self) -> np.ndarray:
        if self.dark_field_format is None and self.dark_field_filename is None:
            image = np.array(0.)
        else:
            image = load_image(
                filepath=self._dark_field_filepath,
                image_format=self.dark_field_format,
            )
        return image
