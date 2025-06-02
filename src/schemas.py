from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel

from src.const.enums import CamerasEnum, CocoaConditionsEnum, ImageFormatsEnum
from src.const.paths import image_filepath, cameras_config_path, annotations_filepath
from src.loading.loading import load_wavelengths, load_image, load_from_json, get_item


class CameraInfo(BaseModel):
    name: CamerasEnum
    rgb_channels: tuple
    bit_depth: int
    permutation: tuple = (0, 1, 2)  # must transpose to (rows, columns, channels)

    def load_wavelengths(self) -> np.ndarray:
        return load_wavelengths(camera_name=self.name)


class AcquisitionInfo(BaseModel):
    scene: int
    camera_name: CamerasEnum
    format: ImageFormatsEnum
    cocoa_condition: CocoaConditionsEnum
    spatial_rotation_order: int = 0

    @property
    def camera_info(self) -> CameraInfo:
        cameras_list = load_from_json(filepath=cameras_config_path())
        camera_info = get_item(items_list=cameras_list, key="name", value=self.camera_name)
        return CameraInfo(**camera_info)

    @property
    def file_extension(self) -> ImageFormatsEnum:
        if self.format == ImageFormatsEnum.ENVI:
            extension = ImageFormatsEnum.HDR
        else:
            extension = self.format

        if isinstance(extension, Enum):
            extension = extension.value.lower()

        return extension

    @property
    def image_filepath(self) -> Path:
        return image_filepath(
            scene=self.scene,
            camera_name=self.camera_name,
            extension=self.file_extension,
            cocoa_condition=self.cocoa_condition,
        )

    @property
    def annotations_filepath(self) -> Path:
        return annotations_filepath(
            scene=self.scene,
            camera_name=self.camera_name,
            cocoa_condition=self.cocoa_condition,
        )

    @property
    def wavelengths(self) -> np.ndarray:
        return load_wavelengths(camera_name=self.camera_name)

    def load_image(
            self,
            normalize: bool = False,
    ) -> np.ndarray:
        image = load_image(
            filepath=self.image_filepath,
            image_format=self.format,
        )

        camera_info = self.camera_info
        image = image.transpose(camera_info.permutation)
        image = np.rot90(image, k=self.spatial_rotation_order, axes=(0, 1))

        if normalize:
            max_val = 2 ** camera_info.bit_depth - 1
            image = image / max_val

        return image.astype("float32")


def main():
    options_list = [
        {
            "scene": 1,
            "camera_name": "eos_m50",
            "format": "jpg",
            "cocoa_condition": "open",
        },
        {
            "scene": 1,
            "camera_name": "specim_iq",
            "format": "envi",
            "cocoa_condition": "open",
        },
        {
            "scene": 1,
            "camera_name": "toucan",
            "format": "npy",
            "cocoa_condition": "open",
        },
        {
            "scene": 1,
            "camera_name": "ultris_sr5",
            "format": "tiff",
            "cocoa_condition": "open",
        },
    ]
    acquisitions_list = [AcquisitionInfo(**options) for options in options_list]

    fig, axs = plt.subplots(nrows=1, ncols=4, squeeze=False, figsize=(15, 5))
    for idx, acq in enumerate(acquisitions_list):
        image = acq.load_image(normalize=True)
        camera = acq.camera_info
        print(f"Loading an image from the {acq.camera_name.value.upper()} camera with shape {image.shape}.")
        print(f"Max = {image.max()}.")
        axs[0, idx].imshow(image[:, :, camera.rgb_channels])
        axs[0, idx].set_title(acq.camera_name.value.upper())
    plt.show()

if __name__ == "__main__":
    main()
