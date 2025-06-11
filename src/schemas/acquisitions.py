from pathlib import Path

import numpy as np
from pydantic import BaseModel

from src.const.enums import CamerasEnum, CocoaConditionsEnum, ImageFormatsEnum, file_extension
from src.const.hyperspectral_image import HyperspectralImage
from src.const.paths import image_filepath, annotations_filepath
from src.loading.loading import load_image, load_acquisitions_pixels_info
from src.schemas.cameras import CameraInfo
from src.schemas.pixel_managers import AcquisitionPixelsInfo


class AcquisitionInfo(BaseModel):
    scene: int
    camera_name: CamerasEnum
    format: ImageFormatsEnum
    cocoa_condition: CocoaConditionsEnum
    bit_depth: int | None = None
    spatial_rotation_order: int = 0

    @property
    def _file_extension(self) -> str:
        return file_extension(image_format=self.format)

    @property
    def _image_filepath(self) -> Path:
        return image_filepath(
            scene=self.scene,
            camera_name=self.camera_name,
            extension=self._file_extension,
            cocoa_condition=self.cocoa_condition,
        )

    @property
    def annotations_filepath(self) -> Path:
        return annotations_filepath(
            scene=self.scene,
            camera_name=self.camera_name,
            cocoa_condition=self.cocoa_condition,
        )

    def load_camera_info(self) -> CameraInfo:
        camera_info = CameraInfo.from_name(camera_name=self.camera_name)
        return camera_info

    def load_pixels_info(self) -> AcquisitionPixelsInfo:
        global_pixels_info = load_acquisitions_pixels_info()
        pixels_info = global_pixels_info[self.scene][self.camera_name][self.cocoa_condition]
        return AcquisitionPixelsInfo(**pixels_info)

    def load_wavelengths(self) -> np.ndarray:
        return self.load_camera_info().load_wavelengths()

    def load_image(
            self,
            normalize: bool = False,
    ) -> np.ndarray:
        image = load_image(
            filepath=self._image_filepath,
            image_format=self.format,
        )

        camera_info = self.load_camera_info()
        image = image.transpose(camera_info.permutation)
        image = np.rot90(image, k=self.spatial_rotation_order, axes=(0, 1))

        if normalize:
            image = self.normalize_image(image=image)

        return image.astype("float32")

    def normalize_image(
            self,
            image: np.ndarray,
    ) -> np.ndarray:
        camera_info = self.load_camera_info()

        if self.bit_depth is None:
            bit_depth = camera_info.bit_depth
        else:
            bit_depth = self.bit_depth

        if bit_depth is None:
            max_val = image.max()
        else:
            max_val = 2 ** bit_depth - 1

        image = image / max_val
        return image

    def load_flat_field(self) -> np.ndarray:
        camera_info = self.load_camera_info()
        if camera_info.flat_field_format is None and camera_info.flat_field_filename is None:
            if self.bit_depth is None:
                bit_depth = camera_info.bit_depth
            else:
                bit_depth = self.bit_depth

            if bit_depth is None:
                image = self.load_image()
                max_val = image.max()
            else:
                max_val = 2 ** bit_depth - 1

            image = np.array(max_val)
        else:
            image = camera_info.load_flat_field()
        return np.asarray(np.maximum(image, 1e-8))

    def load_dark_field(self) -> np.ndarray:
        camera_info = self.load_camera_info()
        image = camera_info.load_dark_field()
        return np.asarray(image)

    def load_hyperspectral_image(
            self,
            normalize: bool = False,
    ) -> HyperspectralImage:
        camera_info = self.load_camera_info()
        return HyperspectralImage.create(
            image=self.load_image(normalize=normalize),
            wavelengths=camera_info.load_wavelengths(),
            wavelengths_unit=camera_info.wavelengths_unit,
            rgb_bands=camera_info.rgb_channels,
        )
