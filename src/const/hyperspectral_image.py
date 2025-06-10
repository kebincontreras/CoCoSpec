from dataclasses import dataclass, replace
from typing import Self

import numpy as np


@dataclass(frozen=True)
class HyperspectralImage:
    image: np.ndarray  # [rows | height, columns | width, channels | bands]
    wavelengths: np.ndarray  # Shape: (channels,)
    rgb_bands: tuple[int, int, int]
    wavelengths_unit: str = "nm"

    @classmethod
    def create(
            cls,
            image: np.ndarray,  # [rows | height, columns | width, channels | bands]
            wavelengths: np.ndarray,  # Shape: (channels,)
            rgb_bands: tuple[int, int, int],
            wavelengths_unit: str = "nm",
    ) -> Self:
        return cls(
            image=image,
            wavelengths=wavelengths,
            rgb_bands=rgb_bands,
            wavelengths_unit=wavelengths_unit,
        )

    def crop_wavelengths(
        self,
        wavelength_min: float = None,
        wavelength_max: float = None,
    ) -> Self:
        mask = np.ones_like(self.wavelengths, dtype=bool)
        if wavelength_min is not None:
            mask &= self.wavelengths >= wavelength_min
        if wavelength_max is not None:
            mask &= self.wavelengths <= wavelength_max
        return replace(self, image=self.image[:, :, mask], wavelengths=self.wavelengths[mask])

    def get_rgb_image(self, normalize: bool = False) -> np.ndarray:
        rgb = self.image[:, :, self.rgb_bands].astype(np.float32)

        if normalize:
            for i in range(3):
                channel = rgb[:, :, i]
                min_val, max_val = channel.min(), channel.max()
                if max_val > min_val:
                    rgb[:, :, i] = (channel - min_val) / (max_val - min_val)

        return np.clip(rgb, 0, 1)

    def plot_rgb(self, axs, normalize: bool = False, title: str = "RGB Image") -> None:
        rgb = self.get_rgb_image(normalize=normalize)
        axs.imshow(rgb)
        axs.set_title(title)
