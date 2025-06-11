from typing import Self

import numpy as np
from pydantic import BaseModel


class PixelsList(BaseModel):
    positions: list[list[int, int]]
    labels: list[str]
    categories: list[str]

    def get_unique_categories(self) -> set:
        return set(self.categories)

    def get_category_items_list(self, category: str) -> list[int]:
        return [idx for idx, cat in enumerate(self.categories) if cat == category]

    def filter_positions_by_category(self, category: str) -> list[list[int, int]]:
        indices = self.get_category_items_list(category=category)
        return [self.positions[idx] for idx in indices]

    def filter_positions_by_category_as_array(self, category: str) -> np.ndarray:
        return np.array(self.filter_positions_by_category(category=category))

    def filter_labels_by_category(self, category: str) -> list[str]:
        indices = self.get_category_items_list(category=category)
        return [self.labels[idx] for idx in indices]

    def filter_by_category(self, category: str) -> Self:
        indices = self.get_category_items_list(category=category)
        return PixelsList(
            positions=[self.positions[idx] for idx in indices],
            labels=[self.labels[idx] for idx in indices],
            categories=[self.categories[idx] for idx in indices]
        )

    def get_positions_array(self) -> np.ndarray:
        return np.array(self.positions)

    def get_spectra(self, image: np.ndarray) -> np.ndarray:
        rows, cols = self.get_positions_array().T
        return image[rows, cols, :]


class PixelsNeighborhood(BaseModel):
    center_position: list  # [row, col]
    size: int  # distance from center

    def get_center_position_array(self) -> np.ndarray:
        return np.array(self.center_position)

    def get_center_spectrum(self, image: np.ndarray) -> np.ndarray:
        return image[self.center_position[0], self.center_position[1], :]

    def get_average_spectrum(self, image: np.ndarray) -> np.ndarray:
        rows, cols = self.get_positions_array()
        spectra = image[rows, cols, :]
        return spectra.mean(axis=0)

    def get_spectrum(self, image: np.ndarray, typ: str) -> np.ndarray:
        if typ == "center":
            ref_spectrum = self.get_center_spectrum(image=image)
        elif typ == "average":
            ref_spectrum = self.get_average_spectrum(image=image)
        else:
            raise ValueError()
        return ref_spectrum

    @property
    def positions(self) -> list[list[int, int]]:
        """
        Returns the positions of all the pixels in the neighborhood as a list of N positions (i.e., list([row, col])).
        """
        return self.get_positions_array().tolist()

    def get_positions_array(self) -> np.ndarray:
        """
        Returns the positions of all the pixels in the neighborhood as an array of shape N x 2 (i.e., N x [row, col]).
        """
        rows = np.arange(self.center_position[0] - self.size, self.center_position[0] + self.size + 1)
        cols = np.arange(self.center_position[1] - self.size, self.center_position[1] + self.size + 1)
        grid_rows, grid_cols = np.meshgrid(rows, cols, indexing='ij')
        positions_array = np.stack([grid_rows.ravel(), grid_cols.ravel()]).T
        return positions_array

    def get_spectra(self, image: np.ndarray) -> np.ndarray:
        rows, cols = self.get_positions_array().T
        return image[rows, cols, :]


class AcquisitionPixelsInfo(BaseModel):
    reference_spectra: PixelsNeighborhood | PixelsList | None
    selected_spectra: PixelsNeighborhood | PixelsList | None
