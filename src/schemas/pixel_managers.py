from typing import Self

import numpy as np
from pydantic import BaseModel


class PixelsList(BaseModel):
    positions: list[list[int, int]]
    labels: list[str]
    categories: list[str]

    @property
    def positions_array(self) -> np.ndarray:
        return np.array(self.positions)

    @property
    def unique_categories(self) -> set:
        return set(self.categories)

    def get_arg_category(self, category: str) -> list[int]:
        return [idx for idx, cat in enumerate(self.categories) if cat == category]

    def filter_positions_by_category(self, category: str) -> list[list[int, int]]:
        indices = self.get_arg_category(category=category)
        return [self.positions[idx] for idx in indices]

    def filter_positions_by_category_as_array(self, category: str) -> np.ndarray:
        return np.array(self.filter_positions_by_category(category=category))

    def filter_labels_by_category(self, category: str) -> list[str]:
        indices = self.get_arg_category(category=category)
        return [self.labels[idx] for idx in indices]

    def filter_by_category(self, category: str) -> Self:
        indices = self.get_arg_category(category=category)
        return PixelsList(
            positions=[self.positions[idx] for idx in indices],
            labels=[self.labels[idx] for idx in indices],
            categories=[self.categories[idx] for idx in indices]
        )


class PixelsNeighborhood(BaseModel):
    center_pos: list  # [row, col]
    size: int  # distance from center
    factor: float

    @property
    def center_pos_array(self) -> np.ndarray:
        return np.array(self.center_pos)

    @property
    def get_grid_pos(self):
        rows = np.arange(self.center_pos[0] - self.size, self.center_pos[0] + self.size + 1)
        cols = np.arange(self.center_pos[1] - self.size, self.center_pos[1] + self.size + 1)
        rows_grid, cols_grid = np.meshgrid(rows, cols, indexing='ij')
        grid_pos = np.stack([rows_grid.ravel(), cols_grid.ravel()]).T
        return grid_pos

    def get_center_spectrum(self, image: np.ndarray) -> np.ndarray:
        return image[self.center_pos[0], self.center_pos[1], :]

    def get_average_spectrum(self, image: np.ndarray) -> np.ndarray:
        rows, cols = self.get_grid_pos.T
        spectra = image[rows, cols, :]
        return spectra.mean(axis=0)

    def get_spectrum(self, image: np.ndarray, typ: str, apply_factor: bool = 1.) -> np.ndarray:
        if typ == "center":
            wr = self.get_center_spectrum(image=image)
        elif typ == "average":
            wr = self.get_average_spectrum(image=image)
        else:
            raise ValueError()
        if apply_factor:
            wr = wr / self.factor
        return wr


class AcquisitionPixelsInfo(BaseModel):
    reference_spectra: PixelsNeighborhood | PixelsList | None
    selected_spectra: PixelsNeighborhood | PixelsList | None
