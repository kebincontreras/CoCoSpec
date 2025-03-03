from enum import Enum

import numpy as np
import scipy.interpolate


class InterpolationMethodsEnum(str, Enum):
    NEAREST = "nearest"
    LINEAR = "linear"
    CUBIC = "cubic"


def generate_mask(tile_indices: np.ndarray, image_shape: tuple) -> np.ndarray:
    mask_tile = np.zeros((*tile_indices.shape, tile_indices.max() + 1), dtype=bool)
    mask_tile[
        np.arange(tile_indices.shape[0])[:, None],
        np.arange(tile_indices.shape[1]),
        tile_indices
    ] = True
    mask_repeats = (image_shape[0] // mask_tile.shape[0], image_shape[1] // mask_tile.shape[1], 1)
    mask = np.tile(mask_tile, reps=mask_repeats)
    return mask


def generate_sparse_image(
        raw_image: np.ndarray,
        mask: np.ndarray,
):
    sparse_image = raw_image[..., None] * mask
    return sparse_image


def interpolate_missing_pixels(
        sparse_channel: np.ndarray,
        mask_channel: np.ndarray,
        interpolation_method: InterpolationMethodsEnum,
):
    x, y = np.meshgrid(np.arange(sparse_channel.shape[0]), np.arange(sparse_channel.shape[1]))
    known_x = x[mask_channel]
    known_y = y[mask_channel]
    known_values = sparse_channel[mask_channel]
    interpolated = scipy.interpolate.griddata(
        points=(known_x, known_y),
        values=known_values,
        xi=(x, y),
        method=interpolation_method,
        fill_value=0.,
    )
    return interpolated


def interpolate_sparse_image(
        sparse_image: np.ndarray,
        mask: np.ndarray,
        interpolation_method: InterpolationMethodsEnum,
):
    demosaicked_image = np.zeros_like(sparse_image, dtype=np.float32)
    for channel_idx in range(demosaicked_image.shape[-1]):
        print(f"Interpolating channel {channel_idx}...")
        demosaicked_image[:, :, channel_idx] = interpolate_missing_pixels(
            sparse_channel=sparse_image[:, :, channel_idx],
            mask_channel=mask[:, :, channel_idx],
            interpolation_method=interpolation_method,
        )
    return demosaicked_image


def demosaic_raw_image(
        raw_image: np.ndarray,
        tile: np.ndarray,
        interpolation_method: InterpolationMethodsEnum,
):
    mask = generate_mask(tile_indices=tile, image_shape=raw_image.shape)
    sparse_image = generate_sparse_image(raw_image=raw_image, mask=mask)
    demosaicked_image = interpolate_sparse_image(
        sparse_image=sparse_image,
        mask=mask,
        interpolation_method=interpolation_method,
    )
    return demosaicked_image
