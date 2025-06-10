import numpy as np
from matplotlib import pyplot as plt

from src.schemas.acquisitions import AcquisitionInfo


def load_pixels_info() -> dict:
    pixels_info = {
        19: {
            "eos_m50": {
                "reference_spectrum": None,
                "selected_positions": [
                    [2292, 3554],
                ],
            },
            "specim_iq": {
                "reference_spectrum": [
                    [288, 76],
                ],
                "selected_positions": [
                    [207, 216],
                ],
            },
            "toucan": {
                "reference_spectrum": [
                    [197, 1357],
                ],
                "selected_positions": [
                    [1162, 1346],
                ],
            },
            "ultris_sr5": {
                "reference_spectrum": [
                    [2, 104],
                ],
                "selected_positions": [
                    [162, 108],
                ],
            },
        }
    }
    return pixels_info


def main():
    pixels_info = load_pixels_info()

    for scene in [19]:
        scene_pixels = pixels_info[scene]
        options_list = [
            {
                "scene": scene,
                "camera_name": "eos_m50",
                "format": "jpg",
                "cocoa_condition": "open",
            },
            {
                "scene": scene,
                "camera_name": "specim_iq",
                "format": "envi",
                "cocoa_condition": "open",
            },
            {
                "scene": scene,
                "camera_name": "toucan",
                "format": "npy",
                "cocoa_condition": "open",
            },
            {
                "scene": scene,
                "camera_name": "ultris_sr5",
                "format": "tiff",
                "cocoa_condition": "open",
            },
        ]
        acquisitions_list = [AcquisitionInfo(**options) for options in options_list]

        fig, axs_orig = plt.subplots(nrows=3, ncols=4, squeeze=False, figsize=(15, 5))
        fig, axs_refr = plt.subplots(nrows=3, ncols=4, squeeze=False, figsize=(15, 5))
        fig, axs_fiel = plt.subplots(nrows=3, ncols=4, squeeze=False, figsize=(15, 5))

        for idx, acq in enumerate(acquisitions_list):
            image_raw = acq.load_image(normalize=False)
            print(f"Loading an image from the {acq.camera_name.value.upper()} camera with shape {image_raw.shape}.")

            image_normalized = acq.normalize_image(image=image_raw)
            fill_figure(axs_orig, col_idx=idx, image=image_normalized, acquisition_info=acq, scene_pixels=scene_pixels)

            image_referenced = apply_reference_spectrum(image_raw, acquisition_info=acq, scene_pixels=scene_pixels)
            fill_figure(axs_refr, col_idx=idx, image=image_referenced, acquisition_info=acq, scene_pixels=scene_pixels)

            image_fielded = apply_flat_and_dark(image_raw, acquisition_info=acq)
            image_fielded = apply_reference_spectrum(image_fielded, acquisition_info=acq, scene_pixels=scene_pixels)
            fill_figure(axs_fiel, col_idx=idx, image=image_fielded, acquisition_info=acq, scene_pixels=scene_pixels)

        plt.show()
        print()


def apply_reference_spectrum(
        image: np.ndarray,
        acquisition_info: AcquisitionInfo,
        scene_pixels,
) -> np.ndarray:
    reference_spectrum = scene_pixels[acquisition_info.camera_name]["reference_spectrum"]
    if reference_spectrum is not None:
        pixels_rows, pixels_cols = np.array(reference_spectrum).T
        selected_spectra = image[pixels_rows, pixels_cols, :]
        reference_spectrum = np.mean(selected_spectra, axis=0)
        image_corrected = correct_reference_spectrum(image, reference_spectrum)
    else:
        image_corrected = image / image.max()
    return image_corrected


def apply_flat_and_dark(
        image: np.ndarray,
        acquisition_info: AcquisitionInfo,
) -> np.ndarray:
    flat = acquisition_info.load_flat_field()
    dark = acquisition_info.load_dark_field()
    image_corrected = correct_flat_and_dark(image, flat, dark)
    return image_corrected


def correct_reference_spectrum(
        image: np.ndarray,
        spectrum: np.ndarray,
) -> np.ndarray:
    image_corrected = image / spectrum[None, None]
    return image_corrected


def correct_flat_and_dark(
        image: np.ndarray,
        flat: np.ndarray,
        dark: np.ndarray,
) -> np.ndarray:
    image_corrected = (image - dark) / (flat - dark)
    return image_corrected


def min_max(
        array: np.ndarray,
) -> np.ndarray:
    min_vals = array.min(axis=1, keepdims=True)
    max_vals = array.max(axis=1, keepdims=True)
    array_min_maxed = (array - min_vals) / (max_vals - min_vals)
    return array_min_maxed


def normalize(
        array: np.ndarray,
) -> np.ndarray:
    max_vals = array.max(axis=1, keepdims=True)
    array_normalized = array / max_vals
    return array_normalized


def fill_figure(
        axs,
        col_idx: int,
        image: np.ndarray,
        acquisition_info: AcquisitionInfo,
        scene_pixels: dict,
        ylim: tuple = (-0.1, 1.2),
):
    camera_info = acquisition_info.load_camera_info()

    image_rgb = image[:, :, camera_info.rgb_channels]
    image_rgb = np.clip(image_rgb, a_min=0., a_max=1.)
    axs[0, col_idx].imshow(image_rgb)
    axs[0, col_idx].set_title(acquisition_info.camera_name.value.upper())

    wavelengths = camera_info.load_wavelengths()

    reference_spectrum = scene_pixels[acquisition_info.camera_name]["reference_spectrum"]
    if reference_spectrum is not None:
        pixels_rows, pixels_cols = np.array(reference_spectrum).T
        selected_spectra = image[pixels_rows, pixels_cols, :]
        selected_spectra = normalize(selected_spectra)

        axs[0, col_idx].scatter(pixels_cols, pixels_rows)
        axs[1, col_idx].plot(wavelengths, selected_spectra.T)
        axs[1, col_idx].set_title(acquisition_info.camera_name.value.upper())
        axs[1, col_idx].set_ylim(ylim)
        axs[1, col_idx].grid()

    selected_positions = scene_pixels[acquisition_info.camera_name]["selected_positions"]
    if selected_positions is not None:
        pixels_rows, pixels_cols = np.array(selected_positions).T
        selected_spectra = image[pixels_rows, pixels_cols, :]
        selected_spectra = min_max(selected_spectra)

        axs[0, col_idx].scatter(pixels_cols, pixels_rows)
        axs[2, col_idx].plot(wavelengths, selected_spectra.T)
        axs[2, col_idx].set_title(acquisition_info.camera_name.value.upper())
        axs[2, col_idx].set_ylim(ylim)
        axs[2, col_idx].grid()

    return axs


if __name__ == "__main__":
    main()
