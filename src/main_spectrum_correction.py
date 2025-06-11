import numpy as np
from matplotlib import pyplot as plt

from src.loading.loading import load_cameras_metadata
from src.schemas.acquisitions import AcquisitionInfo
from src.schemas.cameras import CameraInfo
from src.schemas.pixel_managers import AcquisitionPixelsInfo
from src.utils.arrays import normalize, min_max
from src.utils.spectra import correct_flat_and_dark, correct_reference_spectrum


def fill_figure(
        axs,
        col_idx: int,
        image: np.ndarray,
        acquisition_info: AcquisitionInfo,
        acquisition_pixels: AcquisitionPixelsInfo,
        is_normalize_reference: bool = False,
        xlim: tuple = None,
        ylim: tuple = (-0.1, 1.2),
):
    camera_info = acquisition_info.load_camera_info()

    image_rgb = image[:, :, camera_info.rgb_channels]
    image_rgb = np.clip(image_rgb, a_min=0., a_max=1.)
    axs[0, col_idx].imshow(image_rgb)
    axs[0, col_idx].set_title(acquisition_info.camera_name.value.upper())

    wavelengths = camera_info.load_wavelengths()

    if acquisition_pixels.reference is not None:
        selected_spectra = acquisition_pixels.get_reference_spectra(image=image)
        if is_normalize_reference:
            selected_spectra = normalize(selected_spectra, axis=1)

        pixels_rows, pixels_cols = acquisition_pixels.get_reference_pixels_positions().T
        axs[0, col_idx].scatter(pixels_cols, pixels_rows)
        axs[1, col_idx].plot(wavelengths, selected_spectra.T)
        axs[1, col_idx].set_title(acquisition_info.camera_name.value.upper())
        if xlim is not None:
            axs[1, col_idx].set_xlim([440, 860])
        axs[1, col_idx].set_ylim(ylim)
        axs[1, col_idx].grid()

    if acquisition_pixels.selected is not None:
        selected_spectra = acquisition_pixels.get_selected_spectra(image=image)
        selected_spectra = min_max(selected_spectra)

        pixels_rows, pixels_cols = acquisition_pixels.get_selected_pixels_positions().T
        axs[0, col_idx].scatter(pixels_cols, pixels_rows)
        axs[2, col_idx].plot(wavelengths, selected_spectra.T)
        axs[2, col_idx].set_title(acquisition_info.camera_name.value.upper())
        if xlim is not None:
            axs[2, col_idx].set_xlim([440, 860])
        axs[2, col_idx].set_ylim(ylim)
        axs[2, col_idx].grid()

    return axs


def main():
    cameras_info_list = [CameraInfo(**options) for options in load_cameras_metadata()]
    for scene in [19]:
        for cocoa_condition in ["open", "closed"]:
            print(f"Scene {scene:02}:")
            print(f"Cocoa condition {cocoa_condition}:")

            # Below, we study the pixels composition for each 4 camera acquisitions, per scene and per cocoa_condition
            acquisitions_info_options = [
                {
                    "scene": scene,
                    "camera_name": camera.name,
                    "format": camera.image_format,
                    "cocoa_condition": cocoa_condition,
                }
                for camera in cameras_info_list
                if cocoa_condition in camera.cocoa_conditions
            ]
            acquisitions_info_list = [AcquisitionInfo(**options) for options in acquisitions_info_options]

            fig, axs_original = plt.subplots(nrows=3, ncols=4, squeeze=False, figsize=(15, 5))
            fig, axs_referenced = plt.subplots(nrows=3, ncols=4, squeeze=False, figsize=(15, 5))
            fig, axs_fielded = plt.subplots(nrows=3, ncols=4, squeeze=False, figsize=(15, 5))

            for idx, acq_info in enumerate(acquisitions_info_list):
                acq_pixels = acq_info.load_pixels_info()
                print(f"\tLoading an image from the {acq_info.camera_name.value.upper()} camera.")

                # Load image as it is without any modifications
                image_raw = acq_info.load_image(normalize=False)

                # Normalize the image between [0, 1] to visualize the raw image
                image_normalized = acq_info.normalize_image(image=image_raw)

                # Correct the spectra by a spectral reference pixel, if a Spectralon exists in the scene
                reference_spectrum = acq_pixels.get_reference_spectrum(image=image_raw)
                image_referenced = correct_reference_spectrum(image_raw, spectrum=reference_spectrum)

                # Correct the spectra by the flat and dark field images, if a flat and dark exist for the camera
                flat = acq_info.load_flat_field()
                dark = acq_info.load_dark_field()
                image_fielded = correct_flat_and_dark(image_raw, flat_field=flat, dark_field=dark)
                reference_spectrum = acq_pixels.get_reference_spectrum(image=image_fielded)
                image_fielded = correct_reference_spectrum(image_fielded, spectrum=reference_spectrum)

                # Visualize the figures for each image
                fill_figure(axs_original, col_idx=idx, image=image_normalized, acquisition_info=acq_info,
                            acquisition_pixels=acq_pixels, is_normalize_reference=True)
                fill_figure(axs_referenced, col_idx=idx, image=image_referenced, acquisition_info=acq_info,
                            acquisition_pixels=acq_pixels, is_normalize_reference=True)
                fill_figure(axs_fielded, col_idx=idx, image=image_fielded, acquisition_info=acq_info,
                            acquisition_pixels=acq_pixels, is_normalize_reference=True)

            plt.show()
            print()


if __name__ == "__main__":
    main()
