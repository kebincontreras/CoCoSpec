import numpy as np
from matplotlib import pyplot as plt

from src.const.enums import CocoaConditionsEnum
from src.const.paths import res_dir
from src.loading.loading import load_acquisitions_pixels_info
from src.schemas.acquisitions import AcquisitionInfo
from src.schemas.pixel_managers import AcquisitionPixelsInfo
from src.utils.arrays import normalize, min_max
from src.utils.spectra import correct_flat_and_dark, correct_reference_spectrum


def apply_reference_spectrum(
        image: np.ndarray,
        acquisition_pixels: AcquisitionPixelsInfo,
) -> np.ndarray:
    if acquisition_pixels.reference is not None:
        reference_spectrum = acquisition_pixels.reference.get_average_spectrum(image=image)
        image_corrected = correct_reference_spectrum(image, reference_spectrum)
    else:
        image_corrected = image / image.max()
    return image_corrected


def save_reference_spectrum(
        image: np.ndarray,
        acquisition_info: AcquisitionInfo,
        acquisition_pixels: AcquisitionPixelsInfo,
) -> None:
    if acquisition_pixels.reference is not None:
        reference_spectrum = acquisition_pixels.reference.get_average_spectrum(image=image)
        ref_spectra_dir = res_dir() / "reference_spectra"
        filename = f"{acquisition_info.camera_name.value.lower()}.npy"
        filepath = ref_spectra_dir / filename
        np.save(file=filepath, arr=reference_spectrum)


def fill_figure(
        axs,
        col_idx: int,
        image: np.ndarray,
        acquisition_info: AcquisitionInfo,
        acquisition_pixels: AcquisitionPixelsInfo,
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
        selected_spectra = normalize(selected_spectra, axis=1)

        pixels_rows, pixels_cols = acquisition_pixels.get_reference_pixels_positions().T
        axs[0, col_idx].scatter(pixels_cols, pixels_rows)
        axs[1, col_idx].plot(wavelengths, selected_spectra.T)
        axs[1, col_idx].set_title(acquisition_info.camera_name.value.upper())
        axs[1, col_idx].set_ylim(ylim)
        axs[1, col_idx].grid()

    if acquisition_pixels.selected is not None:
        selected_spectra = acquisition_pixels.get_selected_spectra(image=image)
        selected_spectra = min_max(selected_spectra)

        pixels_rows, pixels_cols = acquisition_pixels.get_selected_pixels_positions().T
        axs[0, col_idx].scatter(pixels_cols, pixels_rows)
        axs[2, col_idx].plot(wavelengths, selected_spectra.T)
        axs[2, col_idx].set_title(acquisition_info.camera_name.value.upper())
        axs[2, col_idx].set_ylim(ylim)
        axs[2, col_idx].grid()

    return axs


def main():
    global_pixels_info = load_acquisitions_pixels_info()

    for scene in [19]:
        for cocoa_condition in ["open"]:
            acquisitions_info_options = [
                {
                    "scene": scene,
                    "camera_name": "eos_m50",
                    "format": "jpg",
                    "cocoa_condition": cocoa_condition,
                },
                {
                    "scene": scene,
                    "camera_name": "specim_iq",
                    "format": "envi",
                    "cocoa_condition": cocoa_condition,
                },
                {
                    "scene": scene,
                    "camera_name": "toucan",
                    "format": "npy",
                    "cocoa_condition": cocoa_condition,
                },
                {
                    "scene": scene,
                    "camera_name": "ultris_sr5",
                    "format": "tiff",
                    "cocoa_condition": cocoa_condition,
                },
            ]
            acquisitions_info_list = [AcquisitionInfo(**options) for options in acquisitions_info_options]

            fig, axs_original = plt.subplots(nrows=3, ncols=4, squeeze=False, figsize=(15, 5))
            fig, axs_referenced = plt.subplots(nrows=3, ncols=4, squeeze=False, figsize=(15, 5))
            fig, axs_fielded = plt.subplots(nrows=3, ncols=4, squeeze=False, figsize=(15, 5))

            for idx, acq_info in enumerate(acquisitions_info_list):
                acq_pixels_dict = global_pixels_info[scene][acq_info.camera_name][acq_info.cocoa_condition]
                acq_pixels = AcquisitionPixelsInfo(**acq_pixels_dict)

                image_raw = acq_info.load_image(normalize=False)
                print(
                    f"Loading an image from the {acq_info.camera_name.value.upper()} camera,"
                    f" with shape {image_raw.shape}."
                )

                image_normalized = acq_info.normalize_image(image=image_raw)
                fill_figure(
                    axs_original,
                    col_idx=idx,
                    image=image_normalized,
                    acquisition_info=acq_info,
                    acquisition_pixels=acq_pixels
                )

                image_referenced = apply_reference_spectrum(image_raw, acquisition_pixels=acq_pixels)
                fill_figure(
                    axs_referenced,
                    col_idx=idx,
                    image=image_referenced,
                    acquisition_info=acq_info,
                    acquisition_pixels=acq_pixels
                )

                flat = acq_info.load_flat_field()
                dark = acq_info.load_dark_field()
                image_fielded = correct_flat_and_dark(image_raw, flat_field=flat, dark_field=dark)
                image_fielded = apply_reference_spectrum(image_fielded, acquisition_pixels=acq_pixels)
                fill_figure(
                    axs_fielded,
                    col_idx=idx,
                    image=image_fielded,
                    acquisition_info=acq_info,
                    acquisition_pixels=acq_pixels
                )

            plt.show()
            print()


if __name__ == "__main__":
    main()
