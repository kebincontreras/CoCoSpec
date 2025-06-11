import numpy as np
from matplotlib import pyplot as plt

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
        scene_pixels = global_pixels_info[scene]
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
            acquisition_pixels = AcquisitionPixelsInfo(
                **global_pixels_info[scene][acq.camera_name][acq.cocoa_condition])

            image_raw = acq.load_image(normalize=False)
            print(f"Loading an image from the {acq.camera_name.value.upper()} camera with shape {image_raw.shape}.")

            image_normalized = acq.normalize_image(image=image_raw)
            fill_figure(
                axs_orig,
                col_idx=idx,
                image=image_normalized,
                acquisition_info=acq,
                acquisition_pixels=acquisition_pixels
            )

            image_referenced = apply_reference_spectrum(image_raw, acquisition_pixels=acquisition_pixels)
            fill_figure(
                axs_refr,
                col_idx=idx,
                image=image_referenced,
                acquisition_info=acq,
                acquisition_pixels=acquisition_pixels
            )

            flat = acq.load_flat_field()
            dark = acq.load_dark_field()
            image_fielded = correct_flat_and_dark(image_raw, flat_field=flat, dark_field=dark)
            image_fielded = apply_reference_spectrum(image_fielded, acquisition_pixels=acquisition_pixels)
            fill_figure(
                axs_fiel,
                col_idx=idx,
                image=image_fielded,
                acquisition_info=acq,
                acquisition_pixels=acquisition_pixels
            )

        plt.show()
        print()


if __name__ == "__main__":
    main()
