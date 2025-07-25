from matplotlib import pyplot as plt

from src.schemas.acquisitions import AcquisitionInfo


def main():
    for scene in range(1, 20):
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

        fig, axs = plt.subplots(nrows=1, ncols=4, squeeze=False, figsize=(15, 5))
        for idx, acq_info in enumerate(acquisitions_list):
            image = acq_info.load_image(normalize=True)
            camera = acq_info.load_camera_info()
            print(f"Loading an image from the {acq_info.camera_name.value.upper()} camera with shape {image.shape}.")
            print(f"Max = {image.max()}.")
            axs[0, idx].imshow(image[:, :, camera.default_bands])
            axs[0, idx].set_title(acq_info.camera_name.value.upper())
        plt.show()


if __name__ == "__main__":
    main()
