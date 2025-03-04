from pathlib import Path

import numpy as np
import tifffile as tiff
from matplotlib import pyplot as plt

from src.demosaicking.utils import demosaic_raw_image, InterpolationMethodsEnum
from src.utils import load_project_dir


def run_experiment(exp_config: dict):
    # Perform Demosaicking
    print(f"Raw Image: {exp_config['raw_image_path'].name}")
    raw_image = tiff.imread(exp_config["raw_image_path"])
    tile = np.array(exp_config["tile"])
    interpolation_method = exp_config["interpolation_method"]
    demosaicked_image = demosaic_raw_image(
        raw_image=raw_image,
        tile=tile,
        interpolation_method=interpolation_method,
    )

    # Save demosaicked image (The hyperspectral image cube)
    output_path = Path(exp_config["demosaicked_image_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(file=exp_config["demosaicked_image_path"], arr=demosaicked_image)
    print("Demosaicking complete! Image saved.")
    print()


def visualize_experiment(exp_config: dict):
    # Visualize results
    raw_image = tiff.imread(exp_config["raw_image_path"])
    demosaicked_image = np.load(file=exp_config["demosaicked_image_path"])

    images = [raw_image, demosaicked_image]
    titles = ["Raw", f"Demosaicked ({exp_config['interpolation_method']})"]
    fig, axes = plt.subplots(1, len(images), squeeze=False, figsize=(18, 10))
    axes[0, 0].imshow(images[0])
    axes[0, 0].set_title(titles[0])
    for i in range(1, len(images)):
        image_vis = images[i][:, :, exp_config["rgb_channels"]]
        image_vis = image_vis / image_vis.max()
        axes[0, i].imshow(image_vis)
        axes[0, i].set_title(titles[i])
    plt.show()


def main():
    project_dir = load_project_dir()
    data_dir = project_dir / "data"

    for image_idx in range(1, 19):
        raw_image_path = data_dir / "toucan_demosaic_test" / "raw" / f"cacao2_{image_idx:05}_raw.tiff"
        demosaicked_image_path = data_dir / "toucan_demosaic_test" / "demosaicked" / f"cacao2_{image_idx:05}_raw.npy"

        exp_config = {
            "raw_image_path": raw_image_path,
            "demosaicked_image_path": demosaicked_image_path,
            "tile": [
                [3, 1, 2, 0],
                [6, 8, 4, 7],
                [2, 0, 3, 1],
                [5, 7, 9, 8]
            ],
            "rgb_channels": [5, 3, 1],
            "interpolation_method": "linear",  # Other Possibilities: "nearest", "linear", "cubic"
            "wavenumbers": [431., 479., 515., 567., 611., 666., 719., 775., 820., 877.],
        }

        run_experiment(exp_config=exp_config)
        visualize_experiment(exp_config=exp_config)


if __name__ == "__main__":
    main()
