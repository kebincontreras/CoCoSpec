from pathlib import Path

import numpy as np
import tifffile as tiff
from matplotlib import pyplot as plt

from src.demosaicking.utils import demosaic_raw_image, InterpolationMethodsEnum
from src.utils import load_project_dir


def run_experiment(exp_config: dict):
    # Perform Demosaicking
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

    # Visualize results
    images = [raw_image, demosaicked_image]
    titles = ["Raw", "Demosaicked (Linear)"]
    fig, axes = plt.subplots(1, len(images), squeeze=False, figsize=(20, 5))
    axes[0, 0].imshow(images[0])
    axes[0, 0].set_title(titles[0])
    for i in range(1, len(images)):
        image_vis = images[i][:, :, exp_config["rgb_channels"]]
        image_vis = image_vis / image_vis.max()
        axes[0, i].imshow(image_vis)
        axes[0, i].set_title(titles[i])
    plt.show()


def main():
    scene = 1
    filename = "HSI_open"

    project_dir = load_project_dir()
    data_dir = project_dir / "data"
    raw_image_path = data_dir / "scenes" / f"Scene_{scene}" / "Toucan" / f"{filename}.tiff"
    demosaicked_image_path = data_dir / "demosaicked" / f"Scene_{scene}" / "Toucan" / f"{filename}.npy"

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
        "interpolation_method": "linear",  # Possibilities: "nearest", "linear", "cubic"
        "wavenumbers": [431., 479., 515., 567., 611., 666., 719., 775., 820., 877.],
    }

    run_experiment(exp_config=exp_config)


if __name__ == "__main__":
    main()
