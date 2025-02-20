from pathlib import Path
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff


def main():
    filepath = r"C:\Users\USUARIO\Downloads\Toucan\cacao2_00012_raw.tiff"
    filepath = Path(filepath)
    image = tiff.imread(filepath)

    print(
        f"Name: {filepath.name}\n"
        f"Shape: {image.shape}\n"
        f"Maximum Value: {image.max()}\n"
        f"Bit Depth: {int(np.ceil(np.log(image.max()) / np.log(2)))}\n"
    )

    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(16, 4), squeeze=False)

    zooms = [
        (0, 0, image.shape[1], image.shape[0]),  # Full image
        (300, 550, 800, 1050),  # Medium zoom
        (500, 750, 550, 800),  # More zoom
        (518.5, 780.5, 522.5, 784.5),  # 4x4 Mosaic zoom
    ]

    titles = ["Full Raw Image", "Medium Zoom", "Repeated 4x4 Mosaic\nPattern Zoom", "Single 4x4 Mosaic Zoom"]

    for i, (x1, y1, x2, y2) in enumerate(zooms):
        axs[0, i].imshow(image)
        axs[0, i].set_xlim(x1, x2)
        axs[0, i].set_ylim(y2, y1)
        axs[0, i].set_title(titles[i])

        if i > 0:
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            axs[0, i - 1].add_patch(rect)

    plt.show()


if __name__ == "__main__":
    main()
