import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.io import imread

from src.utils import process_dat_file, generate_rgb_from_hsi, load_yolo_annotations, generate_rgb_from_ultris


def plot_images_and_annotations(scene_folder, scene_number, wavelengths):
    Specim_IQ_folder = os.path.join(scene_folder, 'Specim_IQ')
    ultris_folder = os.path.join(scene_folder, 'Ultris_SR5')
    Cannon_folder = os.path.join(scene_folder, 'Cannon')

    # Paths for Specim IQ
    dat_closed_path = os.path.join(Specim_IQ_folder, 'HSI_closed.dat')
    dat_open_path = os.path.join(Specim_IQ_folder, 'HSI_open.dat')
    annotations_closed_path = os.path.join(Specim_IQ_folder, 'annotations_closed.txt')
    annotations_open_path = os.path.join(Specim_IQ_folder, 'annotations_open.txt')

    label_ferm_path = os.path.join(Cannon_folder, 'label_fermentation.jpg')
    rgb_path = os.path.join(Cannon_folder, 'RGB.jpg')

    # Paths for Ultris SR5
    closed_tiff_path = os.path.join(ultris_folder, 'HSI_closed.tiff')
    open_tiff_path = os.path.join(ultris_folder, 'HSI_open.tiff')
    closed_annotation_path = os.path.join(ultris_folder, 'annotations_closed.txt')
    open_annotation_path = os.path.join(ultris_folder, 'annotations_open.txt')

    # Annotations RGB
    rgb_open_annotation_path = os.path.join(Cannon_folder, 'annotations_open.txt')

    # Process Specim IQ data
    closed_data = process_dat_file(dat_closed_path)
    open_data = process_dat_file(dat_open_path)
    closed_rgb_specim = generate_rgb_from_hsi(closed_data, wavelengths)
    open_rgb_specim = generate_rgb_from_hsi(open_data, wavelengths)
    closed_boxes_specim = load_yolo_annotations(annotations_closed_path, 512, 512)
    open_boxes_specim = load_yolo_annotations(annotations_open_path, 512, 512)

    label_ferm_img = io.imread(label_ferm_path)
    rgb_img = io.imread(rgb_path)

    # Process Ultris SR5 data
    closed_ultris_rgb = generate_rgb_from_ultris(imread(closed_tiff_path))
    open_ultris_rgb = generate_rgb_from_ultris(imread(open_tiff_path))
    closed_boxes_ultris = load_yolo_annotations(closed_annotation_path, closed_ultris_rgb.shape[1],
                                                closed_ultris_rgb.shape[0])
    open_boxes_ultris = load_yolo_annotations(open_annotation_path, open_ultris_rgb.shape[1], open_ultris_rgb.shape[0])

    rgb_open_boxes_ultris = load_yolo_annotations(rgb_open_annotation_path, rgb_img.shape[1], rgb_img.shape[0])

    # Create plot
    fig, axs = plt.subplots(3, 2, figsize=(20, 200))
    fig.suptitle(f'Scene {scene_number}', fontsize=16)

    # Display Specim IQ images with annotations
    axs[1, 0].imshow(closed_rgb_specim)
    axs[1, 0].set_title("Specim_IQ_closed")

    axs[1, 1].imshow(open_rgb_specim)
    axs[1, 1].set_title("Specim_IQ_open")

    for ax, boxes in zip([axs[1, 0], axs[1, 1]], [closed_boxes_specim, open_boxes_specim]):
        for (x_min, y_min, x_max, y_max, label, color) in boxes:
            ax.add_patch(
                plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, edgecolor=color, linewidth=1, fill=False))
            # ax.text(x_min, y_min - 10, label, color=color, fontsize=8, bbox=dict(facecolor='white', alpha=0.5))

    # Display Ultris SR5 images with annotations
    axs[2, 0].imshow(closed_ultris_rgb)
    axs[2, 0].set_title("Ultris_closed")

    axs[2, 1].imshow(open_ultris_rgb)
    axs[2, 1].set_title("Ultris_open")

    for ax, boxes in zip([axs[2, 0], axs[2, 1]], [closed_boxes_ultris, open_boxes_ultris]):
        for (x_min, y_min, x_max, y_max, label, color) in boxes:
            ax.add_patch(
                plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, edgecolor=color, linewidth=1, fill=False))
            # ax.text(x_min, y_min - 10, label, color=color, fontsize=8, bbox=dict(facecolor='white', alpha=0.5))

    axs[0, 1].imshow(label_ferm_img)
    axs[0, 1].set_title("Labels")

    axs[0, 0].imshow(rgb_img)
    axs[0, 0].set_title("Cannon_open")

    for ax, boxes in zip([axs[0, 0]], [rgb_open_boxes_ultris]):
        for (x_min, y_min, x_max, y_max, label, color) in boxes:
            ax.add_patch(
                plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, edgecolor=color, linewidth=1, fill=False))

    for ax in axs.flat:
        ax.axis('off')

    plt.show()


def main():
    root_dir = Path(__file__).resolve().parents[2]
    data_subfolder = "data/scenes"
    data_dir = root_dir / data_subfolder
    wavelengths = np.linspace(400, 1000, 204)
    for i in range(1, 10):
        scene_folder = os.path.join(data_dir, f'Scene_{i}')
        plot_images_and_annotations(scene_folder, i, wavelengths)


if __name__ == "__main__":
    main()
