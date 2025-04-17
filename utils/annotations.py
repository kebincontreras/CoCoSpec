import os

from matplotlib import pyplot as plt
from skimage.io import imread

from utils.Utils import process_dat_file, generate_rgb_from_hsi, generate_rgb_from_hsi_toucan, generate_rgb_from_ultris, \
    load_yolo_annotations


def plot_images_and_annotations(scene_folder, scene_number, wavelengths):
    specim_iq_folder = os.path.join(scene_folder, 'Specim_IQ')
    ultris_folder = os.path.join(scene_folder, 'Ultris_SR5')
    cannon_folder = os.path.join(scene_folder, 'EOS_M50')
    toucan_folder = os.path.join(scene_folder, 'Toucan')

    dat_closed_iq = os.path.join(specim_iq_folder, 'HSI_closed.dat')
    dat_open_iq = os.path.join(specim_iq_folder, 'HSI_open.dat')
    annotations_closed_iq = os.path.join(specim_iq_folder, 'annotations_closed.txt')
    annotations_open_iq = os.path.join(specim_iq_folder, 'annotations_open.txt')

    dat_closed_toucan = os.path.join(toucan_folder, 'HSI_closed.npy')
    dat_open_toucan = os.path.join(toucan_folder, 'HSI_open.npy')
    annotations_closed_toucan = os.path.join(toucan_folder, 'annotations_closed.txt')
    annotations_open_toucan = os.path.join(toucan_folder, 'annotations_open.txt')

    closed_tiff_ultris = os.path.join(ultris_folder, 'HSI_closed.tiff')
    open_tiff_ultris = os.path.join(ultris_folder, 'HSI_open.tiff')
    annotations_closed_ultris = os.path.join(ultris_folder, 'annotations_closed.txt')
    annotations_open_ultris = os.path.join(ultris_folder, 'annotations_open.txt')

    label_ferm_path = os.path.join(cannon_folder, 'label_fermentation.jpg')
    rgb_path = os.path.join(cannon_folder, 'HSI_open.jpg')
    rgb_open_annotation_path = os.path.join(cannon_folder, 'annotations_open.txt')

    closed_data_iq = process_dat_file(dat_closed_iq)
    open_data_iq = process_dat_file(dat_open_iq)
    closed_rgb_iq = generate_rgb_from_hsi(closed_data_iq, wavelengths)
    open_rgb_iq = generate_rgb_from_hsi(open_data_iq, wavelengths)

    closed_data_toucan = process_dat_file(dat_closed_toucan)
    open_data_toucan = process_dat_file(dat_open_toucan)
    closed_rgb_toucan = generate_rgb_from_hsi_toucan(closed_data_toucan)
    open_rgb_toucan = generate_rgb_from_hsi_toucan(open_data_toucan)

    closed_rgb_ultris = generate_rgb_from_ultris(imread(closed_tiff_ultris))
    open_rgb_ultris = generate_rgb_from_ultris(imread(open_tiff_ultris))

    closed_boxes_iq = load_yolo_annotations(annotations_closed_iq, closed_rgb_iq.shape[1], closed_rgb_iq.shape[0])
    open_boxes_iq = load_yolo_annotations(annotations_open_iq, open_rgb_iq.shape[1], open_rgb_iq.shape[0])

    closed_boxes_toucan = load_yolo_annotations(annotations_closed_toucan, closed_rgb_toucan.shape[1],
                                                closed_rgb_toucan.shape[0])
    open_boxes_toucan = load_yolo_annotations(annotations_open_toucan, open_rgb_toucan.shape[1],
                                              open_rgb_toucan.shape[0])

    label_ferm_img = imread(label_ferm_path)
    rgb_img = imread(rgb_path)

    # Crear plot
    fig, axs = plt.subplots(2, 4, figsize=(20, 30))
    fig.suptitle(f'Scene {scene_number}', fontsize=16)

    # Cannon
    axs[0, 0].imshow(label_ferm_img)
    axs[0, 0].set_title("Fermentation Labels")
    axs[1, 0].imshow(rgb_img)
    axs[1, 0].set_title("Cannon Open")

    # Toucan
    axs[0, 1].imshow(closed_rgb_toucan)
    axs[0, 1].set_title("Toucan Closed")
    axs[1, 1].imshow(open_rgb_toucan)
    axs[1, 1].set_title("Toucan Open")

    # Ultris
    axs[0, 2].imshow(closed_rgb_ultris)
    axs[0, 2].set_title("Ultris Closed")
    axs[1, 2].imshow(open_rgb_ultris)
    axs[1, 2].set_title("Ultris Open")

    # Specim IQ
    axs[0, 3].imshow(closed_rgb_iq)
    axs[0, 3].set_title("Specim IQ Closed")
    axs[1, 3].imshow(open_rgb_iq)
    axs[1, 3].set_title("Specim IQ Open")

    plt.show()
