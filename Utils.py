import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, exposure
from skimage.io import imread
from scipy.interpolate import interp1d

def process_dat_file(data_path):
    """Carga el archivo HSI y lo devuelve como un array 3D (Filas, Bandas, Columnas)."""
    data = np.fromfile(data_path, dtype=np.float32)
    data = data.reshape((512, 204, 512))
    return data

def generate_rgb_from_hsi(hsi_data, wavelengths):
    if hsi_data.shape[1] != 204:
        hsi_data = np.transpose(hsi_data, (1, 0, 2))
    red_curve = np.exp(-0.5 * ((wavelengths - 620) / 50) ** 2)
    green_curve = np.exp(-0.5 * ((wavelengths - 550) / 50) ** 2)
    blue_curve = np.exp(-0.5 * ((wavelengths - 450) / 50) ** 2)
    rgb_image = np.stack([np.einsum('ijk,j->ik', hsi_data, curve) for curve in [red_curve, green_curve, blue_curve]], axis=-1)
    rgb_image = (rgb_image - np.min(rgb_image)) / (np.max(rgb_image) - np.min(rgb_image))
    rgb_image = exposure.equalize_adapthist(rgb_image)
    return rgb_image

def generate_rgb_from_ultris(imagen):
    num_bandas = 51
    longitudes_onda = np.linspace(450, 850, num_bandas)
    longitudes_cie = np.array([450, 470, 490, 510, 530, 550, 570, 590, 610, 630, 650, 670, 690, 710, 730, 750, 770, 790, 810, 830, 850])
    cie_r = np.array([0.00, 0.01, 0.02, 0.04, 0.08, 0.14, 0.20, 0.30, 0.45, 0.60, 0.75, 0.85, 0.90, 0.95, 0.98, 1.00, 0.90, 0.80, 0.60, 0.40, 0.20])
    cie_g = np.array([0.02, 0.05, 0.12, 0.30, 0.50, 0.75, 0.85, 0.80, 0.60, 0.40, 0.20, 0.10, 0.05, 0.02, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00])
    cie_b = np.array([0.90, 0.85, 0.75, 0.60, 0.40, 0.20, 0.10, 0.05, 0.02, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00])
    interp_r = interp1d(longitudes_cie, cie_r, kind="linear", bounds_error=False, fill_value=0)
    interp_g = interp1d(longitudes_cie, cie_g, kind="linear", bounds_error=False, fill_value=0)
    interp_b = interp1d(longitudes_cie, cie_b, kind="linear", bounds_error=False, fill_value=0)
    imagen_r = np.sum(imagen * interp_r(longitudes_onda)[None, None, :], axis=2)
    imagen_g = np.sum(imagen * interp_g(longitudes_onda)[None, None, :], axis=2)
    imagen_b = np.sum(imagen * interp_b(longitudes_onda)[None, None, :], axis=2)
    def normalizar(img):
        return (255 * (img - np.min(img)) / (np.max(img) - np.min(img))).astype(np.uint8)
    return np.stack([normalizar(imagen_r), normalizar(imagen_g), normalizar(imagen_b)], axis=-1)

def load_yolo_annotations(annotation_path, img_width, img_height):
    boxes = []
    label_map = {0: 'Good', 1: 'Bad', 2: 'Partially'}
    colors = {0: 'green', 1: 'red', 2: 'blue'}
    with open(annotation_path, 'r') as file:
        for line in file:
            data = line.strip().split()
            label, x_center, y_center, width, height = map(float, data)
            x_min = int((x_center - width / 2) * img_width)
            y_min = int((y_center - height / 2) * img_height)
            x_max = int((x_center + width / 2) * img_width)
            y_max = int((y_center + height / 2) * img_height)
            boxes.append((x_min, y_min, x_max, y_max, label_map[int(label)], colors[int(label)]))
    return boxes