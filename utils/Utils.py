import os

import numpy as np
from scipy.interpolate import interp1d
from skimage import exposure
from skimage.io import imread


# si
def process_dat_file(data_path):
    """Loads the HSI file depending on its extension and returns it as a 3D array"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No such file or directory: '{data_path}'")

    file_extension = data_path.split('.')[-1]
    if file_extension == 'dat':
        # Para archivos .dat
        data = np.fromfile(data_path, dtype=np.float32)
        data = data.reshape((512, 204, 512))
    elif file_extension == 'npy':
        # Para archivos .npy
        data = np.load(data_path)
        if data.shape != (2048, 2048, 10):
            raise ValueError(f"Unexpected data shape {data.shape}")
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")
    return data


# si
def generate_rgb_from_hsi_toucan(hsi_data):
    wavenumbers = np.array([431., 479., 515., 567., 611., 666., 719., 775., 820., 877.])

    red_peak = 567
    green_peak = 515
    blue_peak = 431

    red_index = np.abs(wavenumbers - red_peak).argmin()
    green_index = np.abs(wavenumbers - green_peak).argmin()
    blue_index = np.abs(wavenumbers - blue_peak).argmin()

    # Crear curvas RGB basadas en las bandas más cercanas
    red_curve = np.zeros_like(wavenumbers)
    green_curve = np.zeros_like(wavenumbers)
    blue_curve = np.zeros_like(wavenumbers)

    red_curve[red_index] = 1
    green_curve[green_index] = 1
    blue_curve[blue_index] = 1

    red_channel = np.einsum('ijk,k->ij', hsi_data, red_curve)
    green_channel = np.einsum('ijk,k->ij', hsi_data, green_curve)
    blue_channel = np.einsum('ijk,k->ij', hsi_data, blue_curve)

    max_value = 1023.0
    rgb_image = np.stack([red_channel, green_channel, blue_channel], axis=-1) / max_value
    rgb_image = np.clip(rgb_image, 0, 1)

    return rgb_image

    return rgb_image


# si
def generate_rgb_from_hsi(hsi_data, wavelengths):
    if hsi_data.shape[1] != 204:
        hsi_data = np.transpose(hsi_data, (1, 0, 2))
    red_curve = np.exp(-0.5 * ((wavelengths - 620) / 50) ** 2)
    green_curve = np.exp(-0.5 * ((wavelengths - 550) / 50) ** 2)
    blue_curve = np.exp(-0.5 * ((wavelengths - 450) / 50) ** 2)
    rgb_image = np.stack([np.einsum('ijk,j->ik', hsi_data, curve) for curve in [red_curve, green_curve, blue_curve]],
                         axis=-1)
    rgb_image = (rgb_image - np.min(rgb_image)) / (np.max(rgb_image) - np.min(rgb_image))
    rgb_image = exposure.equalize_adapthist(rgb_image)
    return rgb_image


# si
def generate_rgb_from_ultris(imagen):
    num_bandas = 51
    longitudes_onda = np.linspace(450, 850, num_bandas)
    longitudes_cie = np.array(
        [450, 470, 490, 510, 530, 550, 570, 590, 610, 630, 650, 670, 690, 710, 730, 750, 770, 790, 810, 830, 850])
    cie_r = np.array(
        [0.00, 0.01, 0.02, 0.04, 0.08, 0.14, 0.20, 0.30, 0.45, 0.60, 0.75, 0.85, 0.90, 0.95, 0.98, 1.00, 0.90, 0.80,
         0.60, 0.40, 0.20])
    cie_g = np.array(
        [0.02, 0.05, 0.12, 0.30, 0.50, 0.75, 0.85, 0.80, 0.60, 0.40, 0.20, 0.10, 0.05, 0.02, 0.01, 0.00, 0.00, 0.00,
         0.00, 0.00, 0.00])
    cie_b = np.array(
        [0.90, 0.85, 0.75, 0.60, 0.40, 0.20, 0.10, 0.05, 0.02, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
         0.00, 0.00, 0.00])
    interp_r = interp1d(longitudes_cie, cie_r, kind="linear", bounds_error=False, fill_value=0)
    interp_g = interp1d(longitudes_cie, cie_g, kind="linear", bounds_error=False, fill_value=0)
    interp_b = interp1d(longitudes_cie, cie_b, kind="linear", bounds_error=False, fill_value=0)
    imagen_r = np.sum(imagen * interp_r(longitudes_onda)[None, None, :], axis=2)
    imagen_g = np.sum(imagen * interp_g(longitudes_onda)[None, None, :], axis=2)
    imagen_b = np.sum(imagen * interp_b(longitudes_onda)[None, None, :], axis=2)

    def normalizar(img):
        return (255 * (img - np.min(img)) / (np.max(img) - np.min(img))).astype(np.uint8)

    return np.stack([normalizar(imagen_r), normalizar(imagen_g), normalizar(imagen_b)], axis=-1)


# si
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


def load_yolo_annotations_fixed(annotation_path, img_width, img_height):
    boxes = []
    try:
        with open(annotation_path, "r") as file:
            lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                label = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])
                x_min = int((x_center - width / 2) * img_width)
                y_min = int((y_center - height / 2) * img_height)
                x_max = int((x_center + width / 2) * img_width)
                y_max = int((y_center + height / 2) * img_height)
                boxes.append((x_min, y_min, x_max, y_max, label))
            except ValueError:
                continue
    except FileNotFoundError:
        print(f"❌ Annotation file not found -> {annotation_path}")
    return boxes


# si
def extract_spectral_signatures(hsi_data, boxes, num_bands):
    signatures = {0: [], 1: [], 2: []}
    for (x_min, y_min, x_max, y_max, label) in boxes:
        label = int(label)
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(hsi_data.shape[1], x_max), min(hsi_data.shape[0], y_max)
        if x_min < x_max and y_min < y_max:
            region = hsi_data[y_min:y_max, x_min:x_max, :]
            mean_spectrum = np.mean(region, axis=(0, 1))
            if mean_spectrum.shape[0] == num_bands:
                signatures[label].append(mean_spectrum)  # Guarda el espectro medio directamente sin normalizar
    for label in signatures:
        signatures[label] = np.mean(signatures[label], axis=0) if signatures[label] else np.zeros(num_bands)
    return signatures


def extract_wavelengths(metadata_path):
    """Extracts wavelength values ​​from a .hdr metadata file and handles the special case of Ultris_SR5."""
    wavelengths = []
    capturing = False
    is_ultris = "Ultris_SR5" in metadata_path

    try:
        with open(metadata_path, 'r') as file:
            for line in file:
                line = line.strip().lower()

                # Manejo normal para Specim_IQ y EOS_M50
                if not is_ultris and "wavelength =" in line:
                    capturing = True
                    line = line.split("{")[-1]

                # Manejo especial para Ultris_SR5
                if is_ultris and "wavelength =" in line:
                    capturing = True
                    line = line.split("{")[-1]

                if capturing:
                    if "}" in line:
                        line = line.split("}")[0]
                        capturing = False

                    # Convertir valores a flotantes
                    wavelengths.extend(
                        [float(w.strip()) for w in line.split(",") if w.strip().replace('.', '', 1).isdigit()])

        # Verificar si se extrajeron valores
        if wavelengths:
            print(f"✅ Correctly extracted wavelengths for {metadata_path.split('/')[-1]}: {wavelengths[:10]} ...")
        else:
            print(f"❌ No wavelength values ​​were found in {metadata_path}.")

        return wavelengths if wavelengths else None

    except FileNotFoundError:
        print(f"❌ Archivo no encontrado: {metadata_path}")
        return None


def extract_rgb_from_hsi(image_data):
    num_bands = image_data.shape[2]
    if num_bands < 3:
        print("⚠️ There are not enough bands to generate RGB, using only the first band.")
        return image_data[:, :, 0]

    red_band = min(num_bands - 1, int(num_bands * 0.75))
    green_band = min(num_bands - 1, int(num_bands * 0.50))
    blue_band = min(num_bands - 1, int(num_bands * 0.25))

    rgb_image = np.stack([
        image_data[:, :, red_band],
        image_data[:, :, green_band],
        image_data[:, :, blue_band]
    ], axis=-1)

    rgb_image = (rgb_image - np.min(rgb_image)) / (np.max(rgb_image) - np.min(rgb_image))
    return rgb_image


def find_brightest_pixels(image_data, num_pixels):
    grayscale = np.mean(image_data, axis=2) if image_data.ndim == 3 else image_data.copy()
    flat_indices = np.argsort(grayscale.ravel())[-num_pixels:]
    bright_pixel_coords = np.unravel_index(flat_indices, grayscale.shape)
    highlighted_image = np.copy(image_data)

    if highlighted_image.ndim == 3 and highlighted_image.shape[2] > 1:
        highlighted_image = np.stack([grayscale] * 3, axis=-1)

    highlighted_image[bright_pixel_coords[0], bright_pixel_coords[1], :] = [0, 1, 0]
    return highlighted_image, bright_pixel_coords


def extract_spectral_signature(image_data, bright_pixel_coords):
    spectral_data = image_data[bright_pixel_coords[0], bright_pixel_coords[1], :]
    norms = np.linalg.norm(spectral_data, axis=1)
    min_norm_index = np.argmin(norms)
    return spectral_data[min_norm_index]


# def visualizar_pixeles_brillantes(camera_name):
def view_bright_pixels(camera_name):
    root_dir = os.path.dirname(os.path.abspath(__file__))

    if camera_name == "Ultris_SR5":
        file_extension = "tiff"
        num_pixels = 400
        num_bands = 51
    elif camera_name == "Toucan":
        file_extension = "npy"
        num_pixels = 100000
        num_bands = 10
    else:
        print(f"❌ Cámara '{camera_name}' no reconocida.")
        return None, None  # Retornar None si no es una cámara válida

    num_scenes = 19
    all_signatures = []

    for i in range(1, num_scenes + 1):
        scene_folder = os.path.join(root_dir, f'Scene_{i}')
        camera_folder = os.path.join(scene_folder, camera_name)

        image_paths = {
            "closed": os.path.join(camera_folder, f"HSI_closed.{file_extension}"),
            "open": os.path.join(camera_folder, f"HSI_open.{file_extension}")
        }

        if not os.path.exists(image_paths["closed"]) or not os.path.exists(image_paths["open"]):
            print(f"⚠️ No se encontraron ambas imágenes en {camera_name} - Scene_{i}, saltando.")
            continue

        if camera_name == "Ultris_SR5":
            closed_data = imread(image_paths["closed"]).astype(np.float32)
            open_data = imread(image_paths["open"]).astype(np.float32)
        else:
            closed_data = np.load(image_paths["closed"]).astype(np.float32)
            open_data = np.load(image_paths["open"]).astype(np.float32)

        _, bright_pixels_closed = find_brightest_pixels(closed_data, num_pixels)
        _, bright_pixels_open = find_brightest_pixels(open_data, num_pixels)

        closed_signature = extract_spectral_signature(closed_data, bright_pixels_closed)
        open_signature = extract_spectral_signature(open_data, bright_pixels_open)
        all_signatures.append((closed_signature, open_signature))

    # Si no hay datos, retornar None
    if not all_signatures:
        print(f"⚠️ No spectral signatures were found for {camera_name}")
        return None, None

    avg_closed_sig = np.mean([sig[0] for sig in all_signatures], axis=0)
    avg_open_sig = np.mean([sig[1] for sig in all_signatures], axis=0)

    return avg_closed_sig, avg_open_sig


def load_yolo_annotations_fixed(annotation_path, img_width, img_height):
    boxes = []
    try:
        with open(annotation_path, "r") as file:
            lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                label = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])
                x_min = int((x_center - width / 2) * img_width)
                y_min = int((y_center - height / 2) * img_height)
                x_max = int((x_center + width / 2) * img_width)
                y_max = int((y_center + height / 2) * img_height)
                boxes.append((x_min, y_min, x_max, y_max, label))
            except ValueError:
                continue
    except FileNotFoundError:
        print(f"❌ Archivo de anotaciones no encontrado -> {annotation_path}")
    return boxes


def extract_spectral_signatures(hsi_data, boxes, num_bands):
    signatures = {0: [], 1: [], 2: []}
    for (x_min, y_min, x_max, y_max, label) in boxes:
        label = int(label)
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(hsi_data.shape[1], x_max), min(hsi_data.shape[0], y_max)
        if x_min < x_max and y_min < y_max:
            region = hsi_data[y_min:y_max, x_min:x_max, :]
            mean_spectrum = np.mean(region, axis=(0, 1))
            if mean_spectrum.shape[0] == num_bands:
                signatures[label].append(mean_spectrum)  # Guarda el espectro medio directamente sin normalizar
    for label in signatures:
        signatures[label] = np.mean(signatures[label], axis=0) if signatures[label] else np.zeros(num_bands)
    return signatures


def extract_wavelengths(metadata_path):
    """Extracts wavelength values ​​from a .hdr metadata file and handles the special case of Ultris_SR5."""
    wavelengths = []
    capturing = False
    is_ultris = "Ultris_SR5" in metadata_path
    try:
        with open(metadata_path, 'r') as file:
            for line in file:
                line = line.strip().lower()

                # Manejo normal para Specim_IQ y EOS_M50
                if not is_ultris and "wavelength =" in line:
                    capturing = True
                    line = line.split("{")[-1]

                # Manejo especial para Ultris_SR5
                if is_ultris and "wavelength =" in line:
                    capturing = True
                    line = line.split("{")[-1]

                if capturing:
                    if "}" in line:
                        line = line.split("}")[0]
                        capturing = False

                    # Convertir valores a flotantes
                    wavelengths.extend(
                        [float(w.strip()) for w in line.split(",") if w.strip().replace('.', '', 1).isdigit()])

        # Verificar si se extrajeron valores
        if wavelengths:
            print(f"✅ Correctly extracted wavelengths for {metadata_path.split('/')[-1]}: {wavelengths[:10]} ...")
        else:
            print(f"❌ No wavelength values ​​were found in {metadata_path}.")

        return wavelengths if wavelengths else None

    except FileNotFoundError:
        print(f"❌ Archivo no encontrado: {metadata_path}")
        return None


def extract_rgb_from_hsi(image_data):
    num_bands = image_data.shape[2]
    if num_bands < 3:
        print("⚠️ There are not enough bands to generate RGB, using only the first band.")
        return image_data[:, :, 0]

    red_band = min(num_bands - 1, int(num_bands * 0.75))
    green_band = min(num_bands - 1, int(num_bands * 0.50))
    blue_band = min(num_bands - 1, int(num_bands * 0.25))

    rgb_image = np.stack([
        image_data[:, :, red_band],
        image_data[:, :, green_band],
        image_data[:, :, blue_band]
    ], axis=-1)

    rgb_image = (rgb_image - np.min(rgb_image)) / (np.max(rgb_image) - np.min(rgb_image))
    return rgb_image


def find_brightest_pixels(image_data, num_pixels):
    grayscale = np.mean(image_data, axis=2) if image_data.ndim == 3 else image_data.copy()
    flat_indices = np.argsort(grayscale.ravel())[-num_pixels:]
    bright_pixel_coords = np.unravel_index(flat_indices, grayscale.shape)
    highlighted_image = np.copy(image_data)

    if highlighted_image.ndim == 3 and highlighted_image.shape[2] > 1:
        highlighted_image = np.stack([grayscale] * 3, axis=-1)

    highlighted_image[bright_pixel_coords[0], bright_pixel_coords[1], :] = [0, 1, 0]
    return highlighted_image, bright_pixel_coords


def extract_spectral_signature(image_data, bright_pixel_coords):
    spectral_data = image_data[bright_pixel_coords[0], bright_pixel_coords[1], :]
    norms = np.linalg.norm(spectral_data, axis=1)
    min_norm_index = np.argmin(norms)
    return spectral_data[min_norm_index]


def view_bright_pixels(camera_name):
    root_dir = os.path.dirname(os.path.abspath(__file__))

    if camera_name == "Ultris_SR5":
        file_extension = "tiff"
        num_pixels = 400
        num_bands = 51
    elif camera_name == "Toucan":
        file_extension = "npy"
        num_pixels = 100000
        num_bands = 10
    else:
        print(f"❌ Camera '{camera_name}' Little recognized.")
        return None, None

    num_scenes = 19
    all_signatures = []

    for i in range(1, num_scenes + 1):
        scene_folder = os.path.join(root_dir, f'Scene_{i}')
        camera_folder = os.path.join(scene_folder, camera_name)

        image_paths = {
            "closed": os.path.join(camera_folder, f"HSI_closed.{file_extension}"),
            "open": os.path.join(camera_folder, f"HSI_open.{file_extension}")
        }

        if not os.path.exists(image_paths["closed"]) or not os.path.exists(image_paths["open"]):
            print(f"⚠️ Both images were not found in {camera_name} - Scene_{i}, saltando.")
            continue

        if camera_name == "Ultris_SR5":
            closed_data = imread(image_paths["closed"]).astype(np.float32)
            open_data = imread(image_paths["open"]).astype(np.float32)
        else:
            closed_data = np.load(image_paths["closed"]).astype(np.float32)
            open_data = np.load(image_paths["open"]).astype(np.float32)

        _, bright_pixels_closed = find_brightest_pixels(closed_data, num_pixels)
        _, bright_pixels_open = find_brightest_pixels(open_data, num_pixels)

        closed_signature = extract_spectral_signature(closed_data, bright_pixels_closed)
        open_signature = extract_spectral_signature(open_data, bright_pixels_open)
        all_signatures.append((closed_signature, open_signature))

    if not all_signatures:
        print(f"⚠️ No spectral signatures were found for {camera_name}")
        return None, None

    avg_closed_sig = np.mean([sig[0] for sig in all_signatures], axis=0)
    avg_open_sig = np.mean([sig[1] for sig in all_signatures], axis=0)

    return avg_closed_sig, avg_open_sig


