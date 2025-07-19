import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.io import imread
from src.schemas.acquisitions import AcquisitionInfo
from src.utils.spectra import correct_flat_and_dark, correct_reference_spectrum

def extract_wavelengths(metadata_path):
    """Extracts wavelength values ​​from a .hdr metadata file and handles the special case of Ultris_SR5."""
    # Si es un archivo .csv, leer con pandas
    if metadata_path.endswith('.csv'):
        try:
            df = pd.read_csv(metadata_path)
            # Buscar columna con nombre similar a 'wavelength' (ignorando mayúsculas/minúsculas)
            col = next((c for c in df.columns if 'wavelength' in c.lower()), None)
            if col is not None:
                arr = df[col].to_numpy(dtype=float)
                return arr
            else:
                print(f"❌ No se encontró columna de longitudes de onda en {metadata_path}")
                return None
        except Exception as e:
            print(f"❌ Error leyendo {metadata_path}: {e}")
            return None
    # Si es un .hdr, usar el método anterior
    wavelengths = []
    capturing = False
    is_ultris = "Ultris_SR5" in metadata_path
    try:
        with open(metadata_path, 'r') as file:
            for line in file:
                line = line.strip().lower()
                if not is_ultris and "wavelength =" in line:
                    capturing = True
                    line = line.split("{")[-1]
                if is_ultris and "wavelength =" in line:
                    capturing = True
                    line = line.split("{")[-1]
                if capturing:
                    if "}" in line:
                        line = line.split("}")[0]
                        capturing = False
                    wavelengths.extend(
                        [float(w.strip()) for w in line.split(",") if w.strip().replace('.', '', 1).isdigit()])
        return np.array(wavelengths) if wavelengths else None
    except FileNotFoundError:
        print(f"❌ File not found: {metadata_path}")
        return None


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
                signatures[label].append(mean_spectrum)
    for label in signatures:
        signatures[label] = np.mean(signatures[label], axis=0) if signatures[label] else np.zeros(num_bands)
    return signatures


#def load_and_correct_image(scene, camera_name, cocoa_condition, general_reference=None):
def load_and_correct_image(scene, camera_name, cocoa_condition, data_dir, general_reference=None):
    """
    Loads and applies all corrections (flat/dark, reference) to the image for a given scene, camera, and condition.
    If general_reference is provided, it is used for all scenes/conditions.
    """
    # Set correct format for each camera
    if camera_name == "ultris_sr5":
        img_format = "tiff"
    elif camera_name == "toucan":
        img_format = "npy"
    elif camera_name == "specim_iq":
        img_format = "dat"
    elif camera_name == "eos_m50":
        img_format = "jpg"
    else:
        img_format = None

    try:
        if camera_name == "specim_iq":
            folder = os.path.join(data_dir, "scenes", f"scene_{scene:02d}", "specim_iq")
            fname = f"hsi_{cocoa_condition}.dat"
            fpath = os.path.join(folder, fname)
            if not os.path.exists(fpath):
                return None
            image_raw = np.fromfile(fpath, dtype=np.float32).reshape((512, 204, 512)).transpose(0, 2, 1)
            return image_raw
        acq_info = AcquisitionInfo(
            scene=scene,
            camera_name=camera_name,
            format=img_format,
            cocoa_condition=str(cocoa_condition) if not isinstance(cocoa_condition, str) else cocoa_condition,
        )
        acq_pixels = acq_info.load_pixels_info()
        image_raw = acq_info.load_image(normalize=False)
        flat = acq_info.load_flat_field()
        dark = acq_info.load_dark_field()
        image_fielded = correct_flat_and_dark(image_raw, flat_field=flat, dark_field=dark)
        if general_reference is not None:
            image_corrected = correct_reference_spectrum(image_fielded, spectrum=general_reference)
        else:
            reference_spectrum = acq_pixels.get_reference_spectrum(image=image_fielded)
            image_corrected = correct_reference_spectrum(image_fielded, spectrum=reference_spectrum)
        return image_corrected
    except Exception as e:
        # Silenciar todas las excepciones
        return None


def plot_normalized_signatures(camera_name, wavelengths, data_dir, reference_dict=None):
    if camera_name not in wavelengths:
        print(f"❌ Camera '{camera_name}' Not found in metadata..")
        return

    camera_wavelengths = wavelengths[camera_name]

    if camera_name == "specim_iq":
        camera_wavelengths = np.array(camera_wavelengths)
        mask_specim = (camera_wavelengths >= 480) & (camera_wavelengths <= 950)
        camera_wavelengths = camera_wavelengths[mask_specim]
    elif camera_name == "toucan":
        camera_wavelengths = np.array(camera_wavelengths)
        mask_toucan = camera_wavelengths >= 570
        camera_wavelengths = camera_wavelengths[mask_toucan]
    else:
        mask_specim = None
        mask_toucan = None

    global_data = {"open": {0: [], 1: [], 2: []}, "closed": {0: [], 1: [], 2: []}}

    LABELS = {0: "Good", 1: "Bad", 2: "Partial"}
    COLORS = {"Good": "blue", "Bad": "red", "Partial": "green"}

    # Load general reference spectrum if available (one for all scenes)
    general_reference = None
    if reference_dict and camera_name in reference_dict:
        general_reference = reference_dict[camera_name]

    for i in range(1, 20):
        scene_folder = os.path.join(data_dir, "scenes", f'scene_{i:02d}')
        camera_folder = os.path.join(scene_folder, camera_name)
        if not os.path.exists(camera_folder):
            continue

        for condition in ["closed", "open"]:
            # Usar load_and_correct_image para aplicar dark/flat/reference a todas las cámaras
            image_corrected = load_and_correct_image(i, camera_name, condition, data_dir, general_reference=general_reference)
            if image_corrected is None:
                continue
            num_bands = image_corrected.shape[2]
            annotation_path = os.path.join(camera_folder, f'annotations_{condition}.txt')
            if not os.path.exists(annotation_path):
                continue
            from src.utils.utils_yolo import load_yolo_annotations
            boxes = load_yolo_annotations(annotation_path, image_corrected.shape[1], image_corrected.shape[0])
            for (x_min, y_min, x_max, y_max, label) in boxes:
                label = int(label)
                x_min, y_min = max(0, x_min), max(0, y_min)
                x_max, y_max = min(image_corrected.shape[1], x_max), min(image_corrected.shape[0], y_max)
                if x_min < x_max and y_min < y_max:
                    region = image_corrected[y_min:y_max, x_min:x_max, :]
                    h, w, _ = region.shape
                    if h < 3 or w < 3:
                        center_y, center_x = h // 2, w // 2
                        sig = region[center_y, center_x, :]
                        if sig.shape[0] == num_bands:
                            global_data[condition][label].append(sig)
                    else:
                        center_y, center_x = h // 2, w // 2
                        window = 2  # 5x5 window
                        for dy in range(-window, window+1):
                            for dx in range(-window, window+1):
                                yy = center_y + dy
                                xx = center_x + dx
                                if 0 <= yy < h and 0 <= xx < w:
                                    sig = region[yy, xx, :]
                                    if sig.shape[0] == num_bands:
                                        global_data[condition][label].append(sig)

    has_data = any(global_data[condition][label] for condition in ["closed", "open"] for label in LABELS)
    if not has_data:
        print(f"⚠️ No data available for the camera '{camera_name}'.")
        return

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 30
    })

    plt.figure(figsize=(15, 5))
    line_styles = {'closed': '-', 'open': '--'}
    line_width = 2.5


    alpha = 1  # Puedes ajustar este valor según lo que desees
    for condition in ["closed", "open"]:
        for label in [0, 1, 2]:
            if global_data[condition][label]:
                signature_stack = np.stack(global_data[condition][label])
                # Normalizar cada firma individualmente desde el inicio
                min_vals = np.min(signature_stack, axis=1, keepdims=True)
                max_vals = np.max(signature_stack, axis=1, keepdims=True)
                norm_stack = (signature_stack - min_vals) / (max_vals - min_vals + 1e-8)
                mean_signature = np.mean(norm_stack, axis=0)
                std_signature = np.std(norm_stack, axis=0)

                # Ajustar longitudes de onda y firmas para cada cámara

                if camera_name == "specim_iq":
                    filtered_wavelengths = np.array(wavelengths[camera_name])[mask_specim]
                    filtered_mean = np.array(mean_signature)[mask_specim]
                    filtered_std = np.array(std_signature)[mask_specim]
                    wavelengths_to_plot = filtered_wavelengths
                    mean_signature = filtered_mean
                    std_signature = filtered_std
                elif camera_name == "toucan":
                    filtered_wavelengths = np.array(wavelengths[camera_name])[mask_toucan]
                    filtered_mean = np.array(mean_signature)[mask_toucan]
                    filtered_std = np.array(std_signature)[mask_toucan]
                    min_len = min(len(filtered_wavelengths), len(filtered_mean))
                    wavelengths_to_plot = filtered_wavelengths[:min_len]
                    mean_signature = filtered_mean[:min_len]
                    std_signature = filtered_std[:min_len]
                elif camera_name == "eos_m50":
                    wavelengths_to_plot = [450, 550, 650][::-1]
                    # Asegurar que std_signature tenga el mismo tamaño
                    std_signature = np.std(norm_stack, axis=0)
                    if len(std_signature) > len(wavelengths_to_plot):
                        std_signature = std_signature[:len(wavelengths_to_plot)]
                    elif len(std_signature) < len(wavelengths_to_plot):
                        std_signature = np.pad(std_signature, (0, len(wavelengths_to_plot)-len(std_signature)), 'constant')
                else:
                    wavelengths_to_plot = camera_wavelengths
                    std_signature = np.std(norm_stack, axis=0)
                    if len(std_signature) > len(wavelengths_to_plot):
                        std_signature = std_signature[:len(wavelengths_to_plot)]
                    elif len(std_signature) < len(wavelengths_to_plot):
                        std_signature = np.pad(std_signature, (0, len(wavelengths_to_plot)-len(std_signature)), 'constant')

                # Graficar media y banda de desviación estándar para todas las cámaras
                plt.plot(wavelengths_to_plot, mean_signature,
                         label=f'{condition} - {LABELS[label]}',
                         color=COLORS[LABELS[label]], linestyle=line_styles[condition], linewidth=line_width)
                plt.fill_between(wavelengths_to_plot,
                                 mean_signature - alpha * std_signature,
                                 mean_signature + alpha * std_signature,
                                 color=COLORS[LABELS[label]], alpha=0.2)

    plt.xlabel('Wavelength (nm)', fontsize=30, family="Times New Roman")
    plt.ylabel('Normalized Reflectance', fontsize=30, family="Times New Roman")
    plt.tick_params(axis='both', which='major', labelsize=30)

    plt.legend()
    plt.grid(True)
    plt.title(f"Spectral Signatures - {camera_name}")
    plt.show()


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


def correct_reference_spectrum_signature(signature, reference):
    """
    Correct a 1D spectral signature by dividing by a reference spectrum.
    """
    signature = np.asarray(signature)
    reference = np.asarray(reference)
    reference = np.where(reference == 0, 1e-8, reference)
    return signature / reference


def extract_signatures_all_cameras(data_dir, wavelengths, n_firmas_por_box=9, reference_dict=None):
    """
    Extracts spectral signatures for all cameras, scenes, and conditions.
    If reference_dict is provided, applies reference correction per camera.
    Returns a DataFrame with columns: scene, camera, condition, label, firma (array).
    """
    cameras = list(wavelengths.keys())
    data = []
    for i in range(1, 20):
        scene_folder = os.path.join(data_dir, "scenes", f'scene_{i:02d}')
        for camera_name in cameras:
            camera_folder = os.path.join(scene_folder, camera_name)
            if not os.path.exists(camera_folder):
                continue
            if camera_name == "specim_iq":
                image_paths = {
                    "closed": os.path.join(camera_folder, "hsi_closed.dat"),
                    "open": os.path.join(camera_folder, "hsi_open.dat")
                }
                num_bands = 204
            elif camera_name == "eos_m50":
                image_paths = {"open": os.path.join(camera_folder, "hsi_open.jpg")}
                num_bands = 3
            elif camera_name == "ultris_sr5":
                image_paths = {
                    "closed": os.path.join(camera_folder, "hsi_closed.tiff"),
                    "open": os.path.join(camera_folder, "hsi_open.tiff")
                }
                num_bands = len(wavelengths[camera_name])
            elif camera_name == "toucan":
                image_paths = {
                    "closed": os.path.join(camera_folder, "hsi_closed.npy"),
                    "open": os.path.join(camera_folder, "hsi_open.npy")
                }
                num_bands = len(wavelengths[camera_name])
            else:
                continue
            for condition in image_paths:
                image_path = image_paths[condition]
                annotation_path = os.path.join(camera_folder, f'annotations_{condition}.txt')
                if not os.path.exists(image_path) or not os.path.exists(annotation_path):
                    continue
                try:
                    if camera_name == "specim_iq":
                        image_data = np.fromfile(image_path, dtype=np.float32).reshape((512, 204, 512)).transpose(0, 2, 1)
                    elif camera_name == "toucan":
                        image_data = np.load(image_path)
                    else:
                        image_data = imread(image_path)
                        if camera_name == "eos_m50":

                            #image_data = image_data.astype(np.float32) / 255.0
                            image_data = image_data[:, :, ::-1].astype(np.float32) / 255.0
                    from src.utils.utils_yolo import load_yolo_annotations
                    boxes = load_yolo_annotations(annotation_path, image_data.shape[1], image_data.shape[0])
                    for (x_min, y_min, x_max, y_max, label) in boxes:
                        label = int(label)
                        x_min, y_min = max(0, x_min), max(0, y_min)
                        x_max, y_max = min(image_data.shape[1], x_max), min(image_data.shape[0], y_max)
                        if x_min < x_max and y_min < y_max:
                            region = image_data[y_min:y_max, x_min:x_max, :]
                            h, w, _ = region.shape
                            if h < 3 or w < 3:
                                center_y, center_x = h // 2, w // 2
                                sig = region[center_y, center_x, :]
                                if sig.shape[0] == num_bands:
                                    # Apply reference correction if needed
                                    if reference_dict and camera_name in reference_dict:
                                        sig = correct_reference_spectrum_signature(sig, reference_dict[camera_name])
                                    data.append({
                                        "scene": i,
                                        "camera": camera_name,
                                        "condition": condition,
                                        "label": label,
                                        "firma": sig
                                    })
                            else:
                                center_y, center_x = h // 2, w // 2
                                # Extraer más firmas alrededor del centro (ventana 5x5)
                                window = 2  # para 5x5
                                for dy in range(-window, window+1):
                                    for dx in range(-window, window+1):
                                        yy = center_y + dy
                                        xx = center_x + dx
                                        if 0 <= yy < h and 0 <= xx < w:
                                            sig = region[yy, xx, :]
                                            if sig.shape[0] == num_bands:
                                                if reference_dict and camera_name in reference_dict:
                                                    sig = correct_reference_spectrum_signature(sig, reference_dict[camera_name])
                                                data.append({
                                                    "scene": i,
                                                    "camera": camera_name,
                                                    "condition": condition,
                                                    "label": label,
                                                    "firma": sig
                                                })
                except Exception as e:
                    print(f"Error in scene {i}, camera {camera_name}, condition {condition}: {e}")
    return pd.DataFrame(data)


def extract_signatures_toucan(data_dir, wavelengths, n_firmas_por_box=9):

    import pandas as pd
    camera_name = "toucan"
    data = []
    num_bands = len(wavelengths[camera_name])
    for i in range(1, 20):
        scene_folder = os.path.join(data_dir, "scenes", f'scene_{i:02d}')
        camera_folder = os.path.join(scene_folder, camera_name)
        for condition in ["closed", "open"]:
            image_path = os.path.join(camera_folder, f"hsi_{condition}.npy")
            annotation_path = os.path.join(camera_folder, f'annotations_{condition}.txt')
            if not os.path.exists(image_path) or not os.path.exists(annotation_path):
                continue
            try:
                image_data = np.load(image_path)
                if image_data.ndim != 3 or image_data.shape[2] != num_bands:
                    print(f"❌ Imagen {image_path} shape {image_data.shape}, se esperaban 3D y {num_bands} bandas")
                    continue
                from src.utils.utils_yolo import load_yolo_annotations
                boxes = load_yolo_annotations(annotation_path, image_data.shape[1], image_data.shape[0])
                print(f"📂 Processing toucan in {camera_folder} - {condition}: {len(boxes)} boxes")
                for (x_min, y_min, x_max, y_max, label) in boxes:
                    print(f"  Box: ({x_min}, {y_min}, {x_max}, {y_max}), label={label}")
                    label = int(label)
                    x_min, y_min = max(0, x_min), max(0, y_min)
                    x_max, y_max = min(image_data.shape[1], x_max), min(image_data.shape[0], y_max)
                    if x_min < x_max and y_min < y_max:
                        region = image_data[y_min:y_max, x_min:x_max, :]
                        h, w, _ = region.shape
                        if h < 1 or w < 1:
                            print(f"    ⚠️ Región vacía para box ({x_min},{y_min},{x_max},{y_max})")
                            continue
                        center_y, center_x = h // 2, w // 2
                        offsets = [ (0,0), (-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1) ]
                        firmas_extraidas = 0
                        for dy, dx in offsets:
                            yy = center_y + dy
                            xx = center_x + dx
                            if 0 <= yy < h and 0 <= xx < w:
                                sig = region[yy, xx, :]
                                if sig.shape[0] == num_bands:
                                    data.append({
                                        "scene": i,
                                        "condition": condition,
                                        "label": label,
                                        "firma": sig
                                    })
                                    firmas_extraidas += 1
                        print(f"    Firmas extraídas de este box: {firmas_extraidas}")
                    else:
                        print(f"    ⚠️ Box fuera de límites o inválido: ({x_min},{y_min},{x_max},{y_max})")
            except Exception as e:
                print(f"❌ Error en escena {i}, toucan, {condition}: {e}")
    df = pd.DataFrame(data)
    # (Debug de resumen de clases eliminado)
    return df


def mean_signatures_by_group(df):
    """
    Devuelve un DataFrame con las mismas columnas y la firma media.
    Agrupa por scene, camera, condition, label y calcula la firma media.
    """
    import pandas as pd
    return df.groupby(['scene', 'camera', 'condition', 'label'], as_index=False)['firma'].apply(
        lambda x: np.mean(np.stack(x), axis=0))


def correct_spectrum_with_reference(spectrum, reference):
    """
    Correct a spectrum by dividing by a reference spectrum (halogen or spectralon).
    Both inputs must be 1D arrays of the same length.
    """
    spectrum = np.asarray(spectrum)
    reference = np.asarray(reference)
    # Avoid division by zero
    reference = np.where(reference == 0, 1e-8, reference)
    return spectrum / reference


def correct_signatures_with_reference(df, reference_spectrum):
    """
    Given a DataFrame with a 'firma' column (spectral signatures),
    returns a new DataFrame with the signatures corrected by the reference spectrum.
    """
    df = df.copy()
    df['firma'] = df['firma'].apply(lambda x: correct_spectrum_with_reference(x, reference_spectrum))
    return df