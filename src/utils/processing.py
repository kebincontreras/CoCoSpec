import os

import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread


def extract_wavelengths(metadata_path):
    """Extracts wavelength values ​​from a .hdr metadata file and handles the special case of Ultris_SR5."""
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

                    # Convertir valores a flotantes
                    wavelengths.extend(
                        [float(w.strip()) for w in line.split(",") if w.strip().replace('.', '', 1).isdigit()])

        return wavelengths if wavelengths else None

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


def plot_normalized_signatures(camera_name, wavelengths, data_dir):
    if camera_name not in wavelengths:
        print(f"❌ Camera '{camera_name}' Not found in metadata..")
        return

    camera_wavelengths = wavelengths[camera_name]

    if camera_name == "specim_iq":
        camera_wavelengths = np.array(camera_wavelengths)
        mask_specim = (camera_wavelengths >= 480) & (camera_wavelengths <= 950)
        camera_wavelengths = camera_wavelengths[mask_specim]
    else:
        mask_specim = None

    global_data = {"open": {0: [], 1: [], 2: []}, "closed": {0: [], 1: [], 2: []}}

    LABELS = {0: "Good", 1: "Bad", 2: "Partial"}
    COLORS = {"Good": "blue", "Bad": "red", "Partial": "green"}

    for i in range(1, 20):
        scene_folder = os.path.join(data_dir, "scenes", f'scene_{i:02d}')
        camera_folder = os.path.join(scene_folder, camera_name)

        if not os.path.exists(camera_folder):
            print(f"⚠️ Cámara '{camera_name}' not found in {scene_folder}.")
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
            num_bands = len(camera_wavelengths)
        elif camera_name == "toucan":
            alternative_camera_folder = os.path.join(data_dir, "toucan", f'scene_{i}')
            if os.path.exists(alternative_camera_folder):
                camera_folder = alternative_camera_folder
            image_paths = {
                "closed": os.path.join(camera_folder, "hsi_closed.npy"),
                "open": os.path.join(camera_folder, "hsi_open.npy")
            }
            num_bands = len(camera_wavelengths)

        available_conditions = [cond for cond in image_paths if os.path.exists(image_paths[cond])]
        if not available_conditions:
            print(f"⚠️ No image files were found for '{camera_name}' in {camera_folder}.")
            continue

        for condition in available_conditions:
            image_path = image_paths[condition]
            annotation_path = os.path.join(camera_folder, f'annotations_{condition}.txt')
            print(f"📂 Processing {camera_name} in {camera_folder} - {condition}")

            if camera_name == "specim_iq":
                image_data = np.fromfile(image_path, dtype=np.float32).reshape((512, 204, 512))
                image_data = np.transpose(image_data, (0, 2, 1))
            elif camera_name == "toucan":
                image_data = np.load(image_path)
            else:
                image_data = imread(image_path)
                if camera_name == "eos_m50":
                    image_data = image_data[:, :, ::-1]  # BGR → RGB
                    image_data = image_data.astype(np.float32) / 255.0

            if os.path.exists(annotation_path):
                boxes = load_yolo_annotations_fixed(annotation_path, image_data.shape[1], image_data.shape[0])
                spectral_signatures = extract_spectral_signatures(image_data, boxes, num_bands)
                for label in LABELS:
                    global_data[condition][label].append(spectral_signatures[label])
            else:
                print(f"⚠️ Annotation file not found: {annotation_path}.")

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

    for condition in ["closed", "open"]:
        for label in [0, 1, 2]:
            if global_data[condition][label]:
                signature_stack = np.stack(global_data[condition][label])
                average_signature = np.mean(signature_stack, axis=0)

                if camera_name == "specim_iq":
                    average_signature = np.array(average_signature)[mask_specim]

                min_val, max_val = np.min(average_signature), np.max(average_signature)
                if max_val != min_val:
                    average_signature = (average_signature - min_val) / (max_val - min_val)

                wavelengths_to_plot = [450, 550, 650] if camera_name == "eos_m50" else camera_wavelengths

                plt.plot(wavelengths_to_plot, average_signature,
                         label=f'{condition} - {LABELS[label]}',
                         color=COLORS[LABELS[label]], linestyle=line_styles[condition], linewidth=line_width)

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
