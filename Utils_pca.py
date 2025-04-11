import os  
import numpy as np  
import matplotlib.pyplot as plt  
from skimage.io import imread  
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from Utils_pca import extract_wavelengths, extract_spectral_signatures, load_yolo_annotations_fixed



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




def extract_wavelengths(metadata_path):
    """Extracts wavelength values from a .hdr metadata file and handles the special case of Ultris_SR5."""
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

                   
                    wavelengths.extend([float(w.strip()) for w in line.split(",") if w.strip().replace('.', '', 1).isdigit()])
        

        if wavelengths:
            print(f"✅ Correctly extracted wavelengths for {metadata_path.split('/')[-1]}: {wavelengths[:10]} ...")
        else:
            print(f"❌ No wavelength values ​​were found in {metadata_path}.")

        return wavelengths if wavelengths else None

    except FileNotFoundError:
        print(f"❌ File not found: {metadata_path}")
        return None




def plot_pca_spectral_signatures(camera_name, wavelengths, root_dir):
    """
    Performs PCA of the scene-normalized spectral signatures and graphs the first two components.
    """
    if camera_name not in wavelengths:
        print(f"❌ Camera '{camera_name}' not found in metadata.")
        return

    global_scene_data = {"open": [], "closed": []}  # Estructura para almacenar firmas por escena

    for i in range(1, 20):  # Escenas 1 a 19
        scene_folder = os.path.join(root_dir, f'Scene_{i}')
        camera_folder = os.path.join(scene_folder, camera_name)

        if not os.path.exists(camera_folder):
            print(f"⚠️ Cámara '{camera_name}' not found in {scene_folder}.")
            continue

        # Definir rutas de imágenes
        image_paths = {
            "closed": os.path.join(camera_folder, "HSI_closed.dat" if camera_name == "Specim_IQ" else "HSI_closed.tiff" if camera_name == "Ultris_SR5" else "HSI_closed.npy"),
            "open": os.path.join(camera_folder, "HSI_open.dat" if camera_name == "Specim_IQ" else "HSI_open.tiff" if camera_name == "Ultris_SR5" else "HSI_open.jpg" if camera_name == "EOS_M50" else "HSI_open.npy")
        }

        for condition, path in image_paths.items():
            if not os.path.exists(path):
                continue
            image_data = imread(path) if camera_name != "Specim_IQ" else np.fromfile(path, dtype=np.float32).reshape((512, 204, 512)).transpose(0, 2, 1)
            annotation_path = os.path.join(camera_folder, f'annotations_{condition}.txt')
            if os.path.exists(annotation_path):
                boxes = load_yolo_annotations_fixed(annotation_path, image_data.shape[1], image_data.shape[0])
                spectral_signatures = extract_spectral_signatures(image_data, boxes, len(wavelengths[camera_name]))
                if spectral_signatures:
                    avg_scene_signature = np.mean([sig for sig in spectral_signatures.values()], axis=0)
                    global_scene_data[condition].append(avg_scene_signature)

    all_scenes = [sig for sigs in global_scene_data.values() for sig in sigs]
    if not all_scenes:
        print(f"⚠️ There are insufficient data for PCA in {camera_name}.")
        return

    data_matrix = np.array(all_scenes)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_matrix)

