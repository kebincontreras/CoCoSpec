import os

import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imread
from sklearn.decomposition import PCA

from utils.Utils import view_bright_pixels, load_yolo_annotations_fixed, extract_spectral_signatures


def plot_normalized_signatures(camera_name, wavelengths, root_dir):
    """
    Processes images and generates spectral signature graphs for a specific camera.
    """
    if camera_name not in wavelengths:
        print(f"❌ Cámara '{camera_name}' no encontrada en metadatos.")
        return

    camera_wavelengths = wavelengths[camera_name]
    global_data = {"open": {0: [], 1: [], 2: []}, "closed": {0: [], 1: [], 2: []}}

    # Diccionario de etiquetas para clasificar los espectros
    LABELS = {0: "Good", 1: "Bad", 2: "Partial"}
    COLORS = {"Good": "blue", "Bad": "red", "Partial": "green"}

    usar_reflectancia = camera_name in ["Ultris_SR5", "Toucan"]

    if usar_reflectancia:
        avg_closed_sig, avg_open_sig = view_bright_pixels(camera_name)

    for i in range(1, 20):
        scene_folder = os.path.join(root_dir, f'Scene_{i}')
        camera_folder = os.path.join(scene_folder, camera_name)

        if not os.path.exists(camera_folder):
            print(f"⚠️ Cámara '{camera_name}' not found in {scene_folder}.")
            continue

        # Procesamiento de archivos según la cámara
        if camera_name == "Specim_IQ":
            image_paths = {
                "closed": os.path.join(camera_folder, "HSI_closed.dat"),
                "open": os.path.join(camera_folder, "HSI_open.dat")
            }
            num_bands = 204
        elif camera_name == "EOS_M50":
            image_paths = {"open": os.path.join(camera_folder, "HSI_open.jpg")}
            num_bands = 3
        elif camera_name == "Ultris_SR5":
            image_paths = {
                "closed": os.path.join(camera_folder, "HSI_closed.tiff"),
                "open": os.path.join(camera_folder, "HSI_open.tiff")
            }
            num_bands = 51
        elif camera_name == "Toucan":
            alternative_camera_folder = os.path.join(root_dir, "Toucan", f'Scene_{i}')
            if os.path.exists(alternative_camera_folder):
                camera_folder = alternative_camera_folder
            image_paths = {
                "closed": os.path.join(camera_folder, "HSI_closed.npy"),
                "open": os.path.join(camera_folder, "HSI_open.npy")
            }
            num_bands = len(camera_wavelengths)

        available_conditions = [cond for cond in image_paths if os.path.exists(image_paths[cond])]
        if not available_conditions:
            print(f"⚠️ No image files were found for '{camera_name}' en {camera_folder}.")
            continue

        for condition in available_conditions:
            image_path = image_paths[condition]
            annotation_path = os.path.join(camera_folder, f'annotations_{condition}.txt')
            print(f"📂 Procesando {camera_name} en {camera_folder} - {condition}")

            if camera_name == "Specim_IQ":
                image_data = np.fromfile(image_path, dtype=np.float32).reshape((512, 204, 512))
                image_data = np.transpose(image_data, (0, 2, 1))  # Ajuste de dimensiones
            elif camera_name == "Toucan":
                image_data = np.load(image_path)
            else:
                image_data = imread(image_path)
                if camera_name == "EOS_M50":
                    image_data = image_data.astype(np.float32) / 255.0  # Normalizar entre 0 y 1

            if os.path.exists(annotation_path):
                boxes = load_yolo_annotations_fixed(annotation_path, image_data.shape[1], image_data.shape[0])
                spectral_signatures = extract_spectral_signatures(image_data, boxes, num_bands)
                for label in LABELS.keys():
                    global_data[condition][label].append(spectral_signatures[label])
            else:
                print(f"⚠️ Annotation file not found: {annotation_path}.")

    has_data = any(global_data[condition][label] for condition in ["closed", "open"] for label in LABELS.keys())
    if not has_data:
        print(f"⚠️ No data available for the camera '{camera_name}'.")
        return

    plt.figure(figsize=(15, 5))
    line_styles = {'closed': '-', 'open': '--'}
    line_width = 2.5

    for condition in ["closed", "open"]:
        for label in [0, 1, 2]:
            if global_data[condition][label]:  # Verifica si hay datos
                signature_stack = np.stack(global_data[condition][label])
                average_signature = np.mean(signature_stack, axis=0)

                if usar_reflectancia:
                    average_signature /= avg_open_sig  # Reflectancia

                min_val, max_val = np.min(average_signature), np.max(average_signature)
                if max_val != min_val:
                    average_signature = (average_signature - min_val) / (max_val - min_val)

                wavelengths_to_plot = [450, 550, 650] if camera_name == "EOS_M50" else camera_wavelengths
                plt.plot(wavelengths_to_plot, average_signature,
                         label=f'{condition} - {LABELS[label]}',
                         color=COLORS[LABELS[label]], linestyle=line_styles[condition], linewidth=line_width)

    plt.xlabel('Wavelength (nm)', fontsize=14)
    plt.ylabel('Normalized Reflectance', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.legend()
    plt.grid(True)
    plt.title(f"Spectral Signatures - {camera_name}")
    plt.show()


def plot_pca_spectral_signatures(camera_name, wavelengths, root_dir):
    """
    Computes the PCA of the scene-normalized spectral signatures and plots the first two principal components.
    """
    if camera_name not in wavelengths:
        print(f"❌ Camera '{camera_name}' not found in metadata.")
        return

    camera_wavelengths = wavelengths[camera_name]
    global_scene_data = {"open": [], "closed": []}

    for i in range(1, 20):
        scene_folder = os.path.join(root_dir, f'Scene_{i}')
        camera_folder = os.path.join(scene_folder, camera_name)

        if not os.path.exists(camera_folder):
            print(f"⚠️ Cámara '{camera_name}' not found in {scene_folder}.")
            continue

        # Rutas de imágenes según la cámara
        if camera_name == "Specim_IQ":
            image_paths = {
                "closed": os.path.join(camera_folder, "HSI_closed.dat"),
                "open": os.path.join(camera_folder, "HSI_open.dat")
            }
            num_bands = 204
        elif camera_name == "EOS_M50":
            image_paths = {"open": os.path.join(camera_folder, "HSI_open.jpg")}
            num_bands = 3
        elif camera_name == "Ultris_SR5":
            image_paths = {
                "closed": os.path.join(camera_folder, "HSI_closed.tiff"),
                "open": os.path.join(camera_folder, "HSI_open.tiff")
            }
            num_bands = 51
        elif camera_name == "Toucan":
            alternative_camera_folder = os.path.join(root_dir, "Toucan", f'Scene_{i}')
            if os.path.exists(alternative_camera_folder):
                camera_folder = alternative_camera_folder
            image_paths = {
                "closed": os.path.join(camera_folder, "HSI_closed.npy"),
                "open": os.path.join(camera_folder, "HSI_open.npy")
            }
            num_bands = len(camera_wavelengths)

        available_conditions = [cond for cond in image_paths if os.path.exists(image_paths[cond])]
        if not available_conditions:
            print(f"⚠️ No image files were found for '{camera_name}' en {camera_folder}.")
            continue

        scene_signatures = {"open": [], "closed": []}

        for condition in available_conditions:
            image_path = image_paths[condition]
            annotation_path = os.path.join(camera_folder, f'annotations_{condition}.txt')

            print(f"📂 Procesando {camera_name} en {camera_folder} - {condition}")

            if camera_name == "Specim_IQ":
                image_data = np.fromfile(image_path, dtype=np.float32)
                image_data = image_data.reshape((512, 204, 512))
                image_data = np.transpose(image_data, (0, 2, 1))
            elif camera_name == "Toucan":
                image_data = np.load(image_path)
            else:
                image_data = imread(image_path)
                if camera_name == "EOS_M50":
                    image_data = image_data.astype(np.float32) / 255.0

            if os.path.exists(annotation_path):
                boxes = load_yolo_annotations_fixed(annotation_path, image_data.shape[1], image_data.shape[0])
                spectral_signatures = extract_spectral_signatures(image_data, boxes, num_bands)

                scene_signatures[condition].extend(spectral_signatures.values())

        for condition in ["open", "closed"]:
            if scene_signatures[condition]:
                avg_scene_signature = np.mean(np.stack(scene_signatures[condition]), axis=0)

                min_val, max_val = np.min(avg_scene_signature), np.max(avg_scene_signature)
                if max_val != min_val:
                    avg_scene_signature = (avg_scene_signature - min_val) / (max_val - min_val)

                global_scene_data[condition].append(avg_scene_signature)

    all_scenes = []
    labels = []
    colors = []

    for condition in ["open", "closed"]:
        for idx, signature in enumerate(global_scene_data[condition]):
            scene_number = idx + 1
            all_scenes.append(signature)
            labels.append(f"Scene_{scene_number} - {condition}")

            if 1 <= scene_number <= 10:
                colors.append("green")
            elif scene_number in [11, 12, 13, 14, 15, 16]:
                colors.append("blue")
            elif scene_number in [17, 18, 19]:
                colors.append("yellow")

    if not all_scenes:
        print(f"⚠️ There are insufficient data for PCA in {camera_name}.")
        return

    data_matrix = np.array(all_scenes)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_matrix)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=colors, s=100, edgecolors="black")

    # Leyenda dentro de la gráfica
    legend_labels = {
        "green": "Scenes 1-10",
        "blue": "Scenes 11-16",
        "yellow": "Scenes 17-19"
    }
    handles = [plt.Line2D([0], [0], marker='o', color='w', markersize=10, markerfacecolor=color, label=label)
               for color, label in legend_labels.items()]

    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.title(f"PCA of Spectral Signatures - {camera_name}")

    # ✅ **Ubicar la leyenda dentro de la gráfica**
    plt.legend(handles=handles, title="Grupo de Escenas", loc="upper left", frameon=True)

    plt.grid(True)
    plt.tight_layout()  # Evita que la leyenda se corte
    plt.show()

    explained_variance = pca.explained_variance_ratio_ * 100
    print(f"Variance explained by PC1: {explained_variance[0]:.2f}%")
    print(f"Variance explained by PC2: {explained_variance[1]:.2f}%")
