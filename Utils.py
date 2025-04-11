import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, exposure
from skimage.io import imread
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
#si
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

#si
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

#si
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

#si
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

#si
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

#si
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

    closed_boxes_toucan = load_yolo_annotations(annotations_closed_toucan, closed_rgb_toucan.shape[1], closed_rgb_toucan.shape[0])
    open_boxes_toucan = load_yolo_annotations(annotations_open_toucan, open_rgb_toucan.shape[1], open_rgb_toucan.shape[0])

    label_ferm_img = imread(label_ferm_path)
    rgb_img = imread(rgb_path)

    # Crear plot
    fig, axs = plt.subplots(4, 2, figsize=(20, 30))  
    fig.suptitle(f'Scene {scene_number}', fontsize=16)

        # Cannon
    axs[0, 0].imshow(label_ferm_img)
    axs[0, 0].set_title("Fermentation Labels")
    axs[0, 1].imshow(rgb_img)
    axs[0, 1].set_title("Cannon Open")

    # Toucan
    axs[1, 0].imshow(closed_rgb_toucan)
    axs[1, 0].set_title("Toucan Closed")
    axs[1, 1].imshow(open_rgb_toucan)
    axs[1, 1].set_title("Toucan Open")

    # Ultris
    axs[2, 0].imshow(closed_rgb_ultris)
    axs[2, 0].set_title("Ultris Closed")
    axs[2, 1].imshow(open_rgb_ultris)
    axs[2, 1].set_title("Ultris Open")


    # Specim IQ
    axs[3, 0].imshow(closed_rgb_iq)
    axs[3, 0].set_title("Specim IQ Closed")
    axs[3, 1].imshow(open_rgb_iq)
    axs[3, 1].set_title("Specim IQ Open")

    for ax_row in axs:
        for ax in ax_row:
            ax.axis('off')

    plt.tight_layout()
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

#si
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
                    wavelengths.extend([float(w.strip()) for w in line.split(",") if w.strip().replace('.', '', 1).isdigit()])
        
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



#def visualizar_pixeles_brillantes(camera_name):
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
                    wavelengths.extend([float(w.strip()) for w in line.split(",") if w.strip().replace('.', '', 1).isdigit()])
        
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


