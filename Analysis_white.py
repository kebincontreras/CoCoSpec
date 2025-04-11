import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import time

def extract_rgb_from_hsi(image_data):
    num_bands = image_data.shape[2]
    if num_bands < 3:
        print("⚠️ No hay suficientes bandas para generar RGB, usando solo la primera banda.")
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

def visualizar_pixeles_brillantes(camera_name):
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
        return
    
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

        closed_rgb = extract_rgb_from_hsi(closed_data)
        open_rgb = extract_rgb_from_hsi(open_data)
        
        closed_highlighted, bright_pixels_closed = find_brightest_pixels(closed_rgb, num_pixels)
        open_highlighted, bright_pixels_open = find_brightest_pixels(open_rgb, num_pixels)
        
        closed_signature = extract_spectral_signature(closed_data, bright_pixels_closed)
        open_signature = extract_spectral_signature(open_data, bright_pixels_open)
        all_signatures.append((closed_signature, open_signature))
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(closed_highlighted)
        axes[0].set_title(f"Scene {i} - Closed ({camera_name})")
        axes[0].axis("off")
        axes[1].imshow(open_highlighted)
        axes[1].set_title(f"Scene {i} - Open ({camera_name})")
        axes[1].axis("off")
        plt.tight_layout()
        plt.show()
        time.sleep(1)  # Mostrar cada imagen por 1 segundo
    
    if all_signatures:
        plt.figure(figsize=(10, 5))
        for i, (closed_sig, open_sig) in enumerate(all_signatures):
            wavelengths = np.linspace(400, 700, num_bands)  
            plt.plot(wavelengths, closed_sig, label=f'Scene {i+1} - Closed', linestyle='-')
            plt.plot(wavelengths, open_sig, label=f'Scene {i+1} - Open', linestyle='--')
        
        avg_closed_sig = np.mean([sig[0] for sig in all_signatures], axis=0)
        avg_open_sig = np.mean([sig[1] for sig in all_signatures], axis=0)
        
        plt.plot(wavelengths, avg_closed_sig, label='Promedio - Closed', linestyle='-', linewidth=2, color='black')
        plt.plot(wavelengths, avg_open_sig, label='Promedio - Open', linestyle='--', linewidth=2, color='gray')
        
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Reflectance')
        plt.title(f'Firmas Espectrales - {camera_name}')
        plt.legend()
        plt.grid(True)
        plt.show()

visualizar_pixeles_brillantes("Ultris_SR5")
visualizar_pixeles_brillantes("Toucan")