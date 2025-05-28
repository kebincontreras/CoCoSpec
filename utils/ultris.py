import os
import numpy as np
from skimage.io import imread
from Utils.processing import *

def load_ultris_data(scene_folder, condition):
    image_path = os.path.join(scene_folder, f"HSI_{condition}.tiff")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found{image_path}")
    return imread(image_path).astype(np.float32)
