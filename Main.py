import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, exposure
from skimage.io import imread
from scipy.interpolate import interp1d
from Utils import *
# Inicialización de variables
def initialize_variables():
    return {
        "all_closed_spectra": [],
        "all_open_spectra": [],
        "all_wavelengths": None,
        "wavelengths": np.linspace(400, 1000, 204),
    }
variables = initialize_variables()
LABELS = {0: "Good", 1: "Bad", 2: "Partially"}
COLORS = {"Good": "blue", "Bad": "red", "Partially": "green"}
root_dir = os.path.dirname(os.path.abspath(__file__))
for i in range(19, 20):
    scene_folder = os.path.join(root_dir, f'Scene_{i}')
    plot_images_and_annotations(scene_folder, i, variables["wavelengths"])
metadata_dir = os.path.join(root_dir, "Metadata")
metadata_paths = {
    "Specim_IQ": os.path.join(metadata_dir, "Metadata_Specim_IQ.hdr"),
    "EOS_M50": os.path.join(metadata_dir, "Metadata_EOS_M50.hdr"),
    "Ultris_SR5": os.path.join(metadata_dir, "Metadata_Ultris_SR5.hdr"),
    "Toucan": os.path.join(metadata_dir, "Metadata_Toucan.hdr")
}
wavelengths = {camera: extract_wavelengths(path) for camera, path in metadata_paths.items()}
for camera in metadata_paths.keys():
    plot_normalized_signatures(camera, wavelengths, root_dir)
    plot_pca_spectral_signatures(camera, wavelengths, root_dir)

