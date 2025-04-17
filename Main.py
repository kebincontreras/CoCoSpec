import os

import numpy as np

from utils.Utils import extract_wavelengths
from utils.spectral_signatures import plot_normalized_signatures, plot_pca_spectral_signatures
from utils.annotations import plot_images_and_annotations


def initialize_variables():
    return {
        "all_closed_spectra": [],
        "all_open_spectra": [],
        "all_wavelengths": None,
        "wavelengths": np.linspace(400, 1000, 204),
    }


variables = initialize_variables()
LABELS = {
    0: "Good",
    1: "Bad",
    2: "Partially",
}
COLORS = {
    "Good": "blue",
    "Bad": "red",
    "Partially": "green",
}
root_dir = os.path.dirname(os.path.abspath(__file__))
metadata_dir = os.path.join(root_dir, "Metadata")
metadata_paths = {
    "Specim_IQ": os.path.join(metadata_dir, "Metadata_Specim_IQ.hdr"),
    "EOS_M50": os.path.join(metadata_dir, "Metadata_EOS_M50.hdr"),
    "Ultris_SR5": os.path.join(metadata_dir, "Metadata_Ultris_SR5.hdr"),
    "Toucan": os.path.join(metadata_dir, "Metadata_Toucan.hdr")
}
wavelengths = {
    camera: extract_wavelengths(path)
    for camera, path in metadata_paths.items()
}

for i in range(19, 20):
    scene_folder = os.path.join(root_dir, f'Scene_{i}')
    plot_images_and_annotations(scene_folder, i, variables["wavelengths"])

for camera in metadata_paths.keys():
    plot_normalized_signatures(camera, wavelengths, root_dir)
    plot_pca_spectral_signatures(camera, wavelengths, root_dir)
