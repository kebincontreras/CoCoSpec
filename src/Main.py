import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np

from src.const.paths import project_dir, res_dir
from src.utils.PCA import extract_specim_signatures_by_scene, reshape_signatures_df, plot_pc1_colormap_per_property
from src.utils.pca_sampling import compute_random_averaged_signatures_specim, plot_reduction_subplots
from src.utils.processing import extract_wavelengths, plot_normalized_signatures, correct_signatures_with_reference

LABELS = {
    0: "Good",
    1: "Bad",
    2: "Partially"
}

COLORS = {
    "Good": "blue",
    "Bad": "red",
    "Partially": "green"
}

data_dir = "C:/Users/USUARIO/Documents/GitHub/CoCoSpec/data"

def load_reference_from_correction(scene, camera_name, cocoa_condition):
    """
    Loads the mean reference spectrum used in main_spectrum_correction.py for a given scene, camera, and condition.
    Assumes the corrected reference is saved in data/corrections/{camera_name}/scene_{scene:02d}_{cocoa_condition}_reference.npy
    """
    ref_path = os.path.join(data_dir, "corrections", camera_name, f"scene_{scene:02d}_{cocoa_condition}_reference.npy")
    if os.path.exists(ref_path):
        return np.load(ref_path)
    else:
        return None

def find_reference_spectrum(camera_name, scene, cocoa_condition):
    """
    Attempts to find a reference spectrum for a given camera, scene, and condition
    inside the data/resources or data/scenes folders.
    Returns the spectrum as a numpy array, or None if not found.
    """
    # Common possible locations and naming conventions
    possible_paths = [
        os.path.join(data_dir, "resources", camera_name, f"scene_{scene:02d}_{cocoa_condition}_reference.npy"),
        os.path.join(data_dir, "resources", camera_name, f"reference_{scene:02d}_{cocoa_condition}.npy"),
        os.path.join(data_dir, "resources", camera_name, f"reference_{cocoa_condition}.npy"),
        os.path.join(data_dir, "resources", camera_name, "reference.npy"),
        os.path.join(data_dir, "scenes", f"scene_{scene:02d}", camera_name, f"reference_{cocoa_condition}.npy"),
        os.path.join(data_dir, "scenes", f"scene_{scene:02d}", camera_name, "reference.npy"),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return np.load(path)
    return None

def main():
    root_dir = project_dir()
    wavelengths_dir = res_dir() / "wavelengths"
    wavelength_filepaths = {
        "specim_iq": os.path.join(wavelengths_dir, "specim_iq.csv"),
        "eos_m50": os.path.join(wavelengths_dir, "eos_m50.csv"),
        "ultris_sr5": os.path.join(wavelengths_dir, "ultris_sr5.csv"),
        "toucan": os.path.join(wavelengths_dir, "toucan.csv")
    }
    wavelengths = {camera: extract_wavelengths(path) for camera, path in wavelength_filepaths.items()}

    # Only apply reference correction for ultris_sr5 and toucan
    reference_dict = {}
    for camera in ["ultris_sr5", "toucan"]:
        ref = find_reference_spectrum(camera, scene=1, cocoa_condition="open")
        if ref is not None:
            reference_dict[camera] = ref

    # Debug: print if files exist for specim_iq in each scene
    for i in range(1, 20):
        folder = os.path.join(data_dir, "scenes", f"scene_{i:02d}", "specim_iq")
        files_needed = [
            "hsi_closed.dat", "hsi_open.dat",
            "annotations_closed.txt", "annotations_open.txt"
        ]
        for fname in files_needed:
            fpath = os.path.join(folder, fname)
            # No imprimir nada, solo verificar si existe

    # Plot for all cameras, but only pass reference_dict for ultris_sr5 and toucan
    for camera in wavelength_filepaths.keys():
        if camera in reference_dict:
            plot_normalized_signatures(camera, wavelengths, data_dir, reference_dict=reference_dict)
        else:
            plot_normalized_signatures(camera, wavelengths, data_dir)

    # --- ONLY FOR SPECIM_IQ: Dimensionality reduction ---
    avg_open = compute_random_averaged_signatures_specim(root_dir, condition="open")
    avg_closed = compute_random_averaged_signatures_specim(root_dir, condition="closed")
    plot_reduction_subplots(avg_open, avg_closed)
    # ---------------------------------------------------

    csv_path = res_dir() / "Physicochemical.csv"
    spec_df = extract_specim_signatures_by_scene(root_dir, wavelengths)
    spec_df = reshape_signatures_df(spec_df)
    for prop in ["cadmium", "fermentation", "moisture", "polyphenols"]:
        plot_pc1_colormap_per_property(spec_df, csv_path, wavelengths, prop)

if __name__ == "__main__":
    main()