import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'Utils'))
from Utils.pca import * 
from Utils.yolo_utils import *
import os   
import pandas as pd
from Utils.processing import *
from Utils.pca_sampling import *
from Utils.pca_sampling import compute_random_averaged_signatures_specim, plot_reduction_subplots

LABELS = {0: "Good", 1: "Bad", 2: "Partially"}
COLORS = {"Good": "blue", "Bad": "red", "Partially": "green"}
root_dir = os.path.dirname(os.path.abspath(__file__))


metadata_dir = os.path.join(root_dir, "Resources", "Metadata")
csv_path = os.path.join(root_dir, "Resources", "Physicochemical.csv")
metadata_paths = {
    "Specim_IQ": os.path.join(metadata_dir, "Metadata_Specim_IQ.hdr"),
    "EOS_M50": os.path.join(metadata_dir, "Metadata_EOS_M50.hdr"),
    "Ultris_SR5": os.path.join(metadata_dir, "Metadata_Ultris_SR5.hdr"),
    "Toucan": os.path.join(metadata_dir, "Metadata_Toucan.hdr")
}
wavelengths = {camera: extract_wavelengths(path) for camera, path in metadata_paths.items()}
for camera in metadata_paths.keys():
    plot_normalized_signatures(camera, wavelengths, root_dir)

avg_open = compute_random_averaged_signatures_specim(root_dir, condition="open")
avg_closed = compute_random_averaged_signatures_specim(root_dir, condition="closed")
plot_reduction_subplots(avg_open, avg_closed)


spec_df = extract_specim_signatures_by_scene(root_dir, wavelengths)
spec_df = reshape_signatures_df(spec_df)
for prop in ["cadmium", "fermentation", "moisture", "polyphenols"]:
    plot_pc1_colormap_per_property(spec_df, csv_path, wavelengths, prop)
