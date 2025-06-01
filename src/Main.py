import os

from const.paths import project_dir, res_dir
from src.utils.PCA import extract_specim_signatures_by_scene, reshape_signatures_df, plot_pc1_colormap_per_property
from src.utils.pca_sampling import compute_random_averaged_signatures_specim, plot_reduction_subplots
from src.utils.processing import extract_wavelengths, plot_normalized_signatures

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


def main():
    root_dir = project_dir()

    metadata_dir = res_dir() / "metadata"
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

    csv_path = res_dir() / "Physicochemical.csv"
    spec_df = extract_specim_signatures_by_scene(root_dir, wavelengths)
    spec_df = reshape_signatures_df(spec_df)
    for prop in ["cadmium", "fermentation", "moisture", "polyphenols"]:
        plot_pc1_colormap_per_property(spec_df, csv_path, wavelengths, prop)


if __name__ == "__main__":
    main()
