import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings("ignore", message=".*n_jobs value 1 overridden.*")
import umap
from Utils.yolo_utils import load_yolo_annotations
from Utils.processing import extract_spectral_signatures

def compute_random_averaged_signatures_specim(root_dir, condition, num_combinations=100, group_size=50):

    data = {0: [], 1: [], 2: []}
    num_bands = 204

    for scene_id in range(1, 20):
        folder = os.path.join(root_dir, f"Scenes/Scene_{scene_id:02d}/Specim_IQ")
        img_path = os.path.join(folder, f"HSI_{condition}.dat")
        ann_path = os.path.join(folder, f"annotations_{condition}.txt")
        if not os.path.exists(img_path) or not os.path.exists(ann_path):
            continue
        try:
            img = np.fromfile(img_path, dtype=np.float32).reshape((512, 204, 512)).transpose(0, 2, 1)
            boxes = load_yolo_annotations(ann_path, img.shape[1], img.shape[0])
            for (x_min, y_min, x_max, y_max, label) in boxes:
                label = int(label)
                x_min, y_min = max(0, x_min), max(0, y_min)
                x_max, y_max = min(img.shape[1], x_max), min(img.shape[0], y_max)
                if x_min < x_max and y_min < y_max:
                    region = img[y_min:y_max, x_min:x_max, :]
                    mean_spectrum = np.mean(region, axis=(0, 1))
                    if mean_spectrum.shape[0] == num_bands:
                        data[label].append(mean_spectrum)
        except Exception as e:
            print(f"Error Scene {scene_id} ({condition}): {e}")
            continue

    averaged_data = {0: [], 1: [], 2: []}
    for label in data:
        sigs = np.array(data[label])
        if len(sigs) < group_size:
            print(f"⚠️ Clase {label} tiene solo {len(sigs)} muestras. Requiere mínimo {group_size}.")
            continue
        for _ in range(num_combinations):
            idx = np.random.choice(len(sigs), group_size, replace=False)
            avg_sig = np.mean(sigs[idx], axis=0)
            averaged_data[label].append(avg_sig)

    return averaged_data

def plot_reduction_subplots(averaged_data_open, averaged_data_closed):

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 30
    })

    methods = {
        "PCA": PCA(n_components=2),
        "t-SNE": TSNE(n_components=2, random_state=42),
        "UMAP": umap.UMAP(n_components=2, random_state=42)
    }

    fig, axs = plt.subplots(2, 3, figsize=(24, 16))
    condition_labels = {'open': averaged_data_open, 'closed': averaged_data_closed}

    for col, (method_name, reducer) in enumerate(methods.items()):
        for row, (condition, data_dict) in enumerate(condition_labels.items()):
            all_data = []
            labels = []
            for label in data_dict:
                all_data.extend(data_dict[label])
                labels.extend([label] * len(data_dict[label]))

            all_data = np.array(all_data)
            labels = np.array(labels)

            if method_name == "PCA":
                reducer.fit(all_data)
                explained = reducer.explained_variance_ratio_
                print(f"📊 Variance explained ({condition} - PCA): PC1 = {explained[0]:.2%}, PC2 = {explained[1]:.2%}")
            reduced = reducer.fit_transform(all_data)

            ax = axs[row, col]
            for label, color, name in zip([0, 1, 2], ['blue', 'red', 'green'], ['Good', 'Bad', 'Partial']):
                idx = labels == label
                ax.scatter(reduced[idx, 0], reduced[idx, 1], label=name, c=color, alpha=0.7)

            if row == 0:
                ax.set_title(method_name)
            if col == 0:
                ax.set_ylabel(f"{condition.capitalize()}")

            ax.tick_params(axis='both', which='major', labelsize=30)
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontname("Times New Roman")

            ax.grid(True)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()
