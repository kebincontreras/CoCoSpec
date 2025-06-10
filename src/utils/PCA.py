import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from sklearn.decomposition import PCA

from src.loading.loading import extract_spectral_signatures
from src.utils.utils_yolo import load_yolo_annotations


def extract_specim_signatures_by_scene(root_dir, wavelengths):
    num_bands = len(wavelengths["Specim_IQ"])
    data = []

    for scene_id in range(1, 20):
        for condition in ["open", "closed"]:
            folder = os.path.join(root_dir, f"Scenes/Scene_{scene_id:02d}/Specim_IQ")
            img_path = os.path.join(folder, f"HSI_{condition}.dat")
            ann_path = os.path.join(folder, f"annotations_{condition}.txt")
            if not os.path.exists(img_path) or not os.path.exists(ann_path):
                continue

            try:
                img = np.fromfile(img_path, dtype=np.float32).reshape((512, 204, 512)).transpose(0, 2, 1)
                boxes = load_yolo_annotations(ann_path, img.shape[1], img.shape[0])
                all_sigs = []
                for box in boxes:
                    sigs = extract_spectral_signatures(img, [box], num_bands)
                    all_sigs.extend(sigs.values())

                if all_sigs:
                    avg_sig = np.mean(np.stack(all_sigs), axis=0)
                    data.append({"scene": scene_id, "condition": condition, **{f"b{i}": v for i, v in enumerate(avg_sig)}})

            except Exception as e:
                print(f"❌ Error en Scene {scene_id} ({condition}): {e}")
                continue

    return pd.DataFrame(data)

def reshape_signatures_df(spec_df):

    grouped = {}
    for _, row in spec_df.iterrows():
        scene = int(row["scene"])
        condition = row["condition"]
        signature = row.iloc[2:].values.astype(np.float32)
        if scene not in grouped:
            grouped[scene] = {}
        grouped[scene][condition] = signature

    rows = []
    for scene, conditions in grouped.items():
        row = {"scene": scene}
        if "open" in conditions:
            row["open_signature"] = conditions["open"]
        if "closed" in conditions:
            row["closed_signature"] = conditions["closed"]
        rows.append(row)

    return pd.DataFrame(rows)

def plot_pc1_colormap_per_property_0(spec_df, csv_path, wavelengths, property_name):
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 30})
    df_phys = pd.read_csv(csv_path)
    df_phys.columns = df_phys.columns.str.strip().str.lower()

    prop_col = property_name.lower()
    if "fermentacion" in df_phys.columns:
        df_phys["fermentacion"] = df_phys["fermentacion"].astype(str).str.replace("%", "").str.replace(",", ".").astype(float)

    if "scenes" not in df_phys.columns or prop_col not in df_phys.columns:
        print(f"❌ La columna 'scenes' o '{property_name}' no se encontró en el CSV.")
        return

    merged_df = spec_df.merge(df_phys[["scenes", prop_col]], left_on="scene", right_on="scenes")

    all_sigs = []
    markers = []
    for condition in ["open", "closed"]:
        if f"{condition}_signature" not in merged_df.columns:
            continue
        all_sigs.extend(merged_df[f"{condition}_signature"].tolist())
        markers.extend([condition] * len(merged_df))

    all_sigs = np.vstack(all_sigs)
    pcs = PCA(n_components=2).fit_transform(all_sigs)

    prop_vals = np.concatenate([merged_df[prop_col].values] * 2)
    prop_norm = (prop_vals - prop_vals.min()) / (prop_vals.max() - prop_vals.min())
    cmap = cm.get_cmap("RdYlGn")
    colors = cmap(prop_norm)

    plt.figure(figsize=(10, 8))
    for i, condition in enumerate(markers):
        marker = "*" if condition == "open" else "o"
        plt.scatter(pcs[i, 0], pcs[i, 1], c=colors[i].reshape(1, -1), s=140, edgecolors='k', marker=marker)

    plt.scatter([], [], c='gray', marker='*', label='Open', s=140, edgecolors='k')
    plt.scatter([], [], c='gray', marker='o', label='Closed', s=140, edgecolors='k')
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"{property_name.capitalize()} - PCA")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=prop_vals.min(), vmax=prop_vals.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label(property_name.capitalize())
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_pc1_colormap_per_property(spec_df, csv_path, wavelengths, property_name):
    """
    Performs PCA and plots PC1 vs PC2 colored by the selected physicochemical property.
    """
    # Estilo
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 30})

    # Cargar CSV y normalizar nombres de columnas
    df_phys = pd.read_csv(csv_path)
    df_phys.columns = df_phys.columns.str.strip().str.lower()
    prop_col = property_name.lower()

    # Verificar columnas requeridas
    if "scenes" not in df_phys.columns or prop_col not in df_phys.columns:
        print(f"❌ Column 'scenes' or '{property_name}' not found in CSV.")
        return

    # Combinar con firmas espectrales
    merged_df = spec_df.merge(df_phys[["scenes", prop_col]], left_on="scene", right_on="scenes")

    # Firmas y etiquetas
    all_sigs = []
    markers = []
    for condition in ["open", "closed"]:
        col_name = f"{condition}_signature"
        if col_name in merged_df.columns:
            all_sigs.extend(merged_df[col_name].tolist())
            markers.extend([condition] * len(merged_df))

    # PCA
    all_sigs = np.vstack(all_sigs)
    pcs = PCA(n_components=2).fit_transform(all_sigs)

    # Valores de propiedad para colorear
    prop_vals = np.concatenate([merged_df[prop_col].values] * 2)
    prop_norm = (prop_vals - prop_vals.min()) / (prop_vals.max() - prop_vals.min())
    cmap = cm.get_cmap("RdYlGn")
    colors = cmap(prop_norm)

    # Gráfica
    plt.figure(figsize=(10, 8))
    for i, condition in enumerate(markers):
        marker = "*" if condition == "open" else "o"
        plt.scatter(pcs[i, 0], pcs[i, 1], c=colors[i].reshape(1, -1), s=140, edgecolors='k', marker=marker)

    # Leyendas
    plt.scatter([], [], c='gray', marker='*', label='Open', s=140, edgecolors='k')
    plt.scatter([], [], c='gray', marker='o', label='Closed', s=140, edgecolors='k')
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"{property_name.capitalize()} - PCA")
    
    # Barra de color
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=prop_vals.min(), vmax=prop_vals.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label(property_name.capitalize())
    
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
