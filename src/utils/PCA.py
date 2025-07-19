import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from sklearn.decomposition import PCA

from src.loading.loading import extract_spectral_signatures
from src.utils.utils_yolo import load_yolo_annotations


def extract_specim_signatures_by_scene(root_dir, wavelengths):
    num_bands = len(wavelengths["specim_iq"])
    data = []

    for scene_id in range(1, 20):
        for condition in ["open", "closed"]:
            folder = os.path.join(root_dir, f"data/scenes/scene_{scene_id:02d}/specim_iq")
            img_path = os.path.join(folder, f"hsi_{condition}.dat")
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
                print(f"Error in Scene {scene_id} ({condition}): {e}")
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

def plot_pc1_colormap_per_property(spec_df, csv_path, wavelengths, property_name):
    # This function generates the PCA plot for each physicochemical property.
    # It is called in your main script for each property in ["cadmium", "fermentation", "moisture", "polyphenols"].
    # The plot shows PC1 vs PC2 colored by the selected property.
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 30})

    df_phys = pd.read_csv(csv_path)
    df_phys.columns = df_phys.columns.str.strip().str.lower()
    prop_col = property_name.lower()

    if "scenes" not in df_phys.columns or prop_col not in df_phys.columns:
        print(f"Column 'scenes' or '{property_name}' not found in CSV.")
        return

    merged_df = spec_df.merge(df_phys[["scenes", prop_col]], left_on="scene", right_on="scenes")

    all_sigs = []
    markers = []
    for condition in ["open", "closed"]:
        col_name = f"{condition}_signature"
        if col_name in merged_df.columns:
            all_sigs.extend(merged_df[col_name].tolist())
            markers.extend([condition] * len(merged_df))

    all_sigs = np.vstack(all_sigs)
    pcs = PCA(n_components=2).fit_transform(all_sigs)

    prop_vals = np.concatenate([merged_df[prop_col].values] * 2)
    prop_norm = (prop_vals - prop_vals.min()) / (prop_vals.max() - prop_vals.min())
    cmap = cm.get_cmap("RdYlGn")
    colors = cmap(prop_norm)

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, condition in enumerate(markers):
        marker = "*" if condition == "open" else "o"
        ax.scatter(pcs[i, 0], pcs[i, 1], c=colors[i].reshape(1, -1), s=140, edgecolors='k', marker=marker)

    ax.scatter([], [], c='gray', marker='*', label='Open', s=140, edgecolors='k')
    ax.scatter([], [], c='gray', marker='o', label='Closed', s=140, edgecolors='k')
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"{property_name.capitalize()} - PCA")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=prop_vals.min(), vmax=prop_vals.max()))
    sm.set_array([])
    try:
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(property_name.capitalize())
    except Exception as e:
        print(f"Warning: Could not display colorbar due to backend issue: {e}")

    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    plt.show()

def plot_pc1_colormap_per_property_0(spec_df, csv_path, wavelengths, property_name):
    # This is an alternative/older version of the PCA plot for physicochemical properties.
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 30})
    df_phys = pd.read_csv(csv_path)
    df_phys.columns = df_phys.columns.str.strip().str.lower()

    prop_col = property_name.lower()
    if "fermentacion" in df_phys.columns:
        df_phys["fermentacion"] = df_phys["fermentacion"].astype(str).str.replace("%", "").str.replace(",", ".").astype(float)

    if "scenes" not in df_phys.columns or prop_col not in df_phys.columns:
        print(f"Column 'scenes' or '{property_name}' not found in CSV.")
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

# def plot_pc1_colormap_per_property_0(spec_df, csv_path, wavelengths, property_name):
#     plt.rcParams.update({"font.family": "Times New Roman", "font.size": 30})
#     df_phys = pd.read_csv(csv_path)
#     df_phys.columns = df_phys.columns.str.strip().str.lower()

#     prop_col = property_name.lower()
#     if "fermentacion" in df_phys.columns:
#         df_phys["fermentacion"] = df_phys["fermentacion"].astype(str).str.replace("%", "").str.replace(",", ".").astype(float)

#     if "scenes" not in df_phys.columns or prop_col not in df_phys.columns:
#         print(f"❌ La columna 'scenes' o '{property_name}' no se encontró en el CSV.")
#         return

#     merged_df = spec_df.merge(df_phys[["scenes", prop_col]], left_on="scene", right_on="scenes")

#     all_sigs = []
#     markers = []
#     for condition in ["open", "closed"]:
#         if f"{condition}_signature" not in merged_df.columns:
#             continue
#         all_sigs.extend(merged_df[f"{condition}_signature"].tolist())
#         markers.extend([condition] * len(merged_df))

#     all_sigs = np.vstack(all_sigs)
#     pcs = PCA(n_components=2).fit_transform(all_sigs)

#     prop_vals = np.concatenate([merged_df[prop_col].values] * 2)
#     prop_norm = (prop_vals - prop_vals.min()) / (prop_vals.max() - prop_vals.min())
#     cmap = cm.get_cmap("RdYlGn")
#     colors = cmap(prop_norm)

#     plt.figure(figsize=(10, 8))
#     for i, condition in enumerate(markers):
#         marker = "*" if condition == "open" else "o"
#         plt.scatter(pcs[i, 0], pcs[i, 1], c=colors[i].reshape(1, -1), s=140, edgecolors='k', marker=marker)

#     plt.scatter([], [], c='gray', marker='*', label='Open', s=140, edgecolors='k')
#     plt.scatter([], [], c='gray', marker='o', label='Closed', s=140, edgecolors='k')
#     plt.xlabel("PC1")
#     plt.ylabel("PC2")
#     plt.title(f"{property_name.capitalize()} - PCA")
#     sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=prop_vals.min(), vmax=prop_vals.max()))
#     sm.set_array([])
#     cbar = plt.colorbar(sm)
#     cbar.set_label(property_name.capitalize())
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()


# def plot_pc1_colormap_per_property(spec_df, csv_path, wavelengths, property_name):
#     """
#     Performs PCA and plots PC1 vs PC2 colored by the selected physicochemical property.
#     """
#     # Estilo
#     plt.rcParams.update({"font.family": "Times New Roman", "font.size": 30})

#     # Cargar CSV y normalizar nombres de columnas
#     df_phys = pd.read_csv(csv_path)
#     df_phys.columns = df_phys.columns.str.strip().str.lower()
#     prop_col = property_name.lower()

#     # Verificar columnas requeridas
#     if "scenes" not in df_phys.columns or prop_col not in df_phys.columns:
#         print(f"❌ Column 'scenes' or '{property_name}' not found in CSV.")
#         return

#     # Combinar con firmas espectrales
#     merged_df = spec_df.merge(df_phys[["scenes", prop_col]], left_on="scene", right_on="scenes")

#     # Firmas y etiquetas
#     all_sigs = []
#     markers = []
#     for condition in ["open", "closed"]:
#         col_name = f"{condition}_signature"
#         if col_name in merged_df.columns:
#             all_sigs.extend(merged_df[col_name].tolist())
#             markers.extend([condition] * len(merged_df))

#     # PCA
#     all_sigs = np.vstack(all_sigs)
#     pcs = PCA(n_components=2).fit_transform(all_sigs)

#     # Valores de propiedad para colorear
#     prop_vals = np.concatenate([merged_df[prop_col].values] * 2)
#     prop_norm = (prop_vals - prop_vals.min()) / (prop_vals.max() - prop_vals.min())
#     cmap = cm.get_cmap("RdYlGn")
#     colors = cmap(prop_norm)

#     # Gráfica
#     plt.figure(figsize=(10, 8))
#     for i, condition in enumerate(markers):
#         marker = "*" if condition == "open" else "o"
#         plt.scatter(pcs[i, 0], pcs[i, 1], c=colors[i].reshape(1, -1), s=140, edgecolors='k', marker=marker)

#     # Leyendas
#     plt.scatter([], [], c='gray', marker='*', label='Open', s=140, edgecolors='k')
#     plt.scatter([], [], c='gray', marker='o', label='Closed', s=140, edgecolors='k')
#     plt.xlabel("PC1")
#     plt.ylabel("PC2")
#     plt.title(f"{property_name.capitalize()} - PCA")
    
#     # Barra de color
#     sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=prop_vals.min(), vmax=prop_vals.max()))
#     sm.set_array([])
#     cbar = plt.colorbar(sm)
#     cbar.set_label(property_name.capitalize())
    
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()


def debug_extract_signatures_toucan(data_dir, wavelengths, n_firmas_por_box=10):
    camera_name = "toucan"
    num_bands = len(wavelengths[camera_name])
    for i in range(1, 20):
        scene_folder = os.path.join(data_dir, "scenes", f'scene_{i:02d}')
        camera_folder = os.path.join(scene_folder, camera_name)
        for condition in ["closed", "open"]:
            image_path = os.path.join(camera_folder, f"hsi_{condition}.npy")
            annotation_path = os.path.join(camera_folder, f'annotations_{condition}.txt')
            if not os.path.exists(image_path) or not os.path.exists(annotation_path):
                continue
            try:
                image_data = np.load(image_path)
                print(f"Escena {i}, {condition}: imagen shape {image_data.shape}")
                if image_data.ndim != 3:
                    print(f"❌ Imagen {image_path} no tiene 3 dimensiones, tiene {image_data.shape}")
                    continue
                if image_data.shape[2] != num_bands:
                    print(f"❌ Imagen {image_path} tiene {image_data.shape[2]} bandas, se esperaban {num_bands}")
                    continue
                boxes = load_yolo_annotations(annotation_path, image_data.shape[1], image_data.shape[0])
                print(f"  {len(boxes)} boxes en anotaciones.")
                for (x_min, y_min, x_max, y_max, label) in boxes:
                    label = int(label)
                    x_min, y_min = max(0, x_min), max(0, y_min)
                    x_max, y_max = min(image_data.shape[1], x_max), min(image_data.shape[0], y_max)
                    if x_min < x_max and y_min < y_max:
                        region = image_data[y_min:y_max, x_min:x_max, :]
                        flat_pixels = region.reshape(-1, region.shape[-1])
                        print(f"    Box label {label}: region shape {region.shape}, {flat_pixels.shape[0]} pixels")
                        if flat_pixels.shape[0] == 0:
                            print("    ⚠️ Box vacío, sin píxeles.")
                        else:
                            # Muestra los primeros 3 valores de la primera firma para verificar
                            print(f"    Ejemplo firma: {flat_pixels[0][:3]}")
            except Exception as e:
                print(f"❌ Error en escena {i}, toucan, {condition}: {e}")

def extract_signatures_toucan_df(data_dir, wavelengths, n_firmas_por_box=9):

    import pandas as pd
    camera_name = "toucan"
    num_bands = len(wavelengths[camera_name])
    data = []
    for i in range(1, 20):
        scene_folder = os.path.join(data_dir, "scenes", f'scene_{i:02d}')
        camera_folder = os.path.join(scene_folder, camera_name)
        for condition in ["closed", "open"]:
            image_path = os.path.join(camera_folder, f"hsi_{condition}.npy")
            annotation_path = os.path.join(camera_folder, f'annotations_{condition}.txt')
            if not os.path.exists(image_path) or not os.path.exists(annotation_path):
                continue
            try:
                image_data = np.load(image_path)
                if image_data.ndim != 3 or image_data.shape[2] != num_bands:
                    print(f"❌ Imagen {image_path} shape {image_data.shape}, se esperaban 3D y {num_bands} bandas")
                    continue
                boxes = load_yolo_annotations(annotation_path, image_data.shape[1], image_data.shape[0])
                print(f"📂 Processing toucan in {camera_folder} - {condition}: {len(boxes)} boxes")
                for (x_min, y_min, x_max, y_max, label) in boxes:
                    print(f"  Box: ({x_min}, {y_min}, {x_max}, {y_max}), label={label}")
                    label = int(label)
                    x_min, y_min = max(0, x_min), max(0, y_min)
                    x_max, y_max = min(image_data.shape[1], x_max), min(image_data.shape[0], y_max)
                    if x_min < x_max and y_min < y_max:
                        region = image_data[y_min:y_max, x_min:x_max, :]
                        h, w, _ = region.shape
                        if h < 1 or w < 1:
                            print(f"    ⚠️ Región vacía para box ({x_min},{y_min},{x_max},{y_max})")
                            continue
                        center_y, center_x = h // 2, w // 2
                        offsets = [ (0,0), (-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1) ]
                        firmas_extraidas = 0
                        for dy, dx in offsets:
                            yy = center_y + dy
                            xx = center_x + dx
                            if 0 <= yy < h and 0 <= xx < w:
                                sig = region[yy, xx, :]
                                if sig.shape[0] == num_bands:
                                    data.append({
                                        "scene": i,
                                        "condition": condition,
                                        "label": label,
                                        "firma": sig
                                    })
                                    firmas_extraidas += 1
                        print(f"    Firmas extraídas de este box: {firmas_extraidas}")
                    else:
                        print(f"    ⚠️ Box fuera de límites o inválido: ({x_min},{y_min},{x_max},{y_max})")
            except Exception as e:
                print(f"❌ Error en escena {i}, toucan, {condition}: {e}")
    df = pd.DataFrame(data)
    # Resumen por clase y escena
    for i in range(1, 20):
        for label in [0, 1, 2]:
            n = len(df[(df['scene'] == i) & (df['label'] == label)])
            print(f"Escena {i} - Clase {label}: {n} firmas extraídas.")
    total = len(df)
    print(f"Total de firmas extraídas para toucan: {total}")
    if total == 0:
        print("❌ No se extrajo ninguna firma. Revisa las anotaciones y los archivos de imagen.")
    return df

