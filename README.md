# CoCoaSpec

This repository contains the code and resources for the CoCoaSpec project: a multimodal hyperspectral dataset of cocoa beans with physicochemical annotation.

**Topic:** Non-invasive quality assessment of cocoa beans using hyperspectral imaging and physicochemical data, following the Colombian NTC 1252:2021 standard.

**Dataset:** [CoCoSpec on HuggingFace](https://huggingface.co/datasets/ecos-nord-ginp-uis/CoCoaSpec)

**Quick Start:**

**Requisito:** Python **3.12.3 únicamente** (no se admite ninguna otra versión)

- En Windows: ejecuta `run_cocospec.bat`
- En Linux:
  ```bash
  bash run_cocospec.sh
  # o, si ves un error de permisos:
  chmod +x run_cocospec.sh
  ./run_cocospec.sh
  ```

Los scripts configuran el entorno, descargan y extraen el dataset, y ejecutan el pipeline principal automáticamente.


---

## Pasos manuales (si los scripts automáticos fallan)

1. **Instala Python 3.12.3**
   - Descárgalo desde https://www.python.org/downloads/release/python-3123/
   - Asegúrate de agregar Python al PATH durante la instalación.

2. **Crea y activa un entorno virtual:**
   ```bash
   python -m venv cocospec_env
   # En Windows:
   cocospec_env\Scripts\activate
   # En Linux/Mac:
   source cocospec_env/bin/activate
   ```

3. **Actualiza pip:**
   ```bash
   python -m pip install --upgrade pip
   ```

4. **Instala las dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Descarga el dataset:**
   - Ve a [CoCoSpec en HuggingFace](https://huggingface.co/datasets/ecos-nord-ginp-uis/CoCoaSpec) y descarga `data.rar`.
   - Coloca `data.rar` en la carpeta `data/` del repositorio.

6. **Extrae el dataset:**
   - Usa 7-Zip, WinRAR o unrar para extraer `data.rar` dentro de la carpeta `data/`.
   - Deben quedar las carpetas `resources/` y `scenes/` dentro de `data/`.

7. **Ejecuta el pipeline principal:**
   ```bash
   python -m src.Main
   ```

---

 
