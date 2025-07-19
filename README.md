# CoCoaSpec

This repository contains the code and resources for the CoCoaSpec project: a multimodal hyperspectral dataset of cocoa beans with physicochemical annotation.

**Topic:** Non-invasive quality assessment of cocoa beans using hyperspectral imaging and physicochemical data, following the Colombian NTC 1252:2021 standard.

**Dataset:** [CoCoSpec on HuggingFace](https://huggingface.co/datasets/ecos-nord-ginp-uis/CoCoaSpec)

**Quick Start:**

**Requirement:** Python **3.12.3 only** (no other version is supported)

- On Windows: run `run_cocospec.bat`
- On Linux:
  ```bash
  bash run_cocospec.sh
  # or, if you get a permission error:
  chmod +x run_cocospec.sh
  ./run_cocospec.sh
  ```

The scripts will set up the environment, download and extract the dataset, and run the main analysis pipeline automatically.

---

## Manual steps (if the automatic scripts fail)

1. **Install Python 3.12.3**
   - Download it from https://www.python.org/downloads/release/python-3123/
   - Make sure to add Python to your PATH during installation.

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv cocospec_env
   # On Windows:
   cocospec_env\Scripts\activate
   # On Linux/Mac:
   source cocospec_env/bin/activate
   ```

3. **Upgrade pip:**
   ```bash
   python -m pip install --upgrade pip
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Download the dataset:**
   - Go to [CoCoSpec on HuggingFace](https://huggingface.co/datasets/ecos-nord-ginp-uis/CoCoaSpec) and download `data.rar`.
   - Place `data.rar` in the `data/` folder of the repository.

6. **Extract the dataset:**
   - Use 7-Zip, WinRAR, or unrar to extract `data.rar` inside the `data/` folder.
   - The folders `resources/` and `scenes/` should appear inside `data/`.

7. **Run the main pipeline:**
   ```bash
   python -m src.Main
   ```

---

 
