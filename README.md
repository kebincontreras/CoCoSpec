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

 
