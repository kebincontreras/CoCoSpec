# CoCoaSpec

This repository contains the code and resources for the CoCoaSpec project: a multimodal hyperspectral dataset of cocoa beans with physicochemical annotation.

**Topic:** Non-invasive quality assessment of cocoa beans using hyperspectral imaging and physicochemical data, following the Colombian NTC 1252:2021 standard.

**Dataset:** [CoCoSpec on HuggingFace](https://huggingface.co/datasets/ecos-nord-ginp-uis/CoCoaSpec)

**Quick Start:**
- On Windows: run `run_cocospec.bat`
- On Linux:
  ```bash
  bash run_cocospec.sh
  # or, if you get a permission error:
  chmod +x run_cocospec.sh
  ./run_cocospec.sh

The scripts will set up the environment, download and extract the dataset, and run the main analysis pipeline automatically.

