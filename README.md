# Comparing different Classification Models
This repository shows how to create and train different models to run classification on an unbalanced dataset.

The models as well as the classification results are tracked using MLFlow.

# Installing required packages

The [environment.yml](environment.yml) can be used to easily install all required packages to run the code in this repo.

## Installation

1. Install Miniforge.

   Download an installer from
   [Miniforge](https://github.com/conda-forge/miniforge). Ideally, the latest
   version.

2. Add current directory to python path by running the [export_ppath.sh](export_ppath.sh) script


3. Install environment with Conda:

```bash
conda env create -f environment.yml
```

4. Activate environment:
```bash
conda activate class-prj
```

