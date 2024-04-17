# Towards Human-in-the-loop Onset Detection: A Transfer Learning Approach for Maracatu

<!-- Code Repository for ISMIR 2024 submission "Towards Human-in-the-loop Onset Detection: A Transfer Learning Approach for Maracatu".   -->
**Note: This repository is provisional and part of a blind submission process for peer review. As such, it currently omits certain elements that could compromise the anonymity required by the review. The complete repository will be updated with the full content, including all identifiable and supplementary items, once the review process has concluded.**


## Contents
- [Towards Human-in-the-loop Onset Detection: A Transfer Learning Approach for Maracatu](#towards-human-in-the-loop-onset-detection-a-transfer-learning-approach-for-maracatu)
  - [Contents](#contents)
  - [Code](#code)
  - [Installation](#installation)
    - [Setting Up the Conda Environment](#setting-up-the-conda-environment)
  - [Additional Figures and Tables](#additional-figures-and-tables)
    - [Results](#results)
    - [Training](#training)
  - [License](#license)

## Code
This directory contains Python scripts for the project. These scripts are provided for reproducibility purposes and are not executable without full environmental setup and data, e.g. base models and Maracatu dataset:
- `code/` 
  - `data/`: Raw and processed datasets used in the study, organized by instrument types. *Note: Maracatu dataset is available upon request to the authors as identified in the paper.*
  - `models/`: Trained models ready for evaluation. *Note: Base models (trained for onset detection `TCNv1` or beat tracking `TCNv2`) are available upon request to the authors as identified in the paper.*
  - `outputs/`: Output files from model runs, categorized by the musical tradition studied.
  - `pkl/`: Serialized Python objects related to the Maracatu datasets.
  - `src/`: Source code of the project.
    - `modules/`: Contains various Python modules for analysis, model definitions, and utilities.
  - `exports/`: Additional exportable resources or processed files.

## Installation
Clone this repository to view all files locally:

```bash
git clone https://github.com/yourusername/Maracatu-Onset-Detection.git
cd Maracatu-Onset-Detection
```

### Setting Up the Conda Environment

This project provides two Conda environment files: 
- `environment.yml` for macOS M1 users which includes specific versions for compatibility.
- `minimum_environment.yml` for users on other platforms, including only the core libraries needed across all systems.


This project uses a Conda environment to ensure all dependencies are managed correctly. The provided `environment.yml` file is tailored for compatibility with macOS M1 systems, which have specific requirements for TensorFlow and Keras versions. **Other users** are advised to start with the `minimum_environment.yml` and make any necessary adjustments based on their specific system requirements.

1. **For macOS M1 Users:**
   ```bash
   conda env create -f environment.yml
   conda activate maracatu
   ```

2. **For non-M1 Users:**
   ```bash
   conda env create -f minimum_environment.yml
   conda activate maracatu_minimum
   ```

Note: This will install all required packages via Conda and pip as specified in the environment file. Please review the package versions and adjust them as necessary.

## Additional Figures and Tables
The `docs/figures` directory is organized into two subdirectories to facilitate navigation and understanding:

### Results
Located under `docs/figures/results/`, this folder contains boxplots related to performance metrics like F-measure, Precision, and Recall:
- Files are named following the convention: `<typeofplot>_<metric>_tol_<tolerance>.<basemodel>.pdf`
  - **typeofplot**: Type of the plot (e.g., `boxplot`)
  - **metric**: Performance metric (e.g., `fmeasure`, `precision`, `recall`)
  - **tolerance**: Tolerance levels indicating the timing precision required (e.g., `10ms`, `15ms`)
  - **basemodel**: The base model version used (e.g., `TCNv1`, `TCNv2`)
- Examples:
  - `boxplot_fmeasure_tol_10ms.TCNv1.pdf`: Boxplot showing F-measure performance at 10ms tolerance using TCN version 1
  - `boxplot_precision_tol_25ms.TCNv2.pdf`: Boxplot showing Precision at 25ms tolerance using TCN version 2

### Training
Located under `docs/figures/training/`, this folder includes loss graphs for various finetuning configurations:
- Files are named as `<plottype>_<instrument>_lr_<learningrate>_ReduceLR_<reducelronplateau>.pdf`
  - **plottype**: Type of the plot (e.g., `losses`)
  - **instrument**: Instrument or dataset variant (e.g., `CAIXA`, `CUICA`)
  - **learningrate**: Learning rate used in the training (e.g., `0.002`, `0.004`)
  - **reducelronplateau**: Indicates whether ReduceLROnPlateau was used (e.g., `True`, `False`)
- Examples:
  - `losses_CAIXA_lr_0.002_ReduceLR_False.pdf`: Loss graph for the CAIXA dataset with a learning rate of 0.002, without ReduceLROnPlateau adjustment.

These documents are provided in high quality to enhance detailed visualization and understanding, not possible within the limited size constraints of the paper.


## License
This project, including all code and figures, is licensed under the GNU General Public License v3.0. By using, distributing, or contributing to this project, you agree to abide by the terms of the GPL v3.0.

The full license text is included in the [LICENSE](LICENSE) file in this repository. Further details and FAQs about the GPL v3.0 can be found at https://www.gnu.org/licenses/gpl-3.0.html.

Please ensure to adhere to the citation requirements outlined in the documentation when using or referencing the project materials.
