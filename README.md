# Towards Human-in-the-loop Onset Detection: A Transfer Learning Approach for *Maracatu*

<!-- Code Repository for ISMIR 2024 submission "Towards Human-in-the-loop Onset Detection: A Transfer Learning Approach for Maracatu".   -->
**Note: This repository is provisional and part of a blind submission process for peer review. As such, it currently omits certain elements that could compromise the anonymity required by the review. The complete repository will be updated with the full content, including all identifiable and supplementary items, once the review process has concluded.**


## Contents
- [Towards Human-in-the-loop Onset Detection: A Transfer Learning Approach for *Maracatu*](#towards-human-in-the-loop-onset-detection-a-transfer-learning-approach-for-maracatu)
  - [Contents](#contents)
  - [Code](#code)
  - [Installation](#installation)
    - [Setting Up the Conda Environment](#setting-up-the-conda-environment)
  - [How to Reproduce the Experiment Results](#how-to-reproduce-the-experiment-results)
  - [How to Rerun the Full Experiment Locally](#how-to-rerun-the-full-experiment-locally)
  - [Additional Figures and Tables](#additional-figures-and-tables)
  - [License](#license)

## Code
This directory contains Python scripts for the project. These scripts are provided for reproducibility purposes and are not executable without full environmental setup and data, e.g. base models and *Maracatu* dataset:
- `code/` 
  - `data/`: Raw and processed datasets used in the study, organized by instrument types. *Note: Maracatu dataset is available upon request to the authors as identified in the paper.*
  - `models/`: Trained models ready for evaluation. *Note: Base models (trained for onset detection `TCNv1` or beat tracking `TCNv2`) are available upon request to the authors as identified in the paper.*
  - `outputs/`: Output files from model runs, categorized by the musical tradition studied.
  - `pkl/`: Serialized Python objects related to the *Maracatu* datasets.
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

**For macOS M1 Users:**
   ```bash
   conda env create -f environment.yml
   conda activate maracatu
   ```

**For non-M1 Users:**
   ```bash
   conda env create -f minimum_environment.yml
   conda activate maracatu_minimum
   ```

Note: This will install all required packages via Conda and pip as specified in the environment file. Please review the package versions and adjust them as necessary.

## How to Reproduce the Experiment Results
To reproduce the paper results, execute:

1. Activate the Environment

    **macOS M1 Users:**
      ```bash
      conda activate maracatu
      ```
    **non-M1 Users:**
      ```bash
      conda activate maracatu_minimum
      ```

2. Run the Reproduction Script
    ```bash
    python evaluate_results.py
    ``` 

    This script:
    - Uses saved results for Inductive TL with TCNv1 (`rid` 000 to 014) and Transductive TL with TCNv2 (`rid` 020 to 034), covering various frozen layer configurations. 
    - Results are outputted as `.csv` files in the `/results` folder under the respective run IDs.


## How to Rerun the Full Experiment Locally

1. Obtain Base Models and *Maracatu* dataset

    Request the `.h5` base model files for TCNv1 and TCNv2 from the corresponding authors (as identified in the paper), and place them in the `/models` folder.

2. Obtain *Maracatu* dataset audio (`*.flac`) from authors and place it in the `/data/**/audio` folder.

1. Run the Experiment Script

    ```bash
    python rerun_experiments.py
    ```

    This script: 
    - Fine-tunes models for each of Maracatu subdatasets.
    - Updates the ledger with new run IDs.
    - Saves finetuned models in the `/exports` folder with respective run IDs.
    - Creates predictions for each baseline model (madmomRNN, madmomCNN) and saves them in the `/exports` folder.
    - Creates original and finetuned model predictions for each Maracatu subdataset and saves them in the `/outputs` folder.


## Additional Figures and Tables
The `docs` folder is organized into two subdirectories to facilitate navigation and understanding

- `docs/figures/`, this folder contains boxplots (`*.pdf`) related to performance metrics like F-measure, Precision, and Recall:
    - Files are named following the convention: `<typeofplot>_<metric>_tol_<tolerance>.<basemodel>.pdf`
      - **typeofplot**: Type of the plot (e.g., `boxplot`)
      - **metric**: Performance metric (e.g., `fmeasure`, `precision`, `recall`)
      - **tolerance**: Tolerance levels indicating the timing precision required (e.g., `10ms`, `15ms`)
      - **basemodel**: The base model version used (e.g., `TCNv1`, `TCNv2`)
    - Examples:
      - `boxplot_fmeasure_tol_10ms.TCNv1.pdf`: Boxplot showing F-measure performance at 10ms tolerance using TCN version 1
      - `boxplot_precision_tol_25ms.TCNv2.pdf`: Boxplot showing Precision at 25ms tolerance using TCN version 2
      - Note: The figures whose names start with `compact_` are the ones used in the paper.

      These documents are provided in high quality to enhance detailed visualization and understanding, not possible within the limited size constraints of the paper.
- `docs/tables/`, this folder contains comprehensive tables (`*.pdf`) detailing performance metrics such as F-measure, Precision, and Recall, along with counts of True Positives (TP), False Positives (FP), and False Negatives (FN) across the *Maracatu* dataset for baseline and finetuned models on the TCNv1 and TCNv2 network.


## License
This project, including all code and figures, is licensed under the GNU General Public License v3.0. By using, distributing, or contributing to this project, you agree to abide by the terms of the GPL v3.0.

The full license text is included in the [LICENSE](LICENSE) file in this repository. Further details and FAQs about the GPL v3.0 can be found at https://www.gnu.org/licenses/gpl-3.0.html.

Please ensure to adhere to the citation requirements outlined in the documentation when using or referencing the project materials.
