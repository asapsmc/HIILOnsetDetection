# Experiment Reproduction Guide

## Overview
This guide provides detailed instructions on how to reproduce the results published in the paper and how to rerun the experiments from scratch using the provided scripts and data.

## Reproducing Published Results

To reproduce the results already published in the paper, follow these steps:

### Step 1: Activate the conda environment
First, ensure you have cloned the repository to your local machine:

**For macOS M1 Users:**
```bash
conda activate maracatu
```

**For non-M1 Users:**
```bash
conda activate maracatu_minimum
```

### Step 3: Run the Reproduction Script
Execute the script to reproduce the results:

```bash
python reproduce_results.py
```

- This script utilizes saved results covering experiments `rid 000` to `rid 014` for Inductive TL using TCNv1 and `rid 020` to `rid 034` for Transductive TL using TCNv2.
- These correspond to various configurations tested involving 15 different sets of frozen layers.
- The full list of published experiments can be found in the 'exports' folder, `ledger.csv`.


## Rerunning Experiments From Scratch

If you wish to rerun the experiments from scratch, please follow the instructions below:

### Note: Request Base Models
You need to obtain the base model files (.h5), corresponding to `TCNv1` and `TCNv2`, and put them in the `models` folder. 
- Contact the authors as specified in the paper to request access to these models.


### Step 2: Execute Rerun Script
Run the experiments using the following command:

```bash
python rerun_experiments.py
```
- This will finetune the models for each subdataset of the Maracatu dataset.
- The ledger file will be updated with the new run IDs and configurations.
- The finetuned models will be saved in the `exports` folder under this specific run ID (rid).

### Step 3: Execute Evaluation Script

In `evaluate_results.py`, edit the list of run IDs with the new rids. Then, run the script:

```bash
python evaluate_results.py
```
- This will obtain the new onset locations for each of the dataset files, and evaluate them.
- The ledger file will be updated with the new run IDs and configurations.
- The results will be saved as `.csv` in the `results`folder under these specific run ID (rid).

## Additional Information

- **Base Models:** You need the `.h5` base models to rerun the experiments. These models contain the pre-trained models.
- **Output Details:** The output includes loss metrics, accuracy figures, and the saved models post fine-tuning, which are all organized in the `results` folder under respective configuration names.

For any further assistance, refer to the contact information provided in the paper.