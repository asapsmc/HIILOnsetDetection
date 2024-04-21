import logging
import os
import pathlib
import pickle
import pickletools
import sys
import warnings

import madmom
import modules.definitions as dfn
import modules.mod_onsets as ons
import modules.report_common as rpc
import modules.utils as utl
import numpy as np
import pandas as pd
from madmom.features.onsets import CNNOnsetProcessor, RNNOnsetProcessor

warnings.filterwarnings('ignore')


def ext_load_models_predictions(filenames, paths, ext_archs=utl.EXT_ARCHS):
    """
    Load predictions for external models for all files.

    Parameters:
    - filenames (list): The names of the files to load predictions from.
    - paths (dict): A dictionary containing the paths to the predictions directory.
    - ext_archs (list, optional): A list of external models. Defaults to utl.EXT_ARCHS.

    Returns:
    - all_models_preds (dict): A dictionary where keys are model names and values are lists of predictions for each file.

    This function loads predictions for external models for all specified files. It iterates over each external model and then over each file, loading the corresponding prediction file using the filenames and paths provided. The loaded predictions for each model are organized in a dictionary, with each key being a model and its value being the list of predictions for all files.

    Example usage:
    all_models_preds = ext_load_models_predictions(['file1', 'file2'], {'predictions': '/path/to/predictions'}, ext_archs=['model1', 'model2'])
    """
    path = paths["predictions"]
    all_models_preds = {model: [] for model in ext_archs}
    for filename in filenames:
        for model in ext_archs:
            pred_path = f'{path}/{model}.{filename}.pred.npy'
            pred = np.load(pred_path, allow_pickle=True)
            all_models_preds[model].append(pred)
    return all_models_preds


def ext_load_models_prediction(filename, paths, ext_archs=utl.EXT_ARCHS):
    """
    Load predictions for external models.

    Parameters:
    - filename (str): The name of the file to load predictions from.
    - paths (dict): A dictionary containing the paths to the predictions directory.
    - ext_archs (list, optional): A list of external models. Defaults to utl.EXT_ARCHS.

    Returns:
    - all_preds (list): A list of predictions for each external model.

    This function loads predictions for external models from the specified file. It iterates over each external model and loads the corresponding prediction file using the filename and paths provided. The loaded predictions are then appended to a list and returned.

    Example usage:
    ext_load_predictions('example_file', {'predictions': '/path/to/predictions'}, ext_archs=['model1', 'model2'])
    """
    path = paths["predictions"]
    all_preds = []
    for model in ext_archs:
        pred = np.load(f'{path}/{model}.{filename}.pred.npy', allow_pickle=True)
        all_preds.append(pred)
    return all_preds


def ext_evaluation_for_datasets(datasets, tol=0.025, ext_archs=utl.EXT_ARCHS):
    """
    Perform evaluation of external models on multiple datasets.

    Parameters:
    - datasets (list): A list of datasets to evaluate the external models on.
    - tol (float, optional): The tolerance value in seconds for evaluating the onsets. Defaults to 0.025.
    - ext_archs (list, optional): A list of external models to evaluate. Defaults to utl.EXT_ARCHS.

    Returns:
    None

    This function performs evaluation of external models on multiple datasets. It loads the annotations and predictions for each dataset and calculates the evaluation metrics for each external model. The evaluation results are saved in CSV files.

    The function follows these steps:
    1. Iterate over the datasets and subdatasets.
    2. Load the annotations and predictions for each subdataset.
    3. Calculate the onsets for each external model using the predictions.
    4. Save the evaluation results for each external model in CSV files.

    Note: The function assumes that the annotations and predictions are stored in pickle files and the evaluation results are saved in a results directory.

    Example usage:
    ext_evaluation_for_datasets(['dataset1', 'dataset2'], tol=0.05, ext_archs=['model1', 'model2'])
    """
    stdm_columns = rpc.STD_METRICS.get('ANA')
    tol_ms = int(tol * 1000)
    for dataset, subdataset, pkl_path in utl.iterate_datasets_and_subdatasets(datasets):
        with open(pkl_path, 'rb') as f:
            db = pickle.load(f)
        paths = utl.get_path(dataset, subdataset)
        res_path = paths["results"]
        logging.info('making ext_archs evaluation for %s', subdataset)
        file_indices = utl.get_evaluation_file_indices(dataset, len(db.files))
        dets_per_model = {x: [] for x in ext_archs}
        lst_anns = []
        for idx in file_indices:
            single_file = db.files[idx]
            anns = db.annotations[idx]
            lst_anns.append(anns)
            all_preds = ext_load_models_prediction(single_file, paths)
            for model, pred in zip(utl.EXT_ARCHS, all_preds):
                dets = ons.get_onsets_from_preds(pred, thresh=0.3, scale='audio')
                dets_per_model[model].append(dets)

        logging.info('\n saving subdataset evaluation in %s', res_path)
        for model in ext_archs:
            evals, se, me = ons.evaluate_onsets(dets_per_model[model], lst_anns)
            ev_std = [e for e in evals]
            filtered_std = [[getattr(ev, measure) for measure in stdm_columns] for ev in ev_std]
            df_std = pd.DataFrame(filtered_std, columns=stdm_columns)
            df_std.to_csv(os.path.join(res_path, f'{model}.{dataset}.{subdataset}.{tol_ms}.std.csv'))


def get_ext_results(datasets, tol=0.025, ext_archs=utl.EXT_ARCHS):
    """
    Retrieve evaluation results for specified datasets.

    Args:
    - datasets (list): List of datasets to retrieve results for.
    - tol (float, optional): Tolerance value in seconds for results retrieval. Defaults to 0.025.
    - ext_archs (list, optional): External models to retrieve results for. Defaults to utl.EXT_ARCHS.

    Returns:
    - results: Dictionary with model names as keys and pandas DataFrames of results as values.
    """
    tol_ms = int(tol * 1000)
    for dataset, subdataset, _ in utl.iterate_datasets_and_subdatasets(datasets):
        paths = utl.get_path(dataset, subdataset)
        res_path = paths["results"]
        results = {model: None for model in ext_archs}
        for model in ext_archs:
            df = pd.read_csv(os.path.join(res_path, f'{model}.{dataset}.{subdataset}.{tol_ms}.std.csv'), index_col=0)
            results[model] = df
    return results


if __name__ == "__main__":
    test_datasets = ['onset_db']
    # ext_create_predictions_for_datasets(datasets)
    # ext_evaluation_for_datasets(datasets)
    get_ext_results(test_datasets)
    print('DONE')
