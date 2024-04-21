import logging
import pickle
from pathlib import Path

import keras.backend as K
import madmom
import numpy as np
from keras.models import load_model
from madmom.features.onsets import CNNOnsetProcessor, RNNOnsetProcessor

import modules.config as cfg
import modules.definitions as dfn
import modules.utils as utl


def finetune(model, arch, finetuning_data, callbacks=None, verbose=False):
    """
    Fine-tunes a model based on the provided architecture configuration and data.

    Args:
        model (tf.keras.Model): The model to fine-tune.
        arch (constant): Architecture type, key to access architecture_configs.
        finetuning_data (data): Data to be used for fine-tuning the model.
        callbacks (list, optional): Additional callbacks to be used during training.
        verbose (bool, optional): Verbosity mode. Defaults to True.

    Returns:
        tf.keras.callbacks.History: The history object generated after training the model.
    """
    config = dfn.architecture_configs[arch]

    optimizer = config['optimizer'](config['learning_rate'], clipnorm=0.5)
    default_callbacks = config['callbacks']()  # Initialize callbacks as per configuration
    if callbacks:
        default_callbacks.extend(callbacks)

    model.compile(optimizer=optimizer, **config['compile_args'])

    history = model.fit(finetuning_data, epochs=config['num_epochs'], callbacks=default_callbacks, verbose=verbose)

    return history


def finetune_target(dataset, subdataset, r_id, type_ft, fz_layers_idx, arch):
    """
    Runs a single fine-tuning experiment on a specified subdataset for a given duration of audio.

    Parameters:
    - subdataset: The subdataset to fine-tune on.
    - r_id: Unique identifier for the run, used for logging and saving outputs.
    - finetuning_config: Configuration dictionary for fine-tuning.
    - arch: Model architecture key.
    - n_seconds: Duration of audio to use from the start of each file, in seconds.
    - verbose: Verbosity level for model training, defaults to True.

    Returns:
    True if the experiment completes successfully.
    """
    paths = utl.get_path(dataset, subdataset)

    with open(f'{utl.PKL_PATH}/{dataset}/{subdataset}.pkl', 'rb') as file:
        ft_db = pickle.load(file)
    ft_target = ft_db.files[0]

    type_ft_dict = cfg.decode_type_ft(type_ft)
    ft_start, ft_stop = 0, type_ft_dict['duration']
    ft_dataset = dfn.create_retrain_dataset(dataset, subdataset, ft_target, start=ft_start, stop=ft_stop)
    ft_data, _ = dfn.get_widened_data_sequence(arch, ft_dataset)

    K.clear_session()
    model = load_model(utl.MDL_PATH / f'onset_{arch}.h5', compile=False)
    model = dfn.set_trainable_layers(model, fz_layers_idx)
    history = finetune(model, arch, ft_data)

    # Save model, training history and finetuning properties
    prefix = cfg.encode_prefix(r_id, type_ft, arch)
    print(f'    saving model to {paths["final"]}/{prefix}.{subdataset}_finetuned.h5')
    model_path = f'{paths["final"]}/{prefix}.{subdataset}_finetuned.h5'
    history_path = f'{paths["stats"]}/{prefix}.{subdataset}_finetune_history.pkl'
    model.save(model_path)
    with open(history_path, 'wb') as file:
        pickle.dump(history.history, file)

    return True


def run_finetuning_experiments(datasets, exp_props, arch, start_rid=None):
    """
    Runs a series of fine-tuning experiments on specified targets based on the granularity.

    Parameters:
    - granularity: The level at which fine-tuning should be performed ('dataset', 'subdataset', or 'file').
    - targets: List of targets to process (datasets, subdatasets, or files). Must be a non-empty list.
    - exp_props: Dictionary containing the properties for the series of experiments.
    - arch: Model architecture key, defaulting to 'TCNV1'.

    Returns:
    - True if all experiments complete successfully.
    """
    if exp_props is None:
        raise ValueError("Experiment properties must be provided.")

    df_ledger = cfg.load_ledger()
    now_rid = start_rid
    for group_layers_idx, fz_limits_idx in enumerate(cfg.LIST_FZ_LIMS[arch]):
        fz_limits_names = cfg.encode_freeze_layers(group_layers_idx, arch)
        type_ft = cfg.encode_type_ft(**exp_props)
        comment = exp_props.get('comment', '')
        df_ledger, rid = cfg.add_experiment(df_ledger, type_ft, fz_limits_names,
                                            arch, datasets, comment=comment, rid=now_rid)
        now_rid = None
        print(
            f"\nrid:{rid} fz:{fz_limits_idx} ({fz_limits_names}) for {arch}")
        r_id_status = {}
        for dataset, subdataset, pkl_path in utl.iterate_datasets_and_subdatasets(datasets):
            try:
                print(f"\       - {dataset}/{subdataset}")
                if finetune_target(dataset, subdataset, rid, type_ft, fz_limits_idx, arch):
                    r_id_status[(dataset, subdataset)] = cfg.STATUS_DONE

                else:
                    r_id_status[(dataset, subdataset)] = cfg.STATUS_ABORTED
                    df_ledger = cfg.update_ledger(df_ledger, rid, 'status', cfg.STATUS_ABORTED)

            except Exception as e:
                logging.error(f"Error occurred during fine-tuning {dataset}/{subdataset}: {str(e)}")
                df_ledger = cfg.remove_experiment(df_ledger, rid)
        if all([status == cfg.STATUS_DONE for status in r_id_status.values()]):
            df_ledger, result = cfg.update_ledger(df_ledger, rid, 'status', cfg.STATUS_DONE)
        else:
            errors = [key for key, value in r_id_status.items() if value != cfg.STATUS_DONE]
            status_msg = ', '.join([f'{key}/{value}' for key, value in errors])
            df_ledger, result = cfg.update_ledger(df_ledger, rid, 'status', cfg.STATUS_DONE + status_msg)

    cfg.save_ledger(df_ledger)
    return True


def get_processors(ext_archs=utl.EXT_ARCHS):
    """
    Get the processors for the specified external models.

    Parameters:
    - ext_archs (list, optional): A list of external models. Defaults to utl.EXT_ARCHS.

    Returns:
    - processors (list): A list of processors for the specified external models.

    """
    processors = []
    for arch in ext_archs:
        try:
            if arch == utl.MADMOM_RNN:
                processors.append(RNNOnsetProcessor())
            elif arch == utl.MADMOM_CNN:
                processors.append(CNNOnsetProcessor())
            else:
                raise ValueError(f'Unknown arch: {arch}')
        except Exception as e:
            raise Exception("Error creating processor: " + str(e))

    return processors


def create_external_baseline_predictions_for_datasets(datasets, ext_archs=utl.EXT_ARCHS):
    """
    Load predictions for external models.

    Parameters:
    - filename (str): The name of the file to load predictions from.
    - paths (dict): A dictionary containing the paths to the predictions directory.
    - ext_archs (list, optional): A list of external models. Defaults to utl.EXT_ARCHS.

    This function loads predictions for external models from the specified file. It iterates over each external model and loads the corresponding prediction file using the filename and paths provided. The loaded predictions are then appended to a list and returned.

    """
    processors = get_processors(ext_archs)
    for dataset, subdataset, pkl_path in utl.iterate_datasets_and_subdatasets(datasets):
        logging.info('iterating dataset %s, subdataset %s', dataset, subdataset)
        with open(pkl_path, 'rb') as f:
            db = pickle.load(f)
        paths = utl.get_path(dataset, subdataset)
        logging.info('making predictions for %s/%s', dataset, subdataset)
        logging.debug('audio_files %d', len(db.audio_files))
        for idx, filepath in enumerate(db.audio_files):
            y, sr = madmom.io.audio.load_audio_file(filepath, dtype=float)
            filename = Path(filepath).stem
            logging.info('(create_predictions) %s %d/%d', filename, idx, len(db))
            for arch, processor in zip(ext_archs, processors):
                pred = processor(y)
                np.save(f'{paths["predictions"]}/{arch}.{filename}.pred.npy', np.asarray(pred))


def create_baseline_predictions_for_datasets(datasets, arch):
    """
    Create baseline predictions for datasets.

    This function iterates over the given datasets and subdatasets, loads the data from pickle files, and generates baseline predictions using the specified model. The baseline predictions are then saved as numpy arrays.

    Parameters:
    - datasets (list): A list of datasets to iterate over.
    - arch (str): The name of the arch to use for generating predictions.

    """
    for dataset, subdataset, pkl_path in utl.iterate_datasets_and_subdatasets(datasets):
        logging.info('creating baseline predictions for dataset %s, subdataset %s', dataset, subdataset)
        with open(pkl_path, 'rb') as f:
            db = pickle.load(f)
        paths = utl.get_path(dataset, subdataset)
        old_model = load_model(utl.LOAD_MODEL_PATH / f'onset_{arch}.h5', compile=False)
        for idx, filename in enumerate(db.files):
            x = np.array(dfn.cnn_pad(db.x[idx], 2))[np.newaxis, ..., np.newaxis]
            if arch == utl.TCNV1:
                old_pred = np.squeeze(np.asarray(old_model.predict_step(x)))
            elif arch == utl.TCNV2:
                old_pred, _, _ = old_model.predict_step(x)
                old_pred = np.squeeze(np.asarray(old_pred))
            # Save Exports
            np.save(f'{paths["predictions"]}/{arch}.{filename}.old_pred.npy', np.asarray(old_pred))


def create_predictions_for_datasets(datasets, arch):
    """
    Generates and saves predictions for given datasets using a specified architecture model.

    Iterates over each dataset and its subdatasets, loads the corresponding model, and makes predictions 
    for each file in the dataset. The predictions are then saved to a .npy file.

    NOTE: This function can only be used for datasets already finetuned to. In this experiment, only 'maracatu'.

    Parameters:
    - datasets (list): List of datasets to create predictions for.
    - arch (str): The architecture model to use for prediction. Supported values are defined in `cfg.LIST_R_IDS`.

    Returns:
    - bool: True if the function executes successfully, otherwise an exception is raised.
    """
    list_rids = cfg.LIST_R_IDS[arch]
    df_ledger = cfg.load_ledger()
    for dataset, subdataset, pkl_path in utl.iterate_datasets_and_subdatasets(datasets):
        logging.info('creating predictions for dataset %s, subdataset %s', dataset, subdataset)
        with open(pkl_path, 'rb') as f:
            db = pickle.load(f)
        paths = utl.get_path(dataset, subdataset)

        for rid in list_rids:
            logging.info('  rid: %s', rid)
            prefix = cfg.get_prefix_from_rid(df_ledger, rid)
            prefix_ = f'{prefix}_' if prefix else ''

            if subdataset:
                model_path = Path(paths["final"]) / f'{prefix_}{subdataset}_finetuned.h5'
            else:
                model_path = Path(paths["final"]) / f'{prefix_}{dataset}_finetuned.h5'
            fin_model = load_model(model_path, compile=False)

            for idx, filename in enumerate(db.files):
                x = np.array(dfn.cnn_pad(db.x[idx], 2))[np.newaxis, ..., np.newaxis]
                if arch == utl.TCNV1:
                    fin_pred = np.squeeze(np.asarray(fin_model.predict_step(x)))
                elif arch == utl.TCNV2:
                    fin_pred, _, _ = fin_model.predict_step(x)
                    fin_pred = np.squeeze(np.asarray(fin_pred))

                # Save predictions
                output_filename = Path(paths["predictions"]) / f'{prefix_}{filename}.fin_pred.npy'
                np.save(output_filename, np.asarray(fin_pred))

            del fin_model
    return True


if __name__ == '__main__':
    ismir24_props = {
        'type': cfg.TREG_FD,
        'validation': cfg.TREG_V0,
        'base': cfg.TREG_B00,
        'duration': '05.00',  # 5 seconds
        'comment': 'new',
    }

    # Redo the fine-tuning experiments for the maracatu dataset
    for arch in [utl.TCNV1, utl.TCNV2]:
        run_finetuning_experiments(['maracatu'], ismir24_props, arch)

    # Redo the predictions for the maracatu dataset (and onset_db for obtaining baseline results published in the paper)
    test_datasets = ['maracatu', 'onset_db']

    # Create baseline predictions for external models
    create_external_baseline_predictions_for_datasets(test_datasets)

    # Create baseline predictions for our base models
    base_archs = [utl.TCNV1, utl.TCNV2]
    for arch in base_archs:
        create_baseline_predictions_for_datasets(test_datasets, arch)

    # Create finetuned predictions for our finetuned archs
    test_datasets = ['maracatu']
    base_archs = [utl.TCNV1, utl.TCNV2]
    for arch in base_archs:
        create_predictions_for_datasets(test_datasets, arch)
