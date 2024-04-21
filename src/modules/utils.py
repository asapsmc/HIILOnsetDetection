from pathlib import Path

TCNV1 = 'TCNv1'
TCNV2 = 'TCNv2'
SUBDATASETS_MAPPING = {
    'maracatu': ['CAIXA', 'CUICA', 'GONGE_LO', 'TAMBOR_HI', 'MINEIRO'],
    'onset_db': [None],
}
MADMOM_RNN = 'madmomRNN'
MADMOM_CNN = 'madmomCNN'
EXT_ARCHS = [MADMOM_RNN, MADMOM_CNN]

FPS = 100

current_path = Path.cwd()
base_path = current_path

# Define all other paths relative to the base path
DATA_PATH = base_path / 'data'
PKL_PATH = base_path / 'pkl'
MDL_PATH = base_path / 'models'
EXP_PATH = base_path / 'exports'
CFG_PATH = base_path / 'configs'
LOAD_MODEL_PATH = MDL_PATH  # Assuming LOAD_MODEL_PATH should point to the same directory as MDL_PATH

# For reporting
report_path = base_path / 'report'
FIG_PATH = report_path / 'figures'
TBL_PATH = report_path / 'tables'
XLS_PATH = report_path / 'xls'
INPUTS_PATH = report_path / 'inputs'

LEDGER_PATH = EXP_PATH / 'ledger.csv'


def get_path(dataset, subdataset=None, suffix='', location=Path.cwd()):
    """
    Generates a comprehensive dictionary of paths related to a specific dataset and subdataset
    within the project structure, addressing the needs for data storage (annotations and audio),
    as well as exports and outputs directories for various analysis stages.

    Parameters:
    - dataset (str): The name of the dataset for which to generate paths.
    - subdataset (str, optional): The name of the subdataset, corresponding to a specific instrument or category within the dataset.
    - suffix (str, optional): An additional suffix to append to the dataset name in the path. Defaults to an empty string.
    - location (Path, optional): The starting point location to calculate paths from. Defaults to the current working directory.

    Returns:
    - dict: A dictionary with keys representing different types of paths (e.g., 'annotations', 'audio', 'baseline', 'analysis') and their corresponding `Path` objects.
    """
    base_path = location

    if subdataset is None:
        data_path = base_path / 'data' / dataset
        exports_base_path = base_path / 'exports' / dataset
        outputs_base_path = base_path / 'outputs' / dataset

    else:
        data_path = base_path / 'data' / dataset / subdataset
        exports_base_path = base_path / 'exports' / dataset / subdataset
        outputs_base_path = base_path / 'outputs' / dataset / subdataset

    annotations_path = data_path / 'annotations'
    audio_path = data_path / 'audio'
    # Define base paths for exports and outputs according to dataset/subdataset structure

    # Specific paths for various types of analysis and exports
    paths = {
        'annotations': annotations_path,
        'audio': audio_path,
        'baseline': outputs_base_path / 'baseline',
        'analysis': outputs_base_path / 'analysis',
        'detections': outputs_base_path / 'detections',
        'results': outputs_base_path / 'results',
        'predictions': exports_base_path / 'predictions',
        'final': exports_base_path / 'final',
        'stats': exports_base_path / 'stats',
    }
    return paths


def set_paths_dataset(dataset, subdataset='', suffix=''):
    """
    Creates directories for a given dataset and subdataset name, including for annotations and audio,
    as well as for various stages of analysis and exports. Ensures all directories exist, creating them if necessary.

    Parameters:
    - dataset (str): The name of the dataset.
    - subdataset (str, optional): The name of the subdataset, corresponding to a specific instrument or category.
    - suffix (str, optional): An additional suffix to append to the dataset name in the path. Defaults to an empty string.

    Returns:
    - Path: The parent path of the 'baseline' directory, effectively the base directory for the dataset's and subdataset's outputs.
    """
    paths = get_path(dataset, subdataset, suffix)
    for path in paths.values():
        Path(path).mkdir(parents=True, exist_ok=True)
        print(f'created {path}')
    return paths["baseline"].parent


def iterate_datasets_and_subdatasets(datasets=None):
    """
    Yields dataset and subdataset information, abstracting the logic for handling both cases.

    Args:
        datasets (iterable, optional): A collection of dataset names to process.
            If None, processes all datasets defined in utl.SUBDATASETS_MAPPING.

    Yields:
        tuple: A tuple containing:
            - dataset (str): The dataset name.
            - subdataset (str or None): The subdataset name, or None if not applicable.
            - pkl_path (Path): Path to the .pkl file for the dataset/subdataset.
    """
    if datasets is None:
        datasets = SUBDATASETS_MAPPING.keys()

    for dataset in datasets:
        subdatasets = SUBDATASETS_MAPPING.get(dataset, [None])
        for subdataset in subdatasets:
            pkl_path = f'{PKL_PATH}/{dataset}.pkl' if subdataset is None else f'{PKL_PATH}/{dataset}/{subdataset}.pkl'

            yield dataset, subdataset, pkl_path
