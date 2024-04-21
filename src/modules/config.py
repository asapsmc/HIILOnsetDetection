import json
import logging

import pandas as pd

import modules.utils as utl

LIST_R_IDS = {'TCNv1': ['000', '001', '002', '003', '004', '005', '006',
              '007', '008', '009', '010', '011', '012', '013', '014'],
              'TCNv2': ['020', '021', '022', '023', '024', '025', '026',
                        '027', '028', '029', '030', '031', '032', '033', '034'],
              }

FZ_LAYERS_DEF = 'None-Activations'
LIST_LAYERS = ['Conv1', 'Conv2', 'Conv3', 'Tcn1', 'Tcn2', 'Tcn4', 'Tcn8', 'Tcn16',
               'Tcn32', 'Tcn64', 'Tcn128', 'Tcn256', 'Tcn512', 'Tcn1024', 'None']

LIST_FZ_LIMS = {"TCNv1": [(1, -3),      # Conv1-Activations(68)
                          (5, -3),      # Conv2-Activations(68)
                          (9, -3),      # Conv3-Activations(68)
                          (13, -3),     # Tcn1-Activations(68)
                          (18, -3),     # Tcn2-Activations(68)
                          (23, -3),     # Tcn4-Activations(68)
                          (28, -3),     # Tcn8-Activations(68)
                          (33, -3),     # Tcn16-Activations(68)
                          (38, -3),     # Tcn32-Activations(68)
                          (43, -3),     # Tcn64-Activations(68)
                          (48, -3),     # Tcn128-Activations(68)
                          (53, -3),     # Tcn256-Activations(68)
                          (58, -3),     # Tcn512-Activations(68)
                          (63, -3),     # Tcn1024-Activations(68)
                          (None, -3)],  # None-Activations(68)

                "TCNv2": [(1, -22),     # Conv1-Activations(105)
                          (5, -22),     # Conv2-Activations(105)
                          (9, -22),     # Conv3-Activations(105)
                          (14, -22),    # Tcn1-Activations(105)
                          (22, -22),    # Tcn2-Activations(105)
                          (30, -22),    # Tcn4-Activations(105)
                          (38, -22),    # Tcn8-Activations(105)
                          (46, -22),    # Tcn16-Activations(105)
                          (54, -22),    # Tcn32-Activations(105)
                          (62, -22),    # Tcn64-Activations(105)
                          (70, -22),    # Tcn128-Activations(105)
                          (78, -22),    # Tcn256-Activations(105)
                          (86, -22),    # Tcn512-Activations(105)
                          (94, -22),    # Tcn1024-Activations(105)
                          (None, -22)]}  # None-Activations(105)

TABLE_ARCH = pd.DataFrame({
    'layer_low': LIST_LAYERS,
    'layer_high': ['Activations'] * len(LIST_LAYERS),
    utl.TCNV1: LIST_FZ_LIMS[utl.TCNV1],
    utl.TCNV2: LIST_FZ_LIMS[utl.TCNV2]
})

# TREG Type Region
TREG_FD = 'FD'  # Fixed Duration
TREG_FR = 'FR'  # Fixed Ratio
TREG_FF = 'F0'  # Fixed File
TREG_CP = 'CP'  # Custom Percentage(for comparative_finetune)
TREG_CS = 'CS'  # Custom Sequential(for comparative_finetune)

# TREG.VAL Type Region Validation
TREG_V1 = 'V1'  # with validation(50%)
TREG_V0 = 'V0'  # without validation

# TREG.BASE Type Region Base
TREG_BA0 = 'BA0'  # Base anns0_0
TREG_B00 = 'B00'  # Base 0


TYPE_FT_SEP = '.'
GLOBAL_SEP = '_'
FZ_SEP = '-'

TREG_DEF = f'{TREG_FD}.00.10.{TREG_V1}.{TREG_BA0}'

# LEDGER
STATUS_PENDING = 'pending'
STATUS_DONE = 'done'
STATUS_ABORTED = 'aborted'


def get_prefix_from_rid(df_ledger, rid):
    """
    Retrieve and concatenate specific fields from the ledger to form a prefix for the given RID.

    Args:
    - df_ledger (DataFrame): The ledger dataframe containing experiment details.
    - rid (str): The run ID for which to generate the prefix.

    Returns:
    - str: A concatenated prefix based on the type_ft, arch, and optional fz_layers fields.
    """
    fields = ['type_ft', 'arch', 'fz_layers']
    field_data = get_fields_from_rid(df_ledger, rid, fields)
    return encode_prefix(rid, field_data['type_ft'], field_data['arch'], field_data['fz_layers'])


def encode_prefix(rid, type_ft, arch, fz_layers=None, sep=GLOBAL_SEP):
    """
    Map components to the prefix format.
    """
    components = filter(None, [rid, fz_layers, type_ft, arch])
    return sep.join(components)


def decode_prefix(prefix, separator=GLOBAL_SEP):
    """
    Direct mapping from prefix to components.
    """
    rid, fz_layers, type_ft, arch = prefix.split(separator, 4)  # Use 2 to avoid errors with additional separators
    return rid, fz_layers, type_ft, arch


###

def update_config(config, **kwargs):
    """
    Update the configuration dictionary with the provided key-value pairs.

    Parameters:
        config (dict): The configuration dictionary to be updated.
        **kwargs: Key-value pairs to update the configuration dictionary.

    Returns:
        dict: The updated configuration dictionary.

    Raises:
        KeyError: If any of the provided keys are not present in the configuration dictionary.
    """
    config.update(kwargs)
    for key, value in kwargs.items():
        if key not in config:
            raise KeyError(f"Invalid key: {key}")
    return config


# DECODE
def decode_type_ft(type_ft, separator=TYPE_FT_SEP):
    """
    Decode the type_ft string into a dictionary with its components based on the specified format.
    The string format should be 'type.duration_part1.duration_part2.validation.base'.

    Args:
        type_ft (str): The type_ft string to decode, expected format 'type.duration_part1.duration_part2.validation.base'.
        separator (str, optional): The separator used to split the type_ft string. Defaults to '_'.

    Returns:
        dict: A dictionary containing the decoded components.

    Raises:
        ValueError: If the type_ft format is incorrect or if the duration parts are not in the expected format 'xx.00'.

    """
    def convert_duration_str_to_int(number_str):
        """
        Converts a formatted string "xx.00" to an integer, ensuring it follows the specified decimal format.

        Args:
            number_str (str): The string to convert, expected to be in the format "xx.00".

        Returns:
            int: The integer part of the number_str, ignoring the decimal part.

        Raises:
            ValueError: If the string is not in the format 'xx.00'.
        """
        if '.' in number_str and number_str.endswith('.00'):
            integer_part = number_str.split('.')[0]
            return int(integer_part)
        else:
            raise ValueError("Input string must be in the format 'xx.00'")

    parts = type_ft.split(separator)
    if len(parts) != 5:
        raise ValueError(f"Invalid type_ft format; expected 5 components separated by '{separator}'")

    duration_str = f"{parts[1]}.{parts[2]}"  # Correctly format the duration part
    duration = convert_duration_str_to_int(duration_str)

    return {
        'type': parts[0],
        'duration': duration,
        'validation': parts[3] == TREG_V1,
        'base': parts[4]
    }


def encode_type_ft(separator=TYPE_FT_SEP, **kwargs):
    """
    Encode components into a type_ft string using the specified separator. Expects keyword arguments
    that include 'type', 'duration', 'validation', and 'base'.

    Args:
        separator (str, optional): The separator to use between components. Defaults to '_'.
        **kwargs (dict): Keyword arguments that must include:
                         - 'type' (str): The type of the regimen (FD, FR, CP, CS).
                         - 'duration' (str): The duration, possibly including units (e.g., '00.05' for seconds).
                         - 'validation' (bool): Whether validation is applied (e.g. 'V1' or 'V0').
                         - 'base' (str): The base specification (e.g., 'B00', 'BA0').

    Returns:
        str: A single string packed with the provided components separated by the specified separator.
    """
    type_str = kwargs.get('type', '')
    duration_str = kwargs.get('duration', '')
    validation_str = kwargs.get('validation', '')
    base_str = kwargs.get('base', '')

    components = [type_str, duration_str, validation_str, base_str]
    return separator.join(components)


# def get_freeze_layers_idx(fz_layers, arch):
#    """
#    Get the low and high layer indices for freezing layers.
#
#    Parameters:
#    - fz_layers (str): A string representing the freezing layers. It should be in the format "low-high" or "-high".
#    - arch (str): The architecture name.
#
#    Returns:
#    - low (int or None): The index of the low layer to freeze. If fz_layers is in the format "-high", it will be None.
#    - high (int): The index of the high layer to freeze.
#    """
#    if '-' not in fz_layers:
#        raise ValueError("Invalid freezing layers format. Expected format: 'low-high' or '-high'")
#
#    parts = fz_layers.split(FZ_SEP)
#
#    try:
#        layer_low, layer_high = TABLE_ARCH.loc[[parts[0], parts[1]], arch].values.flatten()
#    except KeyError as e:
#        raise ValueError(f"Invalid layer name: {e.args[0]}")
#
#    return layer_low, layer_high


def encode_freeze_layers(fz_limits_idx, arch, separator=FZ_SEP):
    """
    Encode the freeze layers parameters into a string format.

    Args:
        fz_limits_idx (index): The index to the group layers freeze.
        arch (str): The architecture name.
        separator (str, optional): The separator used to format the freeze layers string. Defaults to '-'.

    Returns:
        str: A string that represents the freeze layers, formatted as 'fz_low-fz_high'.
    """
    return f"{TABLE_ARCH.at[fz_limits_idx, 'layer_low']}{separator}{TABLE_ARCH.at[fz_limits_idx, 'layer_high']}"


def decode_freeze_layers(encoded_str, separator=FZ_SEP):
    """
    Decode the freeze layers string into its components.

    Args:
        encoded_str (str): The encoded freeze layers string, formatted as 'fz_low-fz_high'.

    Returns:
        tuple: A tuple containing `fz_low` and `fz_high`.
    """
    parts = encoded_str.split(separator)
    if len(parts) == 2:
        return tuple(parts)
    else:
        raise ValueError("Invalid encoded freeze layers format; expected 'fz_low-fz_high'.")


# Regions
def get_cutpoint_frames(base=0, retrain_dur=10):
    cutpoint = int(utl.FPS * (base + retrain_dur))
    return cutpoint


def debase_array(arr, real_base, retrain_dur):
    arr = arr - real_base
    arr = arr[arr > retrain_dur]
    arr = arr - retrain_dur
    return arr


# Save List of configurations
# LEDGER

def load_ledger(path=utl.LEDGER_PATH):
    """
    Load the ledger DataFrame from a CSV file, using 'rid' as the index while preserving it as a string to maintain
    any leading zeros.

    Args:
        file_path (str): The path to the CSV file containing the ledger data.

    Returns:
        DataFrame: The loaded DataFrame with 'rid' as a string index and 'datasets' column parsed as JSON.
    """
    df = pd.read_csv(path, index_col='rid', dtype={'rid': str})
    df['datasets'] = df['datasets'].apply(lambda x: json.loads(x))

    return df


def save_ledger(df_ledger, path=utl.LEDGER_PATH):
    """
    Save the DataFrame ledger to a CSV file, ensuring the 'rid' index is included as a column and
    all 'datasets' are correctly formatted as JSON strings.

    Args:
        df_ledger (DataFrame): The DataFrame containing the ledger to save.
        path (str): The path to the CSV file where the ledger will be saved.
    """
    if df_ledger.index.name == 'rid':
        df_ledger.reset_index(inplace=True)

    df_ledger['datasets'] = df_ledger['datasets'].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x)
    df_ledger.to_csv(path, index=True, index_label='rid')
    logging.info("Ledger saved to %s", path)


def get_next_rid(df_ledger):
    """
    Generate the next run ID based on the last ID in the ledger, ensuring proper zero padding.

    Args:
        df_ledger (DataFrame): The DataFrame containing the ledger data with an 'rid' column.

    Returns:
        str: The next 'rid' properly zero-padded to maintain a three-digit format.
    """
    if not df_ledger.empty:
        last_rid = df_ledger.index[-1]
        next_rid = f"{int(last_rid) + 1:03}"  # Increment, convert to int, and zero-pad
    else:
        next_rid = "000"

    return next_rid


def add_experiment(df_ledger, type_ft, fz_layers, arch, datasets, status=STATUS_PENDING, comment='', rid=None):
    """
    Add a new experiment record to the ledger. Automatically assigns the next available RID if not specified.

    Args:
        df_ledger (DataFrame): The ledger DataFrame to which the new record will be added.
        type_ft (str): Type of the feature (e.g., 'FD.00.05.V0.B00').
        fz_layers (str): Feature layer description (e.g., 'Conv1-Activations').
        arch (str): Architecture used (e.g., 'TCNv1').
        datasets (list): List of datasets involved, should be passed as a list (e.g., ["maracatu"]).
        comment (str, optional): Any additional comments.
        rid (str, optional): The run ID. If None, the next available RID will be automatically generated.

    Returns:
        str: The RID of the newly added record.
    """
    if rid is None:
        rid = get_next_rid(df_ledger)

    if rid in df_ledger.index:
        raise ValueError(f'ERROR: RID {rid} already exists. Choose a new RID.')

    datasets_json = json.dumps(datasets)
    new_record_df = pd.DataFrame({
        'type_ft': [type_ft],
        'fz_layers': [fz_layers],
        'arch': [arch],
        'datasets': [datasets_json],
        'status': ['pending'],
        'comment': [comment]
    }, index=[rid])

    new_record_df = new_record_df.reindex(columns=df_ledger.columns)
    df_ledger = pd.concat([df_ledger, new_record_df])
    save_ledger(df_ledger)
    logging.info('Saved updated ledger')

    return df_ledger, rid


def remove_experiment(df_ledger, rid):
    """
    Remove an experiment record from the ledger based on the provided RID. Raises an error if the RID does not exist.

    Args:
        df_ledger (DataFrame): The ledger dataframe containing experiment details.
        rid (str): The run ID of the experiment to be removed.

    Returns:
        DataFrame: The updated DataFrame after the removal of the specified record.
    """
    if rid in df_ledger['rid'].values:
        df_ledger = df_ledger[df_ledger['rid'] != rid]
        save_ledger(df_ledger)
        logging.info("Experiment record with RID %s has been removed.", rid)
        return df_ledger
    else:
        raise ValueError(f"ERROR: RID {rid} does not exist. No record removed.")


def update_ledger(df_ledger, rid, field, value):
    """
    Update a specific field for an experiment record identified by RID, defaulting the value to 'yes'.

    Args:
        df_ledger (DataFrame): The ledger DataFrame containing the records.
        rid (str): The run ID, which is used as the index in the DataFrame.
        field (str): The field to update (e.g., 'status').
        value (str, optional): The new value for the field (defaults to 'yes').

    Returns:
        bool: True if the update was successful, False otherwise.
    """
    if rid in df_ledger.index:
        df_ledger.at[rid, field] = value
        save_ledger(df_ledger)
        return df_ledger, True
    else:
        logging.error(f'ERROR: No record with RID {rid} found.')
        return df_ledger, False


def get_fields_from_rid(df_ledger, rid, fields):
    """
    Retrieve a field value or values for a specific RID directly from the DataFrame without needing to parse stringified values.
    Can handle both a single field as a string or multiple fields as a list of strings, returning a single value or a dictionary of values respectively.

    Args:
    df_ledger (DataFrame): The ledger dataframe containing experiment details.
    rid (str): The run ID whose data is to be retrieved.
    fields (str or list): The field(s) to retrieve data from.

    Returns:
    single value or a dict of field values depending on whether fields is a string or list.
    """
    idx = df_ledger[df_ledger['rid'] == rid].index.item()
    if isinstance(fields, list):
        return {field: df_ledger.at[idx, field] for field in fields}
    return df_ledger.at[idx, fields]
