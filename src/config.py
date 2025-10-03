import os
from pathlib import Path

# Set the path to the single folder containing all your data.
# This is the only path you need to modify. For Kaggle, it's as follows:
BRATS_TRAIN_FOLDERS = "/kaggle/input/data-npy"

# The variables below are no longer used by the new data splitting logic.
BRATS_VAL_FOLDER = ""
BRATS_TEST_FOLDER = ""


def get_brats_folder(on="train"):
    """
    Returns the path to the data folder.
    With the new logic, all splits (train, val, test) are derived
    from the single BRATS_TRAIN_FOLDERS path.
    This path can be overridden by setting the 'BRATS_FOLDERS' environment variable.
    """
    return os.environ.get('BRATS_FOLDERS', BRATS_TRAIN_FOLDERS)