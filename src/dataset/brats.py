import pathlib

import SimpleITK as sitk
import numpy as np
import torch
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data.dataset import Dataset

from src.config import get_brats_folder
from src.dataset.image_utils import pad_or_crop_image, irm_min_max_preprocess, zscore_normalise


class Brats(Dataset):
    def __init__(self, patients_dir, benchmarking=False, training=True, debug=False, data_aug=False,
                 no_seg=False, normalisation="minmax"):
        super(Brats, self).__init__()
        self.benchmarking = benchmarking
        self.normalisation = normalisation
        self.debug = debug
        self.data_aug = data_aug
        self.training = training
        self.datas = []
        self.validation = no_seg
        self.patterns = ["_t1", "_t1ce", "_t2", "_flair"]
        if not no_seg:
            self.patterns += ["_seg"]
        for patient_dir in patients_dir:
            patient_id = patient_dir.name
            paths = [patient_dir / f"{patient_id}{value}.nii.gz" for value in self.patterns]
            patient = dict(
                id=patient_id, t1=paths[0], t1ce=paths[1],
                t2=paths[2], flair=paths[3], seg=paths[4] if not no_seg else None
            )
            self.datas.append(patient)

    def __getitem__(self, idx):
        _patient = self.datas[idx]
        patient_image = {key: self.load_nii(_patient[key]) for key in _patient if key not in ["id", "seg"]}
        if _patient["seg"] is not None:
            patient_label = self.load_nii(_patient["seg"])
        if self.normalisation == "minmax":
            patient_image = {key: irm_min_max_preprocess(patient_image[key]) for key in patient_image}
        elif self.normalisation == "zscore":
            patient_image = {key: zscore_normalise(patient_image[key]) for key in patient_image}
        patient_image = np.stack([patient_image[key] for key in patient_image])
        if _patient["seg"] is not None:
            et = patient_label == 4
            et_present = 1 if np.sum(et) >= 1 else 0
            tc = np.logical_or(patient_label == 4, patient_label == 1)
            wt = np.logical_or(tc, patient_label == 2)
            patient_label = np.stack([et, tc, wt])
        else:
            patient_label = np.zeros(patient_image.shape)  # placeholders, not gonna use it
            et_present = 0
        if self.training:
            # Remove maximum extent of the zero-background to make future crop more useful
            z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(patient_image, axis=0) != 0)
            # Add 1 pixel in each side
            zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
            zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
            patient_image = patient_image[:, zmin:zmax, ymin:ymax, xmin:xmax]
            patient_label = patient_label[:, zmin:zmax, ymin:ymax, xmin:xmax]
            # default to 128, 128, 128
            patient_image, patient_label = pad_or_crop_image(patient_image, patient_label, target_size=(128, 128, 128))
        else:
            z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(patient_image, axis=0) != 0)
            # Add 1 pixel in each side
            zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
            zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
            patient_image = patient_image[:, zmin:zmax, ymin:ymax, xmin:xmax]
            patient_label = patient_label[:, zmin:zmax, ymin:ymax, xmin:xmax]
        patient_image, patient_label = patient_image.astype("float16"), patient_label.astype("bool")
        patient_image, patient_label = [torch.from_numpy(x) for x in [patient_image, patient_label]]
        return dict(patient_id=_patient["id"],
                    image=patient_image, label=patient_label,
                    seg_path=str(_patient["seg"]) if not self.validation else str(_patient["t1"]),
                    crop_indexes=((zmin, zmax), (ymin, ymax), (xmin, xmax)),
                    et_present=et_present,
                    supervised=True,
                    )

    @staticmethod
    def load_nii(path_folder):
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path_folder)))

    def __len__(self):
        return len(self.datas) if not self.debug else 3


def get_datasets(seed, debug, no_seg=False, on="train", full=False,
                 fold_number=0, normalisation="minmax"):
    # We're only going to use one folder with all the data, BRATS_TRAIN_FOLDERS
    # The "on" parameter will be used to select the dataset split
    base_folder = pathlib.Path(get_brats_folder("train")).resolve()
    assert base_folder.exists(), f"folder {base_folder} does not exist"
    patients_dir = sorted([x for x in base_folder.iterdir() if x.is_dir()])

    # First, split all patients into a training/validation group and a final test group
    train_val_patients, test_patients = train_test_split(patients_dir, test_size=0.2, random_state=seed)

    # If "test" or "val" is specified, we return the held-out test set for inference
    if on in ["test", "val"]:
        print(f"Returning held-out test set with {len(test_patients)} patients.")
        return Brats(test_patients, training=False, debug=debug,
                     no_seg=no_seg, normalisation=normalisation)

    # If "full" is specified, we use the entire train+val set for training (no k-fold)
    if full:
        train_dataset = Brats(train_val_patients, training=True, debug=debug,
                              normalisation=normalisation)
        bench_dataset = Brats(train_val_patients, training=False, benchmarking=True, debug=debug,
                              normalisation=normalisation)
        return train_dataset, bench_dataset

    # This is the default "train" case, where we perform k-fold on the train_val_patients
    if no_seg:
        # This case is unlikely for training, but kept for compatibility
        return Brats(train_val_patients, training=False, debug=debug,
                     no_seg=no_seg, normalisation=normalisation)

    kfold = KFold(5, shuffle=True, random_state=seed)
    # Apply KFold to the train_val_patients, not the full dataset
    splits = list(kfold.split(train_val_patients))
    train_idx, val_idx = splits[fold_number]

    print(f"Total patients in source folder: {len(patients_dir)}")
    print(f"Splitting into {len(train_val_patients)} for Train/Val and {len(test_patients)} for Test.")
    print("-" * 20)
    print(f"Using fold number {fold_number} of {kfold.get_n_splits()}.")
    print(f"Train set size: {len(train_idx)} patients")
    print(f"Validation set size: {len(val_idx)} patients")

    train = [train_val_patients[i] for i in train_idx]
    val = [train_val_patients[i] for i in val_idx]

    train_dataset = Brats(train, training=True,  debug=debug,
                          normalisation=normalisation)
    val_dataset = Brats(val, training=False, data_aug=False,  debug=debug,
                        normalisation=normalisation)
    bench_dataset = Brats(val, training=False, benchmarking=True, debug=debug,
                          normalisation=normalisation)
    return train_dataset, val_dataset, bench_dataset
