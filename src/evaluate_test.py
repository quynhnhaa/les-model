import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader

from src.dataset import get_datasets
from src.loss.dice import EDiceLoss


def load_label_map(patient_id: str, preds_dir: Path) -> np.ndarray:
    """Load a NIfTI segmentation for the specified patient."""
    if preds_dir is None:
        raise ValueError("Prediction directory must be provided for this strategy.")

    candidates = [
        preds_dir / f"{patient_id}.nii.gz",
        preds_dir / f"{patient_id}_seg.nii.gz",
        preds_dir / f"{patient_id}.nii",
    ]

    for path in candidates:
        if path.exists():
            image = sitk.ReadImage(str(path))
            return sitk.GetArrayFromImage(image)

    raise FileNotFoundError(f"Could not find prediction for patient {patient_id} in {preds_dir}")


def labelmap_to_channels(labelmap: np.ndarray) -> np.ndarray:
    """Convert a BraTS-style labelmap (0/1/2/4) into ET/TC/WT binary channels."""
    et = labelmap == 4
    tc = np.logical_or(labelmap == 4, labelmap == 1)
    wt = np.logical_or(tc, labelmap == 2)
    return np.stack([et, tc, wt]).astype(np.float32)


def channels_to_labelmap(channels: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert probabilistic channels back into a BraTS labelmap."""
    assert channels.shape[0] == 3, "Expected channels in order [ET, TC, WT]"
    preds = (channels > threshold).astype(np.uint8)
    et, tc, wt = preds

    labelmap = np.zeros_like(et, dtype=np.uint8)
    labelmap[wt == 1] = 2
    labelmap[tc == 1] = 1
    labelmap[et == 1] = 4
    return labelmap


def combine_region_based(pred_a: np.ndarray, pred_b: np.ndarray) -> np.ndarray:
    """Region-based combination: Pipeline A provides WT, Pipeline B provides TC/ET."""
    if pred_b is None:
        return pred_a

    combined = pred_a.copy()

    # Replace Tumour Core (labels 1 & 4) from Pipeline B
    tc_mask_b = np.isin(pred_b, [1, 4])
    combined[tc_mask_b] = pred_b[tc_mask_b]

    # Ensure ET voxels remain labelled as 4
    et_mask_b = pred_b == 4
    combined[et_mask_b] = 4

    return combined


def combine_average(pred_a: np.ndarray, pred_b: Optional[np.ndarray]) -> np.ndarray:
    """Average ET/TC/WT channels from both pipelines."""
    if pred_b is None:
        return pred_a

    channels_a = labelmap_to_channels(pred_a)
    channels_b = labelmap_to_channels(pred_b)
    averaged = (channels_a + channels_b) / 2.0
    return channels_to_labelmap(averaged, threshold=0.5)


def combine_predictions(pred_a: Optional[np.ndarray],
                        pred_b: Optional[np.ndarray],
                        strategy: str) -> np.ndarray:
    """Combine predictions from Pipeline A & B according to the selected strategy."""
    if strategy == "pipeline_a":
        if pred_a is None:
            raise ValueError("Pipeline A predictions are required for 'pipeline_a' strategy")
        return pred_a

    if strategy == "pipeline_b":
        if pred_b is None:
            raise ValueError("Pipeline B predictions are required for 'pipeline_b' strategy")
        return pred_b

    if strategy == "average":
        if pred_a is None:
            raise ValueError("Pipeline A predictions are required for averaging")
        return combine_average(pred_a, pred_b)

    if strategy == "region":
        if pred_a is None:
            raise ValueError("Pipeline A predictions are required for region-based combination")
        return combine_region_based(pred_a, pred_b)

    raise ValueError(f"Unknown combination strategy '{strategy}'")


def compute_dice_on_test(args):
    dataset = get_datasets(args.seed, args.debug, no_seg=False, on="test", normalisation=args.normalisation)
    loader = DataLoader(dataset, batch_size=1, num_workers=args.workers)

    criterion = EDiceLoss(do_sigmoid=False)

    dices = []
    per_patient_results = {}

    preds_dir_a = Path(args.pipeline_a_dir).resolve() if args.pipeline_a_dir else None
    preds_dir_b = Path(args.pipeline_b_dir).resolve() if args.pipeline_b_dir else None

    if preds_dir_a and not preds_dir_a.exists():
        raise FileNotFoundError(f"Pipeline A directory not found: {preds_dir_a}")
    if preds_dir_b and not preds_dir_b.exists():
        raise FileNotFoundError(f"Pipeline B directory not found: {preds_dir_b}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for batch in loader:
        patient_id = batch["patient_id"][0]
        target = batch["label"].float()  # shape: [1, 3, D, H, W]
        crop_indexes = batch["crop_indexes"]

        pred_a = load_label_map(patient_id, preds_dir_a) if preds_dir_a else None
        pred_b = load_label_map(patient_id, preds_dir_b) if preds_dir_b else None

        # Crop the loaded predictions to match the target label's crop
        if pred_a is not None:
            z_slice = slice(crop_indexes[0][0].item(), crop_indexes[0][1].item())
            y_slice = slice(crop_indexes[1][0].item(), crop_indexes[1][1].item())
            x_slice = slice(crop_indexes[2][0].item(), crop_indexes[2][1].item())
            pred_a = pred_a[z_slice, y_slice, x_slice]
        
        if pred_b is not None:
            z_slice = slice(crop_indexes[0][0].item(), crop_indexes[0][1].item())
            y_slice = slice(crop_indexes[1][0].item(), crop_indexes[1][1].item())
            x_slice = slice(crop_indexes[2][0].item(), crop_indexes[2][1].item())
            pred_b = pred_b[z_slice, y_slice, x_slice]

        combined_labelmap = combine_predictions(pred_a, pred_b, args.strategy)

        # Save combined segmentation for inspection
        combined_image = sitk.GetImageFromArray(combined_labelmap)
        sitk.WriteImage(combined_image, str(output_dir / f"{patient_id}_combined.nii.gz"))

        pred_channels = labelmap_to_channels(combined_labelmap)
        pred_tensor = torch.from_numpy(pred_channels).unsqueeze(0).float()

        with torch.no_grad():
            dice_scores = criterion.metric(pred_tensor, target)
            dice_scores = dice_scores.squeeze(0).cpu().numpy()  # [ET, TC, WT]

        dices.append(dice_scores)
        per_patient_results[patient_id] = dice_scores

    dices_np = np.array(dices) if len(dices) > 0 else np.zeros((0, 3))
    mean_dice = dices_np.mean(axis=0) if len(dices_np) > 0 else np.array([0.0, 0.0, 0.0])

    return mean_dice, dices_np, per_patient_results


def main():
    parser = argparse.ArgumentParser(description="Compute Dice on held-out test split (with labels)")
    parser.add_argument('--seed', type=int, default=16111990, help='fold seed used to build the fixed test split')
    parser.add_argument('--devices', required=False, type=str, help='CUDA devices (unused for pure eval)')
    parser.add_argument('--batch_size', type=int, default=1, help='(deprecated) use --workers instead of batch size')
    parser.add_argument('--normalisation', type=str, default='minmax', choices=['minmax', 'zscore'])
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--pipeline_a_dir', type=str, required=False, help='Directory with Pipeline A predictions (.nii.gz)')
    parser.add_argument('--pipeline_b_dir', type=str, required=False, help='Directory with Pipeline B predictions (.nii.gz)')
    parser.add_argument('--strategy', type=str, default='region', choices=['pipeline_a', 'pipeline_b', 'average', 'region'],
                        help='Combination strategy for Pipeline A & B results')
    parser.add_argument('--output_dir', type=str, default='./preds/test_combined',
                        help='Directory to save combined predictions and dice scores')
    args = parser.parse_args()

    if args.devices:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

    mean_dice, dices_np, per_patient = compute_dice_on_test(args)
    print(f"Mean Dice [ET, TC, WT]: {mean_dice}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / f"test_dice_mean_seed{args.seed}.npy", mean_dice)
    np.save(output_dir / f"test_dice_seed{args.seed}.npy", dices_np)

    # Also write a CSV summary for convenience
    import csv
    csv_path = output_dir / f"test_dice_per_patient_seed{args.seed}.csv"
    with csv_path.open('w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["patient_id", "dice_et", "dice_tc", "dice_wt"])
        for patient_id, scores in per_patient.items():
            writer.writerow([patient_id, *scores])


if __name__ == '__main__':
    main()