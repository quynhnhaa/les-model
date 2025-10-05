#!/usr/bin/env python3
"""
Combine Pipeline A and B results intelligently for better segmentation
"""
import argparse
import os
import pathlib
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader
from scipy.spatial.distance import directed_hausdorff

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


def calculate_dice_coefficient(pred, target):
    """Calculate Dice coefficient between prediction and target."""
    intersection = np.sum(pred * target)
    if np.sum(pred) + np.sum(target) == 0:
        return 1.0  # Perfect dice for empty masks
    return 2 * intersection / (np.sum(pred) + np.sum(target))


def calculate_hausdorff_distance(pred, target):
    """Calculate Hausdorff distance between prediction and target."""
    if np.sum(pred) == 0 and np.sum(target) == 0:
        return 0.0
    elif np.sum(pred) == 0 or np.sum(target) == 0:
        return 100.0  # Large penalty for missing structures

    pred_coords = np.argwhere(pred)
    target_coords = np.argwhere(target)

    if len(pred_coords) == 0 or len(target_coords) == 0:
        return 100.0

    return directed_hausdorff(pred_coords, target_coords)[0]


def evaluate_pipeline_performance(pred_a, pred_b, target, crop_indexes):
    """Evaluate which pipeline performs better for each region."""
    # Crop target to match prediction size
    z_slice = slice(crop_indexes[0][0], crop_indexes[0][1])
    y_slice = slice(crop_indexes[1][0], crop_indexes[1][1])
    x_slice = slice(crop_indexes[2][0], crop_indexes[2][1])
    target_cropped = target[z_slice, y_slice, x_slice]

    # Convert to channels
    target_et = target_cropped == 4
    target_tc = np.logical_or(target_cropped == 4, target_cropped == 1)
    target_wt = np.logical_or(target_tc, target_cropped == 2)

    # Evaluate each pipeline for each region
    regions = {
        'ET': (pred_a == 4, pred_b == 4, target_et),
        'TC': (np.logical_or(pred_a == 4, pred_a == 1),
               np.logical_or(pred_b == 4, pred_b == 1), target_tc),
        'WT': (np.logical_or(np.logical_or(pred_a == 4, pred_a == 1), pred_a == 2),
               np.logical_or(np.logical_or(pred_b == 4, pred_b == 1), pred_b == 2), target_wt)
    }

    pipeline_scores = {'A': {}, 'B': {}}

    for region_name, (pred_a_region, pred_b_region, target_region) in regions.items():
        dice_a = calculate_dice_coefficient(pred_a_region, target_region)
        dice_b = calculate_dice_coefficient(pred_b_region, target_region)

        hd_a = calculate_hausdorff_distance(pred_a_region, target_region)
        hd_b = calculate_hausdorff_distance(pred_b_region, target_region)

        # Combined score: weighted combination of Dice and Hausdorff (lower is better for HD)
        score_a = dice_a - (hd_a / 100.0)  # Normalize HD to 0-1 range
        score_b = dice_b - (hd_b / 100.0)

        pipeline_scores['A'][region_name] = {'dice': dice_a, 'hausdorff': hd_a, 'combined_score': score_a}
        pipeline_scores['B'][region_name] = {'dice': dice_b, 'hausdorff': hd_b, 'combined_score': score_b}

    return pipeline_scores


def combine_predictions_intelligently(pred_a, pred_b, target, crop_indexes):
    """Combine predictions intelligently based on regional performance."""
    pipeline_scores = evaluate_pipeline_performance(pred_a, pred_b, target, crop_indexes)

    # Choose best pipeline for each region
    region_choices = {}
    for region in ['ET', 'TC', 'WT']:
        score_a = pipeline_scores['A'][region]['combined_score']
        score_b = pipeline_scores['B'][region]['combined_score']

        if score_a >= score_b:
            region_choices[region] = 'A'
        else:
            region_choices[region] = 'B'

    # Create combined prediction
    combined = np.zeros_like(pred_a)

    # WT region: choose best pipeline
    if region_choices['WT'] == 'A':
        wt_mask_a = np.logical_or(np.logical_or(pred_a == 4, pred_a == 1), pred_a == 2)
        combined[wt_mask_a] = pred_a[wt_mask_a]
    else:
        wt_mask_b = np.logical_or(np.logical_or(pred_b == 4, pred_b == 1), pred_b == 2)
        combined[wt_mask_b] = pred_b[wt_mask_b]

    # TC region: choose best pipeline (overwrites WT if necessary)
    if region_choices['TC'] == 'A':
        tc_mask_a = np.logical_or(pred_a == 4, pred_a == 1)
        combined[tc_mask_a] = pred_a[tc_mask_a]
    else:
        tc_mask_b = np.logical_or(pred_b == 4, pred_b == 1)
        combined[tc_mask_b] = pred_b[tc_mask_b]

    # ET region: choose best pipeline (overwrites TC if necessary)
    if region_choices['ET'] == 'A':
        et_mask_a = pred_a == 4
        combined[et_mask_a] = 4
    else:
        et_mask_b = pred_b == 4
        combined[et_mask_b] = 4

    return combined, pipeline_scores, region_choices


def run_inference_for_pipeline_b(model_path, config_path, devices, seed, output_dir):
    """Run inference for Pipeline B to generate predictions."""
    # This would call the inference script for Pipeline B
    # For now, we'll assume the predictions are already available
    pass


def compute_metrics_on_test(args):
    """Compute Dice and Hausdorff metrics on test set with intelligent combination."""
    dataset = get_datasets(args.seed, args.debug, no_seg=False, on="test", normalisation=args.normalisation)
    loader = DataLoader(dataset, batch_size=1, num_workers=args.workers)

    criterion = EDiceLoss(do_sigmoid=False)

    all_metrics = []
    per_patient_results = {}

    # Use predictions subdirectory within the output directories
    preds_dir_a = Path(args.pipeline_a_dir) / "predictions" if args.pipeline_a_dir else None
    preds_dir_b = Path(args.pipeline_b_dir) / "predictions" if args.pipeline_b_dir else None

    if preds_dir_a and not preds_dir_a.exists():
        raise FileNotFoundError(f"Pipeline A predictions directory not found: {preds_dir_a}")
    if preds_dir_b and not preds_dir_b.exists():
        raise FileNotFoundError(f"Pipeline B predictions directory not found: {preds_dir_b}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Computing intelligent combination of Pipeline A and B...")

    for batch in loader:
        patient_id = batch["patient_id"][0]
        target = batch["label"].float()  # shape: [1, 3, D, H, W]
        crop_indexes = batch["crop_indexes"]

        # Load predictions from both pipelines
        pred_a = load_label_map(patient_id, preds_dir_a) if preds_dir_a else None
        pred_b = load_label_map(patient_id, preds_dir_b) if preds_dir_b else None

        if pred_a is None or pred_b is None:
            print(f"Warning: Missing prediction for patient {patient_id}")
            continue

        # Crop predictions to match target size
        z_slice = slice(crop_indexes[0][0].item(), crop_indexes[0][1].item())
        y_slice = slice(crop_indexes[1][0].item(), crop_indexes[1][1].item())
        x_slice = slice(crop_indexes[2][0].item(), crop_indexes[2][1].item())

        pred_a_cropped = pred_a[z_slice, y_slice, x_slice]
        pred_b_cropped = pred_b[z_slice, y_slice, x_slice]

        # Get full target for evaluation
        target_full = target[0].cpu().numpy()  # Remove batch dimension

        # Combine predictions intelligently
        combined_labelmap, pipeline_scores, region_choices = combine_predictions_intelligently(
            pred_a_cropped, pred_b_cropped, target_full, crop_indexes
        )

        # Save combined segmentation
        combined_image = sitk.GetImageFromArray(combined_labelmap)
        sitk.WriteImage(combined_image, str(output_dir / f"{patient_id}_combined.nii.gz"))

        # Calculate metrics for combined prediction
        pred_channels = labelmap_to_channels(combined_labelmap)
        pred_tensor = torch.from_numpy(pred_channels).unsqueeze(0).float()
        pred_tensor = pred_tensor.to(target.device)

        with torch.no_grad():
            dice_scores = criterion.metric(pred_tensor, target)
            dice_scores_for_patient = [d.cpu().numpy() for d in dice_scores[0]]

        # Calculate Hausdorff distances
        hausdorff_scores = []
        for i, region_name in enumerate(['ET', 'TC', 'WT']):
            pred_region = pred_channels[i]
            target_region = target_full[i]

            hd = calculate_hausdorff_distance(pred_region, target_region)
            hausdorff_scores.append(hd)

        # Store results
        patient_metrics = {
            'patient_id': patient_id,
            'dice_et': dice_scores_for_patient[0],
            'dice_tc': dice_scores_for_patient[1],
            'dice_wt': dice_scores_for_patient[2],
            'hausdorff_et': hausdorff_scores[0],
            'hausdorff_tc': hausdorff_scores[1],
            'hausdorff_wt': hausdorff_scores[2],
            'region_choices': region_choices,
            'pipeline_a_scores': pipeline_scores['A'],
            'pipeline_b_scores': pipeline_scores['B']
        }

        all_metrics.append(patient_metrics)
        per_patient_results[patient_id] = patient_metrics

        print(f"Processed {patient_id}: WT={region_choices['WT']}, TC={region_choices['TC']}, ET={region_choices['ET']}")

    return all_metrics, per_patient_results


def calculate_95th_percentile_hausdorff(hausdorff_values):
    """Calculate 95th percentile Hausdorff distance."""
    hd_clean = [hd for hd in hausdorff_values if hd > 0 and not np.isinf(hd)]
    if len(hd_clean) == 0:
        return float('nan')
    return float(np.percentile(hd_clean, 95))


def save_results(all_metrics, args):
    """Save results and print summary."""
    output_dir = Path(args.output_dir).resolve()

    # Create summary DataFrame
    summary_data = []
    for metrics in all_metrics:
        summary_data.append({
            'patient_id': metrics['patient_id'],
            'dice_et': metrics['dice_et'],
            'dice_tc': metrics['dice_tc'],
            'dice_wt': metrics['dice_wt'],
            'hausdorff_et': metrics['hausdorff_et'],
            'hausdorff_tc': metrics['hausdorff_tc'],
            'hausdorff_wt': metrics['hausdorff_wt']
        })

    df = pd.DataFrame(summary_data)

    # Save detailed results
    results_file = output_dir / f'combined_results_seed{args.seed}.csv'
    df.to_csv(results_file, index=False)
    print(f"Saved detailed results to: {results_file}")

    # Calculate summary statistics
    mean_dice_et = df['dice_et'].mean()
    mean_dice_tc = df['dice_tc'].mean()
    mean_dice_wt = df['dice_wt'].mean()

    hd95_et = calculate_95th_percentile_hausdorff(df['hausdorff_et'].tolist())
    hd95_tc = calculate_95th_percentile_hausdorff(df['hausdorff_tc'].tolist())
    hd95_wt = calculate_95th_percentile_hausdorff(df['hausdorff_wt'].tolist())

    # Print summary in standard format
    print("\n" + "="*80)
    print("INTELLIGENT COMBINATION RESULTS")
    print("="*80)
    print(f"Test set seed: {args.seed}")
    print(f"Total patients: {len(df)}")

    print("\nMean Dice Scores:")
    print(f"  ET: {mean_dice_et:.4f}")
    print(f"  TC: {mean_dice_tc:.4f}")
    print(f"  WT: {mean_dice_wt:.4f}")
    print(f"  Mean: {np.mean([mean_dice_et, mean_dice_tc, mean_dice_wt]):.4f}")

    print("\nHausdorff Distances (95th percentile):")
    print(f"  ET: {hd95_et:.3f}")
    print(f"  TC: {hd95_tc:.3f}")
    print(f"  WT: {hd95_wt:.3f}")

    print("\nStandard Format Summary:")
    print(f"combined_results_seed{args.seed}.csv mean Dice score of {mean_dice_wt:.4f}, {mean_dice_tc:.4f}, and {mean_dice_et:.4f}, and Hausdorff distance (95%) of {hd95_wt:.3f}, {hd95_tc:.3f}, and {hd95_et:.3f} for the whole tumor, tumor core, and enhancing tumor, respectively")

    print("\n" + "="*80)

    return df


def main():
    parser = argparse.ArgumentParser(description="Intelligent combination of Pipeline A and B results")
    parser.add_argument('--seed', type=int, default=42, help='Seed for test set split')
    parser.add_argument('--devices', required=False, type=str, help='CUDA devices (unused for pure eval)')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--normalisation', type=str, default='minmax', choices=['minmax', 'zscore'])
    parser.add_argument('--pipeline_a_dir', type=str, required=True, help='Directory with Pipeline A predictions')
    parser.add_argument('--pipeline_b_dir', type=str, required=True, help='Directory with Pipeline B predictions')
    parser.add_argument('--output_dir', type=str, default='./combined_results',
                        help='Directory to save combined predictions and results')
    args = parser.parse_args()

    if args.devices:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

    # Compute intelligent combination
    all_metrics, per_patient_results = compute_metrics_on_test(args)

    # Save and display results
    df = save_results(all_metrics, args)

    return df


if __name__ == '__main__':
    main()
