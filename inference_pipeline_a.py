#!/usr/bin/env python3
"""
Inference script for Pipeline A - Calculate Dice and Hausdorff metrics on test set
"""
import argparse
import os
import pathlib
import time
from datetime import datetime
from types import SimpleNamespace

import SimpleITK as sitk
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.cuda.amp import autocast

from src import models
from src.dataset import get_datasets
from src.dataset.batch_utils import pad_batch1_to_compatible_size
from src.models import get_norm_layer
from src.utils import calculate_metrics, HAUSSDORF, DICE, SENS, SPEC, METRICS

parser = argparse.ArgumentParser(description='Pipeline A Inference - Calculate Dice and Hausdorff on test set')
parser.add_argument('--model_path', required=True, type=str,
                    help='Path to the trained model checkpoint (.pth.tar)')
parser.add_argument('--config_path', required=True, type=str,
                    help='Path to the model config YAML file')
parser.add_argument('--devices', required=True, type=str,
                    help='Set the CUDA_VISIBLE_DEVICES env var from this string')
parser.add_argument('--seed', default=42, type=int,
                    help='Seed for test set split (should match training seed)')
parser.add_argument('--output_dir', default='./inference_results',
                    help='Directory to save inference results')
parser.add_argument('--threshold', default=0.5, type=float,
                    help='Threshold for binary segmentation')


def load_model_config(config_path):
    """Load model configuration from YAML file"""
    config_file = pathlib.Path(config_path).resolve()
    with config_file.open("r") as file:
        config = yaml.safe_load(file)
        return SimpleNamespace(**config)


def load_model_checkpoint(model_path, model):
    """Load trained model weights"""
    if not os.path.isfile(model_path):
        raise ValueError(f"Model checkpoint not found: {model_path}")

    print(f"Loading checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')

    # Handle DataParallel state_dict
    state_dict = checkpoint['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # remove 'module.' prefix
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'N/A')}")
    return model


def generate_predictions(model, data_loader, args):
    """Generate predictions for the test set"""
    model.eval()
    model.cuda()

    predictions = []
    patient_info = []

    print("Generating predictions...")
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            patient_id = batch["patient_id"][0]
            ref_path = batch["seg_path"][0]
            crops_idx = batch["crop_indexes"]

            # Get input image
            inputs = batch["image"]
            inputs, pads = pad_batch1_to_compatible_size(inputs)
            inputs = inputs.cuda()

            # Generate prediction
            with autocast():
                if hasattr(model, 'deep_supervision') and model.deep_supervision:
                    pre_segs, _ = model(inputs)
                else:
                    pre_segs = model(inputs)
                pre_segs = torch.sigmoid(pre_segs)

            # Remove padding
            maxz = pre_segs.size(2) - pads[0]
            maxy = pre_segs.size(3) - pads[1]
            maxx = pre_segs.size(4) - pads[2]
            pre_segs = pre_segs[:, :, 0:maxz, 0:maxy, 0:maxx].cpu()

            # Create full-size segmentation canvas
            canvas_dhw = (155, 240, 240)  # Standard BraTS volume size
            segs = torch.zeros((1, 3, *canvas_dhw))

            # Place prediction in correct location
            segs[0, :, slice(*crops_idx[0]), slice(*crops_idx[1]), slice(*crops_idx[2])] = pre_segs[0]

            # Convert to binary segmentation
            segs_binary = segs[0].numpy() > args.threshold

            # Convert to BraTS labelmap format
            et = segs_binary[0]
            net = np.logical_and(segs_binary[1], np.logical_not(et))
            ed = np.logical_and(segs_binary[2], np.logical_not(segs_binary[1]))

            labelmap = np.zeros(segs_binary[0].shape, dtype=np.uint8)
            labelmap[et] = 4  # Enhancing tumor
            labelmap[net] = 1  # Necrotic core
            labelmap[ed] = 2   # Edema

            predictions.append(labelmap)
            patient_info.append({
                'patient_id': patient_id,
                'ref_path': ref_path,
                'crops_idx': crops_idx
            })

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(data_loader)} patients")

    return predictions, patient_info


def calculate_metrics_for_predictions(predictions, patient_info, args):
    """Calculate Dice and Hausdorff metrics for all predictions"""
    all_metrics = []

    print("Calculating metrics...")
    for i, (pred, info) in enumerate(zip(predictions, patient_info)):
        patient_id = info['patient_id']
        ref_path = info['ref_path']

        # Load ground truth
        ref_img = sitk.ReadImage(ref_path)
        ref_seg = sitk.GetArrayFromImage(ref_img)

        # Convert ground truth to channels format (ET, TC, WT)
        refmap_et = ref_seg == 4
        refmap_tc = np.logical_or(refmap_et, ref_seg == 1)
        refmap_wt = np.logical_or(refmap_tc, ref_seg == 2)
        refmap = np.stack([refmap_et, refmap_tc, refmap_wt])

        # Convert prediction to channels format
        pred_et = pred == 4
        pred_tc = np.logical_or(pred == 4, pred == 1)
        pred_wt = np.logical_or(pred_tc, pred == 2)
        predmap = np.stack([pred_et, pred_tc, pred_wt])

        # Calculate metrics for this patient
        patient_metrics = calculate_metrics(predmap, refmap, patient_id, tta=False)

        # Add overall patient information
        for metric in patient_metrics:
            metric['patient_index'] = i

        all_metrics.extend(patient_metrics)

        if (i + 1) % 10 == 0:
            print(f"Calculated metrics for {i + 1}/{len(predictions)} patients")

    return all_metrics


def save_results(all_metrics, args):
    """Save metrics results to files"""
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create DataFrame with all metrics
    df = pd.DataFrame(all_metrics)

    # Save detailed results
    results_file = output_dir / f'pipeline_a_test_results_seed{args.seed}.csv'
    df.to_csv(results_file, index=False)
    print(f"Saved detailed results to: {results_file}")

    # Calculate and save summary statistics
    summary = df.groupby('label')[METRICS].agg(['mean', 'std', 'min', 'max', 'count'])
    summary_file = output_dir / f'pipeline_a_test_summary_seed{args.seed}.csv'
    summary.to_csv(summary_file)
    print(f"Saved summary statistics to: {summary_file}")

    # Print summary to console
    print("\n" + "="*60)
    print("PIPELINE A TEST SET RESULTS")
    print("="*60)
    print(f"Test set seed: {args.seed}")
    print(f"Threshold: {args.threshold}")
    print(f"Total patients: {len(df) // 3}")  # 3 labels per patient

    print("\nMean Dice Scores:")
    dice_means = df[df['label'] == 'ET']['dice'].mean(), \
                 df[df['label'] == 'TC']['dice'].mean(), \
                 df[df['label'] == 'WT']['dice'].mean()
    print(f"  ET: {dice_means[0]:.4f}")
    print(f"  TC: {dice_means[1]:.4f}")
    print(f"  WT: {dice_means[2]:.4f}")
    print(f"  Mean: {np.mean(dice_means):.4f}")

    print("\nMean Hausdorff Distances:")
    hausdorff_means = df[df['label'] == 'ET']['haussdorf'].mean(), \
                      df[df['label'] == 'TC']['haussdorf'].mean(), \
                      df[df['label'] == 'WT']['haussdorf'].mean()
    print(f"  ET: {hausdorff_means[0]:.2f}")
    print(f"  TC: {hausdorff_means[1]:.2f}")
    print(f"  WT: {hausdorff_means[2]:.2f}")
    print(f"  Mean: {np.mean(hausdorff_means):.2f}")

    print("\n" + "="*60)

    return df, summary


def main(args):
    """Main inference function"""
    # Setup
    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices
    ngpus = torch.cuda.device_count()
    print(f"Running inference with {ngpus} GPUs")

    # Load model configuration
    model_args = load_model_config(args.config_path)
    print(f"Model configuration: {model_args.arch}, width={model_args.width}")

    # Create model
    model_maker = getattr(models, model_args.arch)
    model = model_maker(
        4, 3,
        width=model_args.width,
        deep_supervision=model_args.deep_sup,
        norm_layer=get_norm_layer(model_args.norm_layer),
        dropout=model_args.dropout
    )

    # Load trained weights
    model = load_model_checkpoint(args.model_path, model)

    # Setup test dataset
    print(f"Loading test set (seed={args.seed})...")
    test_dataset = get_datasets(args.seed, debug=False, no_seg=False, on="test")
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, num_workers=2, shuffle=False
    )

    print(f"Test set size: {len(test_dataset)} patients")

    # Generate predictions
    predictions, patient_info = generate_predictions(model, test_loader, args)

    # Calculate metrics
    all_metrics = calculate_metrics_for_predictions(predictions, patient_info, args)

    # Save and display results
    df, summary = save_results(all_metrics, args)

    return df, summary


if __name__ == '__main__':
    arguments = parser.parse_args()
    main(arguments)
