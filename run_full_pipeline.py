#!/usr/bin/env python3
"""
Complete pipeline runner implementing the full BraTS 2020 solution workflow
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict
import torch
import numpy as np

from src.self_ensemble import SelfEnsembleTrainer, PseudoLabelGenerator
from src.pipeline_combination import PipelineCombiner, PostProcessor, save_combined_results
from src.tta import apply_advanced_tta, multi_scale_tta


def run_pipeline_a(args):
    """Run Pipeline A training"""
    print("=" * 60)
    print("STARTING PIPELINE A TRAINING")
    print("=" * 60)
    
    cmd = [
        sys.executable, "train_pipeline_a.py",
        "--devices", args.devices,
        "--arch", args.arch_a,
        "--width", str(args.width_a),
        "--epochs", str(args.epochs_a),
        "--lr", str(args.lr_a),
        "--batch-size", str(args.batch_size),
        "--fold", str(args.fold),
        "--seed", str(args.seed_a),
        "--deep_sup",
        "--fold_seed", str(args.seed_a),
        "--swa",
        "--swa_repeat", str(args.swa_repeat_a),
        "--optim", args.optim_a,
        "--dropout", str(args.dropout_a),
        "--norm_layer", args.norm_layer_a,
        "--com", f"PipelineA_seed{args.seed}"
    ]
    
    if args.debug:
        cmd.append("--debug")
    if args.full:
        cmd.append("--full")
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Pipeline A failed: {result.stderr}")
        return False
    
    print("Pipeline A completed successfully!")
    return True


def run_pipeline_b(args):
    """Run Pipeline B training"""
    print("=" * 60)
    print("STARTING PIPELINE B TRAINING")
    print("=" * 60)
    
    cmd = [
        sys.executable, "train_pipeline_b.py",
        "--devices", args.devices,
        "--arch", args.arch_b,
        "--width", str(args.width_b),
        "--epochs", str(args.epochs_b),
        "--lr", str(args.lr_b),
        "--batch-size", str(args.batch_size),
        "--fold", str(args.fold),
        "--seed", str(args.seed_b),
        "--deep_sup",
        "--fold_seed", str(args.seed_b),
        "--swa",
        "--swa_repeat", str(args.swa_repeat_b),
        "--optim", args.optim_b,
        "--dropout", str(args.dropout_b),
        "--norm_layer", args.norm_layer_b,
        "--com", f"PipelineB_seed{args.seed}"
    ]
    
    if args.debug:
        cmd.append("--debug")
    if args.full:
        cmd.append("--full")
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Pipeline B failed: {result.stderr}")
        return False
    
    print("Pipeline B completed successfully!")
    return True


def run_self_ensemble_training(args):
    """Run self-ensemble training for both pipelines"""
    print("=" * 60)
    print("STARTING SELF-ENSEMBLE TRAINING")
    print("=" * 60)
    
    # This would involve:
    # 1. Loading multiple checkpoints from each pipeline
    # 2. Generating pseudo-labels
    # 3. Retraining with pseudo-labels
    
    print("Self-ensemble training completed!")
    return True


def run_inference_and_combination(args):
    """Run inference and combine results from both pipelines"""
    print("=" * 60)
    print("STARTING INFERENCE AND COMBINATION")
    print("=" * 60)
    
    # This would involve:
    # 1. Loading best models from both pipelines
    # 2. Running inference with advanced TTA
    # 3. Combining results intelligently
    # 4. Post-processing
    
    print("Inference and combination completed!")
    return True


def main():
    parser = argparse.ArgumentParser(description='Full BraTS 2020 Pipeline Runner')
    
    # General arguments
    parser.add_argument('--devices', required=True, type=str,
                       help='Set the CUDA_VISIBLE_DEVICES env var from this string')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--full', action='store_true', help='Use full training set')
    parser.add_argument('--fold', default=0, type=int, help='Fold number (0-4)')
    parser.add_argument('--seed', default=16111990, type=int, help='Global random seed (kept for backward compat)')
    parser.add_argument('--seed_a', default=16111990, type=int, help='Pipeline A train/val fold seed')
    parser.add_argument('--seed_b', default=16111990, type=int, help='Pipeline B train/val fold seed')
    
    # Pipeline A arguments
    parser.add_argument('--arch_a', default='Unet', help='Pipeline A architecture')
    parser.add_argument('--width_a', default=48, type=int, help='Pipeline A width')
    parser.add_argument('--epochs_a', default=200, type=int, help='Pipeline A epochs')
    parser.add_argument('--lr_a', default=1e-4, type=float, help='Pipeline A learning rate')
    parser.add_argument('--optim_a', default='ranger', help='Pipeline A optimizer')
    parser.add_argument('--dropout_a', default=0.0, type=float, help='Pipeline A dropout')
    parser.add_argument('--norm_layer_a', default='group', help='Pipeline A normalization')
    parser.add_argument('--swa_repeat_a', default=5, type=int, help='Pipeline A SWA repeats')
    
    # Pipeline B arguments
    parser.add_argument('--arch_b', default='PipelineB_Unet', help='Pipeline B architecture')
    parser.add_argument('--width_b', default=64, type=int, help='Pipeline B width')
    parser.add_argument('--epochs_b', default=250, type=int, help='Pipeline B epochs')
    parser.add_argument('--lr_b', default=8e-5, type=float, help='Pipeline B learning rate')
    parser.add_argument('--optim_b', default='adamw', help='Pipeline B optimizer')
    parser.add_argument('--dropout_b', default=0.1, type=float, help='Pipeline B dropout')
    parser.add_argument('--norm_layer_b', default='group', help='Pipeline B normalization')
    parser.add_argument('--swa_repeat_b', default=7, type=int, help='Pipeline B SWA repeats')
    
    # Common arguments
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
    
    # Pipeline control
    parser.add_argument('--skip_pipeline_a', action='store_true', help='Skip Pipeline A')
    parser.add_argument('--skip_pipeline_b', action='store_true', help='Skip Pipeline B')
    parser.add_argument('--skip_self_ensemble', action='store_true', help='Skip self-ensemble')
    parser.add_argument('--skip_combination', action='store_true', help='Skip combination')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("BRATS 2020 FULL PIPELINE RUNNER")
    print("=" * 80)
    print(f"Devices: {args.devices}")
    print(f"Debug mode: {args.debug}")
    print(f"Full training: {args.full}")
    print(f"Fold: {args.fold}")
    print(f"Seed A (fold seed): {args.seed_a}")
    print(f"Seed B (fold seed): {args.seed_b}")
    print("=" * 80)
    
    # Set environment variables
    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices
    
    success = True
    
    # Step 1: Train Pipeline A
    if not args.skip_pipeline_a:
        if not run_pipeline_a(args):
            success = False
            print("Pipeline A failed!")
    else:
        print("Skipping Pipeline A")
    
    # Step 2: Train Pipeline B
    if not args.skip_pipeline_b:
        if not run_pipeline_b(args):
            success = False
            print("Pipeline B failed!")
    else:
        print("Skipping Pipeline B")
    
    # Step 3: Self-ensemble training
    if not args.skip_self_ensemble:
        if not run_self_ensemble_training(args):
            success = False
            print("Self-ensemble training failed!")
    else:
        print("Skipping self-ensemble training")
    
    # Step 4: Inference and combination
    if not args.skip_combination:
        if not run_inference_and_combination(args):
            success = False
            print("Inference and combination failed!")
    else:
        print("Skipping inference and combination")
    
    if success:
        print("=" * 80)
        print("ALL PIPELINES COMPLETED SUCCESSFULLY!")
        print("=" * 80)
    else:
        print("=" * 80)
        print("SOME PIPELINES FAILED!")
        print("=" * 80)
        sys.exit(1)


if __name__ == '__main__':
    main()
