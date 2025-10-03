import argparse
import os
import pathlib
import numpy as np
import torch
import SimpleITK as sitk
from torch.utils.data import DataLoader

from src.dataset import get_datasets
from src.loss.dice import EDiceLoss


def compute_dice_on_test(seed: int, debug: bool, batch_size: int = 1, normalisation: str = "minmax"):
    dataset = get_datasets(seed, debug, no_seg=False, on="test", normalisation=normalisation)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=2)

    criterion = EDiceLoss(do_sigmoid=False)

    dices = []
    for batch in loader:
        targets = batch["label"].float()
        # TODO: replace this placeholder with real predictions loaded from NIfTI
        preds = targets  # Placeholder for demonstration

        with torch.no_grad():
            d = criterion.metric(preds, targets)
            dices.extend([[x.item() for x in per_patient] for per_patient in d])

    dices_np = np.array(dices)
    mean_dice = dices_np.mean(axis=0) if len(dices_np) > 0 else np.array([0, 0, 0])
    return mean_dice, dices_np


def main():
    parser = argparse.ArgumentParser(description="Compute Dice on held-out test split (with labels)")
    parser.add_argument('--seed', type=int, default=16111990, help='fold seed used to build the fixed test split')
    parser.add_argument('--devices', required=False, type=str, help='CUDA devices (unused for pure eval)')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--normalisation', type=str, default='minmax', choices=['minmax', 'zscore'])
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.devices:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

    mean_dice, dices_np = compute_dice_on_test(args.seed, args.debug, args.batch_size, args.normalisation)
    print(f"Mean Dice [ET, TC, WT]: {mean_dice}")

    out_dir = pathlib.Path('./preds').resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"test_dice_seed{args.seed}.npy", dices_np)
    np.save(out_dir / f"test_dice_mean_seed{args.seed}.npy", mean_dice)


if __name__ == '__main__':
    main()


