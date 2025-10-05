#!/usr/bin/env python3

"""

  Summarize mean Dice and 95th percentile Hausdorff distance per label (WT, TC, ET)

  from pipeline_a_test_results/pipeline_a_test_results_seed42.csv and print a

  one-line summary in the requested format.



  Usage:

      python sumary_results.py

"""

from __future__ import annotations



import argparse

from pathlib import Path

import numpy as np

import pandas as pd





def compute_summary(csv_path: Path) -> str:

    df = pd.read_csv(csv_path)



    # Coerce numeric columns and handle missing/invalid values

    for col in ("haussdorf", "dice", "sens", "spec"):

      if col in df.columns:

        df[col] = pd.to_numeric(df[col], errors="coerce")



    # Keep only rows for the three standard labels

    valid_labels = ["WT", "TC", "ET"]

    df = df[df["label"].isin(valid_labels)].copy()



    def agg_for_label(label: str) -> tuple[float, float]:

      d = df[df["label"] == label]



      # Mean Dice (ignore NaN)

      mean_dice = float(d["dice"].mean(skipna=True))



      # 95th percentile of Hausdorff across cases:

      # - drop NaN

      # - drop non-positive values (0 often appears as placeholder for invalid HD)

      hd = d["haussdorf"].dropna()

      hd = hd[hd > 0]

      if len(hd) == 0:

        hd95 = float("nan")

      else:

        hd95 = float(np.percentile(hd.to_numpy(), 95))

      return mean_dice, hd95



    wt_dice, wt_hd95 = agg_for_label("WT")

    tc_dice, tc_hd95 = agg_for_label("TC")

    et_dice, et_hd95 = agg_for_label("ET")



    # Build the requested sentence, keeping the order: WT, TC, ET

    fn = Path(csv_path).name

    summary = (

      f"{fn} mean Dice score of "

      f"{wt_dice:.4f}, {tc_dice:.4f}, and {et_dice:.4f}, "

      f"and Hausdorff distance (95%) of "

      f"{wt_hd95:.3f}, {tc_hd95:.3f}, and {et_hd95:.3f} "

      f"for the whole tumor, tumor core, and enhancing tumor, respectively"

    )

    return summary



def main():

    file_path = "pipeline_a_test_results/pipeline_a_test_results_seed42.csv"

    print(compute_summary(file_path))





if __name__ == "__main__":

    main()
