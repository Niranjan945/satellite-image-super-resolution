"""
Stage 4: Blind (no-reference) Quality Evaluation
Scores every image in ground_truth / bicubic / realesrgan using OpenCV's
native BRISQUE module - no ground truth needed for the scoring itself,
we just also score ground_truth for reference/comparison.

Lower BRISQUE score = better perceived quality (less distorted-looking).

Excludes two kinds of unreliable patches:
  1. Blank/near-uniform patches (std dev too low) - e.g. black image borders.
  2. Patches where BRISQUE saturates at its extreme boundary (0 or 100) -
     this happens when a patch is statistically abnormal enough (e.g. mostly
     blank with a tiny sliver of real content) that the scoring model pins
     to its limit rather than returning a meaningful mid-range value.

Reads:  data/outputs/ground_truth/*.png
        data/outputs/bicubic/*.png
        data/outputs/realesrgan/*.png
Writes: results/logs/brisque_scores.csv
"""

import os
import csv
import cv2
import numpy as np

GT_DIR = "data/outputs/ground_truth"
BICUBIC_DIR = "data/outputs/bicubic"
SR_DIR = "data/outputs/realesrgan"
OUT_CSV = "results/logs/brisque_scores.csv"

BRISQUE_MODEL = "models/brisque_model_live.yml"
BRISQUE_RANGE = "models/brisque_range_live.yml"

STD_THRESHOLD = 10.0     # raised from 5.0 - catches near-blank, not just fully blank
SATURATION_EPS = 0.5     # scores within this of 0 or 100 are treated as saturated


def brisque_score(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return cv2.quality.QualityBRISQUE_compute(img, BRISQUE_MODEL, BRISQUE_RANGE)[0]


def is_blank(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return np.std(img) < STD_THRESHOLD


def is_saturated(score):
    return score <= SATURATION_EPS or score >= (100 - SATURATION_EPS)


def main():
    os.makedirs("results/logs", exist_ok=True)

    filenames = sorted(f for f in os.listdir(GT_DIR) if f.lower().endswith(".png"))
    print(f"Found {len(filenames)} patches.\n")

    skipped = []
    rows = []
    sum_gt, sum_bicubic, sum_sr = 0.0, 0.0, 0.0
    scored_count = 0

    for i, fname in enumerate(filenames, 1):
        gt_path = os.path.join(GT_DIR, fname)
        bicubic_path = os.path.join(BICUBIC_DIR, fname)
        sr_path = os.path.join(SR_DIR, fname)

        if is_blank(gt_path):
            skipped.append((fname, "low std dev (blank/near-uniform)"))
            print(f"[{i}/{len(filenames)}] {fname:28s} SKIPPED (blank/near-uniform)")
            continue

        gt_score = brisque_score(gt_path)
        bicubic_score = brisque_score(bicubic_path)
        sr_score = brisque_score(sr_path)

        if is_saturated(gt_score) or is_saturated(bicubic_score) or is_saturated(sr_score):
            skipped.append((fname, "BRISQUE score saturated at boundary"))
            print(f"[{i}/{len(filenames)}] {fname:28s} SKIPPED (score saturation: "
                  f"GT={gt_score:.2f} Bicubic={bicubic_score:.2f} RealESRGAN={sr_score:.2f})")
            continue

        rows.append([fname, gt_score, bicubic_score, sr_score])
        sum_gt += gt_score
        sum_bicubic += bicubic_score
        sum_sr += sr_score
        scored_count += 1

        print(f"[{i}/{len(filenames)}] {fname:28s} "
              f"GT={gt_score:7.2f}  Bicubic={bicubic_score:7.2f}  RealESRGAN={sr_score:7.2f}")

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "brisque_ground_truth", "brisque_bicubic", "brisque_realesrgan"])
        writer.writerows(rows)

    print(f"\n{len(skipped)} patch(es) excluded:")
    for fname, reason in skipped:
        print(f"    {fname}  -  {reason}")

    print(f"\n=== AVERAGE BRISQUE SCORES over {scored_count} valid patches "
          f"(lower = better perceived quality) ===")
    print(f"  Ground truth (real, undamaged): {sum_gt / scored_count:7.2f}")
    print(f"  Bicubic baseline:               {sum_bicubic / scored_count:7.2f}")
    print(f"  Real-ESRGAN:                     {sum_sr / scored_count:7.2f}")
    print(f"\nFull per-patch results saved to: {OUT_CSV}")


if __name__ == "__main__":
    main()