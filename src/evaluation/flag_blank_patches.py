"""
Data-quality check: find near-blank/uniform patches that will confuse BRISQUE.
A real photo patch has variation in pixel brightness (std deviation is high).
A blank/black/flat patch has almost no variation (std deviation near 0).

Reads:  data/patches/left/*.png
Prints: any patch whose standard deviation is suspiciously low - these should
        be excluded from evaluation, since BRISQUE scores on them are meaningless.
"""

import os
import cv2
import numpy as np

PATCHES_DIR = "data/patches/left"
STD_THRESHOLD = 5.0   # patches with less pixel variation than this are "flat"


def main():
    flagged = []
    for fname in sorted(os.listdir(PATCHES_DIR)):
        if not fname.lower().endswith(".png"):
            continue
        img = cv2.imread(os.path.join(PATCHES_DIR, fname), cv2.IMREAD_GRAYSCALE)
        std = np.std(img)
        if std < STD_THRESHOLD:
            flagged.append((fname, std))

    if not flagged:
        print("No blank/flat patches found - dataset looks clean.")
        return

    print(f"Found {len(flagged)} suspiciously flat patch(es) - exclude these from evaluation:\n")
    for fname, std in flagged:
        print(f"  {fname}   (pixel std dev = {std:.2f}, very low = likely blank/uniform)")


if __name__ == "__main__":
    main()