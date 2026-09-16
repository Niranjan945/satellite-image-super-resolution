"""
Stage 2: Preprocessing
Cuts raw NavCam Left/Right stereo pairs into smaller, matched patches.

Reads:  data/raw/left/*.png , data/raw/right/*.png
Writes: data/patches/left/*.png , data/patches/right/*.png

Matching logic: NavCam filenames look like
    ch3_nav_nrl_20230902T0617542696_d_img_gnh_038.png   (Left)
    ch3_nav_nrr_20230902T0618343982_d_img_gnh_038.png   (Right)
The part after "_d_img_" (e.g. "gnh_038") is the shared pair ID.
This script matches Left/Right files by that ID, then cuts each into
non-overlapping patches, keeping the same crop region from both sides.
"""

import os
import cv2

RAW_LEFT = "data/raw/left"
RAW_RIGHT = "data/raw/right"
OUT_LEFT = "data/patches/left"
OUT_RIGHT = "data/patches/right"
PATCH_SIZE = 256   # each tile will be 256x256 pixels


def pair_id(filename):
    """Extract the shared pair identifier from a NavCam filename."""
    # e.g. 'ch3_nav_nrl_20230902T0617542696_d_img_gnh_038.png' -> 'gnh_038'
    tail = filename.split("_d_img_")[1]
    return os.path.splitext(tail)[0]


def find_matched_pairs():
    left_files = {pair_id(f): f for f in os.listdir(RAW_LEFT) if f.lower().endswith(".png")}
    right_files = {pair_id(f): f for f in os.listdir(RAW_RIGHT) if f.lower().endswith(".png")}

    common_ids = sorted(set(left_files) & set(right_files))
    missing_left = sorted(set(right_files) - set(left_files))
    missing_right = sorted(set(left_files) - set(right_files))

    if missing_left:
        print(f"[warn] {len(missing_left)} right image(s) have no matching left: {missing_left}")
    if missing_right:
        print(f"[warn] {len(missing_right)} left image(s) have no matching right: {missing_right}")

    return [(pid, left_files[pid], right_files[pid]) for pid in common_ids]


def cut_patches(img, patch_size):
    """Yield (row, col, patch) for every non-overlapping patch that fully fits."""
    h, w = img.shape[:2]
    for row in range(0, h - patch_size + 1, patch_size):
        for col in range(0, w - patch_size + 1, patch_size):
            patch = img[row:row + patch_size, col:col + patch_size]
            yield row // patch_size, col // patch_size, patch


def main():
    os.makedirs(OUT_LEFT, exist_ok=True)
    os.makedirs(OUT_RIGHT, exist_ok=True)

    pairs = find_matched_pairs()
    print(f"Found {len(pairs)} matched Left/Right pairs.\n")

    total_patches = 0
    for pid, left_name, right_name in pairs:
        left_img = cv2.imread(os.path.join(RAW_LEFT, left_name), cv2.IMREAD_COLOR)
        right_img = cv2.imread(os.path.join(RAW_RIGHT, right_name), cv2.IMREAD_COLOR)

        if left_img is None or right_img is None:
            print(f"[error] could not read pair '{pid}', skipping.")
            continue

        if left_img.shape != right_img.shape:
            print(f"[warn] pair '{pid}' has mismatched shapes {left_img.shape} vs "
                  f"{right_img.shape} — cutting each independently, rows/cols may not "
                  f"line up between sides.")

        count_this_pair = 0
        for row, col, patch in cut_patches(left_img, PATCH_SIZE):
            out_name = f"{pid}_r{row}_c{col}.png"
            cv2.imwrite(os.path.join(OUT_LEFT, out_name), patch)
            count_this_pair += 1

        for row, col, patch in cut_patches(right_img, PATCH_SIZE):
            out_name = f"{pid}_r{row}_c{col}.png"
            cv2.imwrite(os.path.join(OUT_RIGHT, out_name), patch)

        print(f"  pair '{pid}': {left_img.shape[1]}x{left_img.shape[0]} -> "
              f"{count_this_pair} patches of {PATCH_SIZE}x{PATCH_SIZE}")
        total_patches += count_this_pair

    print(f"\nDone. {total_patches} matched patch-pairs written to:")
    print(f"  {OUT_LEFT}")
    print(f"  {OUT_RIGHT}")


if __name__ == "__main__":
    main()