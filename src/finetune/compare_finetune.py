"""
Phase B follow-up: compare the ORIGINAL pretrained model against the
lightly fine-tuned one, on the same real patch - visually and with BRISQUE.

Usage:
    python src/finetune/compare_finetune.py <patch_filename>
    python src/finetune/compare_finetune.py gnh_038_r1_c3.png
"""

import sys
import cv2
import numpy as np
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

PATCHES_DIR = "data/patches/left"
ORIGINAL_WEIGHTS = "models/pretrained/RealESRGAN_x4plus.pth"
FINETUNED_WEIGHTS = "models/pretrained/RealESRGAN_x4plus_finetuned.pth"
BRISQUE_MODEL = "models/brisque_model_live.yml"
BRISQUE_RANGE = "models/brisque_range_live.yml"


def run_model(weights_path, low_res_img):
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(scale=4, model_path=weights_path, model=model,
                              tile=0, tile_pad=10, pre_pad=0, half=False)
    output, _ = upsampler.enhance(low_res_img, outscale=4)
    return output


def brisque_score(img):
    return cv2.quality.QualityBRISQUE_compute(img, BRISQUE_MODEL, BRISQUE_RANGE)[0]


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/finetune/compare_finetune.py <patch_filename>")
        return
    fname = sys.argv[1]

    ground_truth = cv2.imread(f"{PATCHES_DIR}/{fname}", cv2.IMREAD_COLOR)
    if ground_truth is None:
        print(f"Could not find patch: {PATCHES_DIR}/{fname}")
        return
    h, w = ground_truth.shape[:2]
    low_res = cv2.resize(ground_truth, (64, 64), interpolation=cv2.INTER_AREA)

    print("Running ORIGINAL model...")
    original_out = run_model(ORIGINAL_WEIGHTS, low_res)
    original_out = cv2.resize(original_out, (w, h), interpolation=cv2.INTER_CUBIC)

    print("Running FINE-TUNED model...")
    finetuned_out = run_model(FINETUNED_WEIGHTS, low_res)
    finetuned_out = cv2.resize(finetuned_out, (w, h), interpolation=cv2.INTER_CUBIC)

    cv2.imwrite("compare_original.png", original_out)
    cv2.imwrite("compare_finetuned.png", finetuned_out)

    gt_score = brisque_score(ground_truth)
    orig_score = brisque_score(original_out)
    ft_score = brisque_score(finetuned_out)

    print(f"\n=== BRISQUE scores for {fname} (lower = better) ===")
    print(f"  Ground truth:    {gt_score:.2f}")
    print(f"  Original model:  {orig_score:.2f}")
    print(f"  Fine-tuned model:{ft_score:.2f}")

    pixel_diff = np.abs(original_out.astype(np.float32) - finetuned_out.astype(np.float32)).mean()
    print(f"\nAverage pixel difference between original and fine-tuned output: {pixel_diff:.3f}")
    print("(0 would mean identical outputs; larger = the fine-tune visibly changed something)")

    panel = np.hstack([ground_truth, original_out, finetuned_out])
    window = f"[{fname}] Ground Truth | Original Model | Fine-tuned Model"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window, panel)
    print("\nPress any key on the image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()