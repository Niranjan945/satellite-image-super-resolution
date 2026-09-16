"""
Stage 3: Super-Resolution
For each real patch we cut in Stage 2:
  1. Load the sharp 256x256 patch (our ground truth)
  2. Downsample it to 64x64 (simulate a weaker sensor - NOT cropping, NOT compression,
     see project notes: this shrinks detail evenly, keeps the whole scene)
  3. Run Real-ESRGAN on the 64x64 version to reconstruct a 256x256 output
  4. Also create a plain bicubic-upscaled 256x256 version, as the "dumb" baseline
  5. Save all three (ground truth / bicubic / real-esrgan) so we can compare them

Reads:  data/patches/left/*.png   (we start with the left camera only for now)
Writes: data/outputs/ground_truth/*.png
        data/outputs/bicubic/*.png
        data/outputs/realesrgan/*.png
"""

import os
import time
import cv2
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

PATCHES_DIR = "data/patches/left"
OUT_GT = "data/outputs/ground_truth"
OUT_BICUBIC = "data/outputs/bicubic"
OUT_SR = "data/outputs/realesrgan"
WEIGHTS_PATH = "models/pretrained/RealESRGAN_x4plus.pth"

DOWNSAMPLE_SIZE = 64   # simulate a weak sensor: shrink 256x256 -> 64x64
UPSCALE_FACTOR = 4     # 64 * 4 = 256, back to original patch size


def load_upsampler():
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    return RealESRGANer(
        scale=4,
        model_path=WEIGHTS_PATH,
        model=model,
        tile=0, tile_pad=10, pre_pad=0, half=False,
    )


def main():
    os.makedirs(OUT_GT, exist_ok=True)
    os.makedirs(OUT_BICUBIC, exist_ok=True)
    os.makedirs(OUT_SR, exist_ok=True)

    patch_files = sorted(f for f in os.listdir(PATCHES_DIR) if f.lower().endswith(".png"))
    print(f"Found {len(patch_files)} patches to process.\n")

    print("Loading Real-ESRGAN model...")
    upsampler = load_upsampler()
    print("Model loaded.\n")

    total_time = 0.0
    for i, fname in enumerate(patch_files, 1):
        gt_path = os.path.join(PATCHES_DIR, fname)
        ground_truth = cv2.imread(gt_path, cv2.IMREAD_COLOR)
        h, w = ground_truth.shape[:2]

        # Step 1: simulate a weak sensor by downsampling (NOT cropping)
        low_res = cv2.resize(ground_truth, (DOWNSAMPLE_SIZE, DOWNSAMPLE_SIZE),
                              interpolation=cv2.INTER_AREA)

        # Step 2: dumb baseline - just stretch the low-res image back up
        bicubic = cv2.resize(low_res, (w, h), interpolation=cv2.INTER_CUBIC)

        # Step 3: Real-ESRGAN's attempt to reconstruct detail
        t0 = time.time()
        sr_output, _ = upsampler.enhance(low_res, outscale=UPSCALE_FACTOR)
        t1 = time.time()
        total_time += (t1 - t0)

        # Real-ESRGAN's output size depends on input size x scale; force it to match
        # the ground truth exactly so all three images are directly comparable
        sr_output = cv2.resize(sr_output, (w, h), interpolation=cv2.INTER_CUBIC)

        cv2.imwrite(os.path.join(OUT_GT, fname), ground_truth)
        cv2.imwrite(os.path.join(OUT_BICUBIC, fname), bicubic)
        cv2.imwrite(os.path.join(OUT_SR, fname), sr_output)

        print(f"[{i}/{len(patch_files)}] {fname}  (inference {t1 - t0:.2f}s)")

    print(f"\nDone. Total Real-ESRGAN inference time: {total_time:.2f}s "
          f"({total_time / len(patch_files):.2f}s per patch on average).")
    print(f"\nOutputs saved to:")
    print(f"  {OUT_GT}")
    print(f"  {OUT_BICUBIC}")
    print(f"  {OUT_SR}")


if __name__ == "__main__":
    main()