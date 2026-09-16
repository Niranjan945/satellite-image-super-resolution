"""
Demo v2: Full-image comparison, WITHOUT visible patch seams.

Improvements over v1:
  1. Overlapping patches + smooth blending at the edges (removes the
     visible grid-line seams from independently-processed tiles).
  2. Configurable damage severity, to test whether hallucination gets
     worse as we destroy more information (64x64) vs less (128x128).

IMPORTANT: this does NOT fix the hallucination itself - a pretrained
model not trained on lunar terrain will still invent unrealistic
texture. This script only removes the SEAM artifact and lets you test
how damage severity affects hallucination severity - it does not make
the AI's guesses more factually correct.

Usage:
    python src/demo/full_image_comparison_v2.py <image> [damage_size]
"""

import sys
import os
import cv2
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

PATCH_SIZE = 256
STEP = 128
WEIGHTS_PATH = "models/pretrained/RealESRGAN_x4plus.pth"


def make_weight_mask(size):
    ramp = np.hanning(size)
    ramp = np.clip(ramp, 1e-3, None)
    mask2d = np.outer(ramp, ramp).astype(np.float32)
    return mask2d[:, :, None]


def match_exposure(img, reference):
    img_f = img.astype(np.float32)
    ref_f = reference.astype(np.float32)
    img_mean, img_std = img_f.mean(), img_f.std() + 1e-6
    ref_mean, ref_std = ref_f.mean(), ref_f.std() + 1e-6
    adjusted = (img_f - img_mean) * (ref_std / img_std) + ref_mean
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def reconstruct_blended(real_full, upsampler, damage_size, use_ai):
    h, w = real_full.shape[:2]
    accum = np.zeros((h, w, 3), dtype=np.float32)
    weight_sum = np.zeros((h, w, 1), dtype=np.float32)
    mask = make_weight_mask(PATCH_SIZE)

    ys = list(range(0, h - PATCH_SIZE + 1, STEP))
    xs = list(range(0, w - PATCH_SIZE + 1, STEP))
    if ys[-1] != h - PATCH_SIZE:
        ys.append(h - PATCH_SIZE)
    if xs[-1] != w - PATCH_SIZE:
        xs.append(w - PATCH_SIZE)

    total = len(ys) * len(xs)
    count = 0
    for y0 in ys:
        for x0 in xs:
            patch = real_full[y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE]
            damaged = cv2.resize(patch, (damage_size, damage_size), interpolation=cv2.INTER_AREA)

            if use_ai:
                out, _ = upsampler.enhance(damaged, outscale=4)
                out = cv2.resize(out, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_CUBIC)
            else:
                out = cv2.resize(damaged, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_CUBIC)

            accum[y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE] += out.astype(np.float32) * mask
            weight_sum[y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE] += mask

            count += 1
            print(f"  {'AI' if use_ai else 'bicubic'} patch {count}/{total}", end="\r")

    print()
    result = accum / np.maximum(weight_sum, 1e-6)
    return np.clip(result, 0, 255).astype(np.uint8)


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/demo/full_image_comparison_v2.py <image> [damage_size]")
        return
    image_path = sys.argv[1]
    damage_size = int(sys.argv[2]) if len(sys.argv) > 2 else 64

    real_full = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if real_full is None:
        print(f"Could not load image: {image_path}")
        return

    h, w = real_full.shape[:2]
    h_crop = ((h - PATCH_SIZE) // STEP) * STEP + PATCH_SIZE
    w_crop = ((w - PATCH_SIZE) // STEP) * STEP + PATCH_SIZE
    real_full = real_full[:h_crop, :w_crop]
    print(f"Working image size: {w_crop}x{h_crop}, damage_size={damage_size}x{damage_size}")

    print("Loading Real-ESRGAN model...")
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(scale=4, model_path=WEIGHTS_PATH, model=model,
                              tile=0, tile_pad=10, pre_pad=0, half=False)
    print("Model loaded.\n")

    print("Reconstructing with bicubic (blended, seamless)...")
    bicubic_full = reconstruct_blended(real_full, upsampler, damage_size, use_ai=False)

    print("Reconstructing with AI (blended, seamless)...")
    ai_full = reconstruct_blended(real_full, upsampler, damage_size, use_ai=True)

    base = os.path.splitext(os.path.basename(image_path))[0]
    tag = f"{base}_damage{damage_size}"
    cv2.imwrite(f"demo_{tag}_real.png", real_full)
    cv2.imwrite(f"demo_{tag}_bicubic.png", bicubic_full)
    cv2.imwrite(f"demo_{tag}_ai.png", ai_full)
    print(f"\nSaved: demo_{tag}_real.png / _bicubic.png / _ai.png")

    bicubic_display = match_exposure(bicubic_full, real_full)
    ai_display = match_exposure(ai_full, real_full)

    panel = np.hstack([real_full, bicubic_display, ai_display])
    window = f"[{tag}] REAL | BICUBIC (blended) | AI (blended)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window, panel)
    print("Press any key on the image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()