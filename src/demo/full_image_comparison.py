"""
Demo: Full-image comparison on ANY input photo, not just one small patch.
Cuts the image into a grid of patches, damages (downsamples) each one,
reconstructs the full image two ways (bicubic vs Real-ESRGAN), and
displays a FAIR, brightness+contrast-normalized side-by-side of:
REAL FULL PHOTO | BICUBIC RECONSTRUCTION | AI RECONSTRUCTION

Usage:
    python src/demo/full_image_comparison.py path/to/image.png
"""

import sys
import os
import cv2
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

PATCH_SIZE = 256
DAMAGE_SIZE = 64
WEIGHTS_PATH = "models/pretrained/RealESRGAN_x4plus.pth"


def match_exposure(img, reference):
    """
    Adjusts img's brightness AND contrast to match reference's, for FAIR
    VISUAL DISPLAY ONLY. Matches both the mean (brightness) and standard
    deviation (contrast/spread of tones - "saturation") of pixel intensities.
    Never touches the saved/scored data - display purposes only.
    """
    img_f = img.astype(np.float32)
    ref_f = reference.astype(np.float32)

    img_mean, img_std = img_f.mean(), img_f.std() + 1e-6
    ref_mean, ref_std = ref_f.mean(), ref_f.std() + 1e-6

    adjusted = (img_f - img_mean) * (ref_std / img_std) + ref_mean
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    return adjusted


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/demo/full_image_comparison.py <path_to_image>")
        return
    image_path = sys.argv[1]

    real_full = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if real_full is None:
        print(f"Could not load image: {image_path}")
        return

    h, w = real_full.shape[:2]
    h_crop = (h // PATCH_SIZE) * PATCH_SIZE
    w_crop = (w // PATCH_SIZE) * PATCH_SIZE
    if h_crop == 0 or w_crop == 0:
        print(f"Image too small ({w}x{h}) for patch size {PATCH_SIZE}. Skipping.")
        return
    real_full = real_full[:h_crop, :w_crop]
    print(f"Working image size (cropped to clean grid): {w_crop}x{h_crop}")

    bicubic_full = np.zeros_like(real_full)
    ai_full = np.zeros_like(real_full)

    print("Loading Real-ESRGAN model...")
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(scale=4, model_path=WEIGHTS_PATH, model=model,
                              tile=0, tile_pad=10, pre_pad=0, half=False)
    print("Model loaded.\n")

    n_rows = h_crop // PATCH_SIZE
    n_cols = w_crop // PATCH_SIZE
    total = n_rows * n_cols
    count = 0

    for row in range(n_rows):
        for col in range(n_cols):
            y0, y1 = row * PATCH_SIZE, (row + 1) * PATCH_SIZE
            x0, x1 = col * PATCH_SIZE, (col + 1) * PATCH_SIZE
            patch = real_full[y0:y1, x0:x1]

            damaged = cv2.resize(patch, (DAMAGE_SIZE, DAMAGE_SIZE), interpolation=cv2.INTER_AREA)

            bicubic_patch = cv2.resize(damaged, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_CUBIC)
            bicubic_full[y0:y1, x0:x1] = bicubic_patch

            ai_patch, _ = upsampler.enhance(damaged, outscale=4)
            ai_patch = cv2.resize(ai_patch, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_CUBIC)
            ai_full[y0:y1, x0:x1] = ai_patch

            count += 1
            print(f"  patch {count}/{total} done", end="\r")

    print(f"\nAll {total} patches processed and stitched back into full images.")

    base = os.path.splitext(os.path.basename(image_path))[0]
    cv2.imwrite(f"demo_{base}_real.png", real_full)
    cv2.imwrite(f"demo_{base}_bicubic.png", bicubic_full)
    cv2.imwrite(f"demo_{base}_ai.png", ai_full)

    bicubic_display = match_exposure(bicubic_full, real_full)
    ai_display = match_exposure(ai_full, real_full)

    panel = np.hstack([real_full, bicubic_display, ai_display])
    window = f"[{base}] REAL | BICUBIC (exposure-matched) | AI (exposure-matched)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window, panel)
    print("Press any key on the image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()