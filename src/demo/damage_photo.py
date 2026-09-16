"""
Demo Step 2: take a real patch cropped out of the full NavCam photo
(the one just shown full-screen in Step 1), and deliberately damage it
by downsampling - simulating what a weaker satellite camera would
actually capture. Displays full-screen for the live demo.

Usage:
    python src/demo/damage_photo.py
    python src/demo/damage_photo.py path/to/full_image.png
"""

import sys
import cv2

DEFAULT_IMAGE = "data/raw/left/ch3_nav_nrl_20230902T0617542696_d_img_gnh_038.png"
CROP_SIZE = 256      # size of the real patch we cut out
DAMAGE_SIZE = 64      # size we shrink it to, simulating a weak sensor

OUT_CROP = "demo_real_crop.png"
OUT_DAMAGED = "demo_lowres.png"


def show_fullscreen(title, img):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(title, img)
    print(f"Showing: {title}  (press any key on the image window to continue)")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE

    full_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if full_img is None:
        print(f"Could not load image: {image_path}")
        return

    h, w = full_img.shape[:2]
    cy, cx = h // 2, w // 2
    half = CROP_SIZE // 2
    crop = full_img[cy - half:cy + half, cx - half:cx + half]

    cv2.imwrite(OUT_CROP, crop)
    print(f"Real, undamaged patch: {crop.shape[1]}x{crop.shape[0]} -> saved to {OUT_CROP}")

    # NOTE: this is downsampling (shrinking the pixel grid), NOT cropping
    # and NOT compression - see project notes for the distinction.
    damaged = cv2.resize(crop, (DAMAGE_SIZE, DAMAGE_SIZE), interpolation=cv2.INTER_AREA)
    cv2.imwrite(OUT_DAMAGED, damaged)
    print(f"Damaged (simulated weak sensor): {damaged.shape[1]}x{damaged.shape[0]} -> saved to {OUT_DAMAGED}")

    damaged_display = cv2.resize(damaged, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_NEAREST)
    show_fullscreen("Damaged - simulated weak satellite camera (blocky = fewer real pixels)", damaged_display)


if __name__ == "__main__":
    main()