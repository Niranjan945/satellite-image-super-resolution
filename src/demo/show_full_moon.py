"""
Demo helper: display the FULL, original 1024x1024 NavCam image,
full-screen, before we crop/damage anything - for a live demo's
opening "here's the real photo" moment.

Usage:
    python src/demo/show_full_moon.py                  (uses a default image)
    python src/demo/show_full_moon.py path/to/image.png (uses a specific one)
"""

import sys
import cv2

DEFAULT_IMAGE = "data/raw/left/ch3_nav_nrl_20230902T0617542696_d_img_gnh_038.png"

def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Could not load image: {image_path}")
        print("Tip: pass a specific file, e.g.")
        print("  python src/demo/show_full_moon.py data/raw/left/<filename>.png")
        return

    h, w = img.shape[:2]
    print(f"Showing: {image_path}")
    print(f"Real resolution: {w}x{h}")

    window_name = "Real Chandrayaan-3 NavCam Photo (full resolution, untouched)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window_name, img)
    print("Press any key (with the image window focused) to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
EOF