"""
Phase B: Light proof-of-concept fine-tuning.

IMPORTANT - what this is and isn't:
  This is NOT a full, converged fine-tune (that needs GPU compute, a much
  larger dataset, and many more training steps than we can afford on CPU
  today). This is a small, honest demonstration that fine-tuning CAN nudge
  the model's behaviour on real lunar terrain - a proof of concept, not a
  finished solution. Document it as such in the report.

Strategy for speed:
  - Freeze almost the entire network (all 23 RRDB blocks stay frozen -
    these encode general "how to sharpen images" knowledge we still want).
  - Only unfreeze the FINAL output layer (conv_last), which is the layer
    most directly responsible for the final pixel colors/texture written
    out - the cheapest place to nudge behaviour with very little data/time.
  - Train on a small subset of real patches (not all 75), for a small
    number of steps - minutes, not hours.

Reads:  data/patches/left/*.png
Writes: models/pretrained/RealESRGAN_x4plus_finetuned.pth
"""

import os
import random
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from basicsr.archs.rrdbnet_arch import RRDBNet

PATCHES_DIR = "data/patches/left"
WEIGHTS_PATH = "models/pretrained/RealESRGAN_x4plus.pth"
OUT_WEIGHTS_PATH = "models/pretrained/RealESRGAN_x4plus_finetuned.pth"

NUM_TRAIN_PATCHES = 12
NUM_STEPS = 20
LEARNING_RATE = 1e-5
DAMAGE_SIZE = 64
HR_SIZE = 256


def load_training_pairs():
    all_files = sorted(f for f in os.listdir(PATCHES_DIR) if f.lower().endswith(".png"))
    random.seed(42)
    chosen = random.sample(all_files, min(NUM_TRAIN_PATCHES, len(all_files)))

    pairs = []
    for fname in chosen:
        hr = cv2.imread(os.path.join(PATCHES_DIR, fname), cv2.IMREAD_COLOR)
        lr = cv2.resize(hr, (DAMAGE_SIZE, DAMAGE_SIZE), interpolation=cv2.INTER_AREA)
        pairs.append((fname, lr, hr))
    return pairs


def to_tensor(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chw = np.transpose(rgb, (2, 0, 1))
    return torch.from_numpy(chw).unsqueeze(0)


def main():
    print(f"Loading {NUM_TRAIN_PATCHES} training patches...")
    pairs = load_training_pairs()
    print(f"Using: {[p[0] for p in pairs]}\n")

    print("Loading pretrained model...")
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    ckpt = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["params_ema"], strict=True)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.conv_last.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} total "
          f"({100*trainable/total:.3f}% of the network)\n")

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE
    )
    loss_fn = nn.L1Loss()

    model.train()
    print(f"Training for {NUM_STEPS} steps (this may take a few minutes on CPU)...\n")

    t0 = time.time()
    for step in range(1, NUM_STEPS + 1):
        fname, lr_img, hr_img = pairs[step % len(pairs)]

        lr_tensor = to_tensor(lr_img)
        hr_tensor = to_tensor(hr_img)

        optimizer.zero_grad()
        output = model(lr_tensor)

        if output.shape[-2:] != hr_tensor.shape[-2:]:
            output = nn.functional.interpolate(output, size=hr_tensor.shape[-2:], mode="bilinear")

        loss = loss_fn(output, hr_tensor)
        loss.backward()
        optimizer.step()

        print(f"  step {step}/{NUM_STEPS}  patch={fname}  loss={loss.item():.5f}")

    elapsed = time.time() - t0
    print(f"\nTraining done in {elapsed:.1f}s.")

    model.eval()
    os.makedirs(os.path.dirname(OUT_WEIGHTS_PATH), exist_ok=True)
    torch.save({"params_ema": model.state_dict()}, OUT_WEIGHTS_PATH)
    print(f"Saved fine-tuned weights to: {OUT_WEIGHTS_PATH}")


if __name__ == "__main__":
    main()