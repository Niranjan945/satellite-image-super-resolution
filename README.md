
# Satellite Image Super-Resolution with Blind Evaluation

> **AI-Powered Satellite Imagery Enhancement & Trust Verification Pipeline**  

[![Python Version](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?style=for-the-badge&logo=cplusplus)](https://isocpp.org/)
[![ISRO Data](https://img.shields.io/badge/Data-ISRO%20PRADAN-orange?style=for-the-badge)](https://pradan.issdc.gov.in)

---

## 🌐 Pipeline Status

| Component | Technology | Status |
|---------|-----|--------|
| **Data Processing** | PRADAN (ISRO) / OpenCV | ✅ Active |
| **Super-Resolution** | PyTorch / Real-ESRGAN | ✅ Verified |
| **Blind Evaluation** | BRISQUE (OpenCV Native) | ✅ Online |
| **Edge Deployment** | C++17 / ONNX Runtime | ✅ Compiled |

---

## 📋 Project Overview

An end-to-end pipeline that applies AI-based super-resolution to real Chandrayaan-3 NavCam imagery, independently evaluates output quality without a reference image (blind evaluation), and deploys the trained model through a standalone C++ inference engine. 

Built in response to **ISRO's Bharatiya Antariksh Hackathon** problem statement on dual-image super-resolution and blind evaluation for satellite imagery.

### **Key Achievements**
- ✅ **Full 7-Stage Pipeline** - Data collection through C++ deployment, all working on real data
- ✅ **Real ISRO Data** - Genuine Chandrayaan-3 NavCam stereo (Left/Right) imagery utilized
- ✅ **Verified Bit-Exact Export** - PyTorch → ONNX conversion confirmed accurate to 8e-7 max pixel deviation
- ✅ **Independent Failure-Mode Discovery** - Identified and triple-verified a structural hallucination problem in a widely-used pretrained model

---

## ⚡ Features

### **Data Processing Pipeline**
- Real Chandrayaan-3 NavCam Left/Right stereo pair collection
- Automated stereo-pair matching by filename ID
- Patch-based preprocessing (256×256 tiles) to handle high-resolution inputs
- Automatic blank and corrupted-patch detection and exclusion

### **Super-Resolution Engine**
- Pretrained Real-ESRGAN inference on all patches
- Benchmarked directly against a bicubic baseline for delta measurements
- Investigated hallucination severity based on input damage scaling (64x64 vs 128x128)

### **Blind Quality Evaluation**
- No-reference (blind) quality scoring using OpenCV's native BRISQUE implementation
- Automatic exclusion of statistically invalid or score-saturated patches
- Cross-validation of metric reliability against human visual inspection

### **Edge Deployment**
- PyTorch to ONNX model export, numerically verified against the original model
- Standalone C++17 inference engine with zero Python dependency
- Direct Python vs. C++ inference-time benchmarking

---

## 🛠️ Technology Stack

### **Data & Preprocessing**
```text
Python 3.11      →  Core pipeline orchestration language
OpenCV           →  Image I/O, patch cutting, downsampling
NumPy            →  Array and statistical mathematical operations

```

### **Model & Evaluation**

```text
PyTorch          →  Deep learning framework and tensor operations
basicsr          →  RRDBNet architecture (Real-ESRGAN backbone)
realesrgan       →  Pretrained super-resolution wrapper
opencv-contrib   →  Native BRISQUE no-reference quality scoring

```

### **Export & Deployment**

```text
ONNX             →  Framework-neutral model export format
ONNX Runtime C++ →  Standalone model inference engine
C++17            →  High-performance deployment language
OpenCV C++       →  Image I/O and processing in the compiled engine

```

---

## 🏗️ Architecture

```text
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│   DATA LAYER    │    │   MODEL LAYER   │    │ EVALUATION LAYER│    │ DEPLOYMENT LAYER │
│  PRADAN NavCam  │───►│  Real-ESRGAN    │───►│  BRISQUE        │───►│  ONNX + C++      │
│  Stereo Pairs   │    │  PyTorch        │    │  (blind, no-ref)│    │  Standalone      │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └──────────────────┘

```

### **Project Structure**

```text
satellite-sr-project/
├── data/
│   ├── raw/                # Original downloaded NavCam images
│   ├── patches/            # Cropped, aligned 256x256 tiles
│   ├── demo/               # Sample images used for testing
│   └── outputs/            # Ground truth / bicubic / SR outputs
├── src/
│   ├── data_prep/          # make_patches.py (cuts/matches pairs)
│   ├── sr_model/           # run_sr.py (runs Real-ESRGAN)
│   ├── evaluation/         # score_brisque.py, flag_blank_patches.py
│   ├── export/             # export_onnx.py (verified bit-exact)
│   ├── finetune/           # finetune_lite.py, compare_finetune.py
│   └── demo/               # full_image_comparison.py
├── cpp_engine/
│   ├── src/main.cpp        # Standalone C++ engine source
│   ├── build/              # Compiled binaries
│   └── onnxruntime/        # ONNX Runtime C++ binaries
├── models/
│   ├── pretrained/         # RealESRGAN_x4plus.pth
│   └── onnx/               # Exported .onnx model
├── results/                # BRISQUE score CSVs
└── README.md

```

---

## 🚀 Quick Start

### **Prerequisites**

* Python 3.10 or 3.11 (Required for `basicsr` and `lmdb` compatibility)
* Visual Studio Build Tools with C++ workload (For C++ engine)
* Git for version control

### **Local Python Development**

1. **Clone Repository**
```bash
git clone [https://github.com/Niranjan945/satellite-sr-project.git](https://github.com/Niranjan945/satellite-sr-project.git)
cd satellite-sr-project

```


2. **Python Environment Setup**
```bash
py -3.11 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install --no-build-isolation basicsr

```


*Note: Open `venv\Lib\site-packages\basicsr\data\degradations.py` and change `functional_tensor` to `functional` in the `rgb_to_grayscale` import.*
3. **Download Assets**
```bash
mkdir models\pretrained
curl -L -o models\pretrained\RealESRGAN_x4plus.pth [https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth)
curl -L -o models\brisque_model_live.yml [https://raw.githubusercontent.com/opencv/opencv_contrib/master/modules/quality/samples/brisque_model_live.yml](https://raw.githubusercontent.com/opencv/opencv_contrib/master/modules/quality/samples/brisque_model_live.yml)
curl -L -o models\brisque_range_live.yml [https://raw.githubusercontent.com/opencv/opencv_contrib/master/modules/quality/samples/brisque_range_live.yml](https://raw.githubusercontent.com/opencv/opencv_contrib/master/modules/quality/samples/brisque_range_live.yml)

```


4. **Data Sourcing**
Download NavCam Left/Right stereo pairs from [PRADAN](https://pradan.issdc.gov.in) and place them in `data\raw\left` and `data\raw\right`.

---

## 🔧 Configuration

### **Key Pipeline Parameters**

```env
PATCH_SIZE=256           # Size of cropped tiles from raw NavCam images
DAMAGE_SIZE=64           # Simulated weak-sensor downsample size
UPSCALE_FACTOR=4         # Real-ESRGAN's native upscale factor
STD_THRESHOLD=10.0       # Minimum pixel variance to accept valid patch

```

### **Model Paths**

```env
WEIGHTS_PATH=models/pretrained/RealESRGAN_x4plus.pth
ONNX_PATH=models/onnx/RealESRGAN_x4plus.onnx
BRISQUE_MODEL=models/brisque_model_live.yml
BRISQUE_RANGE=models/brisque_range_live.yml

```

---

## 📚 Pipeline Reference

### **Data & Execution**

```text
python src/data_prep/make_patches.py       # Cut and align stereo patches
python src/sr_model/run_sr.py              # Execute SR and baseline models
python src/evaluation/flag_blank_patches.py# Detect near-uniform patches
python src/evaluation/score_brisque.py     # Blind scoring, excluding invalid

```

### **Export & Fine-tuning**

```text
python src/export/export_onnx.py           # Export ONNX (verified exact)
python src/finetune/finetune_lite.py       # Proof-of-concept fine-tune
python src/finetune/compare_finetune.py    # Before/after comparison

```

---

## 🔒 Integrity & Verification Measures

* **Bit-Exact Export Verification:** PyTorch and ONNX outputs compared directly, confirmed to differ by at most 8e-7 (floating-point noise only).
* **Data-Quality Filtering:** Automatic detection and exclusion of blank/corrupted patches and BRISQUE score-saturation cases to prevent skewed metrics.
* **Cross-Implementation Validation:** Anomalous outputs verified against both raw PyTorch models and the official wrapper before being cited as findings.
* **Exposure-Normalized Comparisons:** Brightness and contrast matched before visual judgment to avoid mistaking lighting differences for quality differences.

---

## ⚡ Performance & Scientific Findings

* **The Illusion of Quality:** Real-ESRGAN scores as "natural" as real, untouched photos on average (BRISQUE ~25–29 for both) across 75 valid patches.
* **Structural Hallucinations:** The model hallucinates. Round craters reconstruct as faceted polygons; smooth terrain gains fake scratch-like textures.
* **Metric Blind Spots:** BRISQUE sometimes scored the hallucinated output *better* than the real photo, penalizing genuine lunar regolith.
* **Content-Dependent Accuracy:** Geometric, repeating patterns (rover tracks) reconstruct far more faithfully than chaotic natural textures.
* **Deployment Speeds:** C++ deployment removes the Python dependency but does not change inference speed (~1.9s vs ~1.8s), confirming the heavy compute was always in the underlying compiled C math.

---

## 🧪 Development Practices

### **Methodology**

* **Skeptical Verification:** Every anomalous result was independently re-verified before being accepted as a genuine finding, rather than assumed.
* **Explicit Scoping:** Deliberate, staged scope decisions (e.g., minimal fine-tune, single-image SR) were explicitly documented rather than silently substituted.
* **Cross-Checking:** Visual and quantitative evidence were cross-checked against each other at every pipeline stage.

### **Reproducibility**

* All pipeline stages are isolated, independently runnable scripts with clear I/O.
* Model exports are numerically verified, not assumed correct.
* Data-cleaning thresholds and exclusions are heavily logged and explained.

---

## 🎯 Future Enhancements

* [ ] True dual-image fusion using both Left and Right NavCam stereo views
* [ ] Full fine-tuning on a larger lunar-imagery dataset with GPU compute
* [ ] BRISQUE (or equivalent) implemented natively in the C++ engine
* [ ] Evaluation extended to genuine orbital imagery (e.g., TMC-2)
* [ ] Seamless overlap-blended reconstruction as the default pipeline mode
* [ ] Comparison against a secondary no-reference metric (e.g., NIQE)

---

## 📊 Project Statistics

| Metric | Value |
| --- | --- |
| **Real Patches Processed** | 80 (75 valid after data-quality filtering) |
| **Stereo Pairs Used** | 5 matched Left/Right NavCam pairs |
| **ONNX Export Accuracy** | 8e-7 max pixel deviation vs. original model |
| **Avg BRISQUE (Real-ESRGAN)** | ~25–29 (Comparable to real photos) |
| **Avg BRISQUE (Bicubic)** | ~60–64 (Significantly worse) |
| **Deployment Target** | Standalone C++17 Binary |

---

## 👨‍💻 Developer

**Niranjan Reddy**

* 📧 Email: [niranjan024cmrit@gmail.com](https://www.google.com/search?q=mailto%3Aniranjan024cmrit%40gmail.com)
* 🐙 GitHub: [@Niranjan945](https://github.com/Niranjan945)
* 💼 LinkedIn: [Niranjan Profile](https://www.linkedin.com/in/avula-niranjan/)

---

## 🙏 Acknowledgments

* **[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)** (Wang et al.) - For the underlying open-source super-resolution architecture
* **[BRISQUE](https://ieeexplore.ieee.org/document/6272356)** (Mittal et al.) - No-reference image quality metric via OpenCV
* **ISRO PRADAN** - For providing public access to the Chandrayaan-3 mission data
* **Bharatiya Antariksh Hackathon** - For the original problem statement that inspired this research

---

## 🎯 Project Impact

> **Built to investigate not just whether AI super-resolution works, but whether it can be trusted.**

This project demonstrates end-to-end ownership of a real, unsolved problem—from acquiring genuine mission data, through building a working AI pipeline, to independently discovering, verifying, and quantifying a real limitation in a widely-used pretrained model.

**Key Competencies Demonstrated:**

* Full-stack ML engineering: data pipeline, model inference, evaluation, and deployment
* Cross-language systems work: Python research pipeline + standalone C++ production engine
* Rigorous, skeptical verification methodology—anomalies investigated, not assumed
* Scientific honesty in reporting both positive results and negative/limiting findings

---

*Investigating what it actually takes to trust an AI's guess, when there's no right answer to check against.*
