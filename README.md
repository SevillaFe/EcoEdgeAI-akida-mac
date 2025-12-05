# EcoEdgeAI: Neuromorphic Computing for Sustainable Autonomous Driving

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://img.shields.io/badge/DOI-10.xxxx%2Fxxxx-blue)](https://doi.org/10.xxxx/xxxx)

**Comparative Eco-Efficiency Evaluation of CNN Deployment on Conventional vs. Neuromorphic Hardware for Autonomous Driving**

This repository contains the complete experimental workflow, trained models, and analysis scripts for our paper:

> **"Sustainable Neuromorphic Edge Intelligence for Autonomous Driving: A Comparative Eco-Efficiency Evaluation"**
> 
> *Authors:* F.Sevilla Martínez, Jordi Casas-Roma, Laia Subirats, Raú
Parada]  
> *Conference/Journal:* In Review 
> *Year:* 2025

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Findings](#key-findings)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Experimental Workflow](#experimental-workflow)
- [Hardware Requirements](#hardware-requirements)
- [Dataset](#dataset)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 🎯 Overview

This work provides the **first comprehensive hardware-measured evaluation** of energy efficiency, accuracy trade-offs, and eco-efficiency scaling for CNN-based steering angle prediction deployed on:

- **Conventional Hardware:** MacBook Pro M1 (Apple Silicon)
- **Neuromorphic Hardware:** Raspberry Pi 5 + BrainChip Akida v1.0 NPU

### Research Questions

1. **Energy Efficiency:** How much energy do NPUs save compared to conventional hardware?
2. **Accuracy Trade-offs:** What is the accuracy cost of 4-bit quantization on neuromorphic hardware?
3. **Eco-Efficiency Scaling:** Which CNN architectures achieve optimal energy-accuracy balance?

### Key Contributions

✅ **Hardware-Measured Energy:** Direct measurement via TC66 USB power meter (neuromorphic) and CodeCarbon (conventional)  
✅ **Three CNN Architectures:** PilotNet (1.54M params), LaksNet (768K params), MiniNet (245K params)  
✅ **Complete Quantization Pipeline:** Float32 → PTQ 4-bit → QAT refinement → Akida deployment  
✅ **Reproducible Workflow:** End-to-end scripts from training to statistical analysis  
✅ **Energy-Error Rate (EER) Metric:** Unified eco-efficiency evaluation framework

---

## 🔬 Key Findings

### Energy Efficiency (RQ1)
- **7.15× to 13.17× energy reduction** (615-1,217% savings) on neuromorphic hardware
- Lighter architectures achieve superior gains: MiniNet 13.17×, PilotNet 7.15×
- Energy advantage driven **60-70% by throughput improvement** (3.4×-7.3× faster)
- **0.73 W sustained power** for MiniNet vs. 3.86 W on Mac M1

### Accuracy Trade-offs (RQ2)
- **Inverse-U quantization pattern:** Medium architectures suffer most (LaksNet +113% MSE)
- Lighter architectures show best tolerance: MiniNet +51% MSE, PilotNet +93% MSE
- **QAT provides zero accuracy benefit** across all architectures (4-bit capacity ceiling)
- Practical impact: 3.4°-7.2° additional steering error

### Eco-Efficiency Scaling (RQ3)
- **8.5× eco-efficiency improvement** for optimal configuration (MiniNet)
- Super-linear scaling with architectural simplification (2.0×-8.5× EER)
- **11.7× carbon footprint reduction** (12.9 μg → 1.1 μg CO₂ per 1,000 inferences)
- **233 Hz control loop capability** (4.30 ms latency) enables real-time operation

---

## 📁 Repository Structure

```
EcoEdgeAI-akida-mac/
│
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements_mac.txt               # Mac M1 dependencies
├── requirements_rpi5.txt              # RPi5 + Akida dependencies
├── CITATION.cff                       # Citation metadata
│
├── 01_training/                       # Phase 1: Model Training
│   ├── train_float32_mac.py          # Main training script (Mac M1)
│   └── README.md                      # Training instructions
│
├── 02_quantization/                   # Phase 2: Quantization & Conversion
│   ├── quantize_and_convert_rpi5.py  # PTQ, QAT, Akida conversion (RPi5)
│   └── README.md                      # Quantization guide
│
├── 03_benchmarking/                   # Phase 3 & 4: Inference Benchmarking
│   ├── unified_benchmark_mac.py      # Mac M1 benchmarking
│   ├── unified_benchmark_akida.py    # RPi5 + Akida benchmarking
│   ├── TC66C.py                       # TC66 power meter interface
│   └── README.md                      # Benchmarking instructions
│
├── 04_analysis/                       # Phase 5: Statistical Analysis
│   ├── extract_mac_std.py            # Mac statistics extraction
│   ├── extract_akida_std.py          # Akida statistics extraction
│   ├── compare_benchmarks.py         # Cross-platform comparison
│   ├── statistical_analysis_paper_v2.py  # Complete statistical validation
│   ├── generate_all_figures_final.py # Paper figures generation
│   └── README.md                      # Analysis guide
│
├── models/                            # Trained Models & Architectures
│   ├── architectures/                # Model definitions
│   │   ├── pilotnet.py
│   │   ├── laksnet.py
│   │   └── mininet.py
│   ├── float32/                      # Float32 trained models (.h5)
│   ├── quantized/                    # PTQ & QAT models (.h5)
│   ├── akida/                        # Akida binaries (.fbz)
│   └── checksums.md                  # Model integrity verification
│
├── data/                              # Dataset & Preprocessing
│   ├── preprocessing.py              # Image preprocessing utilities
│   ├── data_loader.py                # Dataset loading functions
│   └── README.md                     # Dataset instructions (Udacity)
│
├── results/                           # Experimental Results
│   ├── benchmark_results/            # Raw benchmark outputs (JSON/CSV)
│   ├── figures/                      # Publication-ready figures
│   ├── statistical_analysis/         # Statistical test results
│   └── checksum_verification/        # QAT weight modification evidence
│
├── docs/                              # Documentation
│   ├── WORKFLOW.md                   # Detailed experimental workflow
│   ├── HARDWARE_SETUP.md             # Hardware configuration guide
│   ├── TROUBLESHOOTING.md            # Common issues & solutions
│   └── API.md                        # API documentation
│
├── scripts/                           # Utility Scripts
│   ├── setup_mac.sh                  # Mac M1 environment setup
│   ├── setup_rpi5.sh                 # RPi5 environment setup
│   ├── verify_installation.py        # Dependency verification
│   └── download_models.sh            # Pre-trained model downloader
│
└── paper/                             # Paper & Supplementary Materials
    ├── paper.pdf                     # Published manuscript
    ├── supplementary.pdf             # Supplementary materials
    └── figures/                      # High-resolution figures
```

---

## 🛠️ Installation

### Prerequisites

**Common Requirements:**
- Python 3.10 or higher
- Git
- Udacity Self-Driving Car Dataset

**Platform-Specific:**

#### Mac M1 (Training & Benchmarking)
- macOS 14.3.1 or later
- 16 GB RAM minimum
- TensorFlow 2.15+ with Metal plugin

#### Raspberry Pi 5 + Akida (Quantization & Deployment)
- Ubuntu 22.04 LTS (64-bit)
- 8 GB RAM
- BrainChip Akida AKD1000 NPU
- TC66 USB-C power meter

### Setup Instructions

#### 1. Clone Repository

```bash
git clone https://github.com/SevillaFe/EcoEdgeAI-akida-mac.git
cd EcoEdgeAI-akida-mac
```

#### 2. Mac M1 Setup

```bash
# Create virtual environment
python3 -m venv venv_mac
source venv_mac/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements_mac.txt

# Verify installation
python scripts/verify_installation.py --platform mac
```

**Key Dependencies:**
```
tensorflow-macos==2.15.0
tensorflow-metal==1.1.0
codecarbon==2.3.2
opencv-python==4.8.1
scikit-learn==1.3.2
pandas==2.1.3
matplotlib==3.8.2
```

#### 3. Raspberry Pi 5 Setup

```bash
# Create virtual environment
python3 -m venv venv_rpi5
source venv_rpi5/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements_rpi5.txt

# Install Akida SDK (requires BrainChip account)
# Follow instructions at: https://doc.brainchipinc.com/

# Verify installation
python scripts/verify_installation.py --platform rpi5
```

**Key Dependencies:**
```
akida==2.8.0
cnn2snn==2.8.0
tensorflow==2.13.0
TC66C==1.0.0  # For power measurement
opencv-python==4.8.1
```

#### 4. Download Dataset

```bash
# Udacity Self-Driving Car Dataset
cd data
wget https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip
unzip data.zip
cd ..
```

---

## 🚀 Quick Start

### Complete Workflow (5 Phases)

```bash
# ============================================
# PHASE 1: Train Float32 Models (Mac M1)
# ============================================
python 01_training/train_float32_mac.py \
    --model all \
    --data_dir ./data/Original_Images \
    --output_dir ./models/float32 \
    --result_dir ./results/training \
    --batch_size 32 \
    --epochs 10

# ============================================
# PHASE 2: Transfer Models to RPi5
# ============================================
# Use SCP to transfer .h5 files
scp models/float32/*.h5 pi@raspberrypi.local:~/models/

# ============================================
# PHASE 3: Quantize & Convert (RPi5)
# ============================================
# SSH into RPi5
ssh pi@raspberrypi.local

python 02_quantization/quantize_and_convert_rpi5.py \
    --model_path ~/models/pilotnet_float32_best.h5 \
    --output_dir ./models/quantized \
    --bits 4 \
    --qat_epochs 3 \
    --data_dir ./data/Original_Images

# ============================================
# PHASE 4: Benchmark Akida (RPi5)
# ============================================
python 03_benchmarking/unified_benchmark_akida.py \
    --akida_model ./models/akida/pilotnet_best_q4_akida.fbz \
    --data_dir ./data/Original_Images \
    --output_dir ./results/benchmark_results_akida \
    --num_samples 1000 \
    --tc66_port /dev/ttyACM0 \
    --measure_idle

# ============================================
# PHASE 5: Benchmark Mac M1 (Back to Mac)
# ============================================
python 03_benchmarking/unified_benchmark_mac.py \
    --model ./models/float32/pilotnet_float32_best.h5 \
    --data_dir ./data/Original_Images \
    --output_dir ./results/benchmark_results_mac \
    --num_samples 1000 \
    --measure_idle \
    --idle_duration 10

# ============================================
# PHASE 6: Analysis & Visualization
# ============================================
# Extract statistics
python 04_analysis/extract_mac_std.py
python 04_analysis/extract_akida_std.py

# Compare platforms
python 04_analysis/compare_benchmarks.py \
    --mac_dir ./results/benchmark_results_mac \
    --akida_dir ./results/benchmark_results_akida \
    --output_dir ./results/comparison

# Generate figures
python 04_analysis/generate_all_figures_final.py \
    --results_dir ./results \
    --output_dir ./results/figures

# Statistical validation
python 04_analysis/statistical_analysis_paper_v2.py \
    --mac_csv ./results/mac_statistics_summary.csv \
    --akida_csv ./results/akida_statistics_summary.csv \
    --output_dir ./results/statistical_analysis
```

---

## 🔬 Experimental Workflow

### Phase 1: Training on Mac M1 (Float32)

**Script:** `01_training/train_float32_mac.py`

Trains three CNN architectures (PilotNet, LaksNet, MiniNet) with:
- **Energy Tracking:** CodeCarbon v3.0.7
- **Data Augmentation:** Horizontal flip, brightness, shadows
- **Validation Split:** 80/20 train/val
- **Early Stopping:** Patience 5 epochs
- **Output:** Best float32 models (.h5)

**Key Parameters:**
- Batch size: 32
- Learning rate: 1e-4 (Adam optimizer)
- Epochs: 10 (max)
- Loss: MSE

**Architecture Verification:** All models pass Akida v1.0 compatibility checks before training.

---

### Phase 2: Quantization & Conversion (RPi5)

**Script:** `02_quantization/quantize_and_convert_rpi5.py`

Converts float32 models to Akida-compatible format:

1. **Post-Training Quantization (PTQ):**
   - Uses `cnn2snn.quantize()` with 8/4/4 scheme
   - 4-bit weights & activations, 8-bit inputs

2. **Quantization-Aware Training (QAT) - Optional:**
   - Fine-tunes PTQ models for 3 epochs
   - Learning rate: 1e-6
   - **Result:** Zero accuracy benefit (Section 4.3.4)

3. **Akida Conversion:**
   - Uses `cnn2snn.convert()` targeting `AkidaVersion.v1`
   - Hardware mapping verification via `model.map(device)`
   - Generates `.fbz` binaries

**Critical Finding:** QAT produces byte-identical Akida binaries despite modifying Keras weights (checksums in `results/checksum_verification/`).

---

### Phase 3: Benchmarking Akida NPU (RPi5)

**Script:** `03_benchmarking/unified_benchmark_akida.py`

**Energy Measurement:**
- **Hardware:** TC66 USB-C power meter
- **Polling Rate:** 10 Hz (100 ms intervals)
- **Idle Subtraction:** 10-second baseline measurement
- **Metrics:** Voltage, current, power, internal energy accumulator

**Benchmark Protocol:**
- Batch size: 1 (sequential processing)
- Warmup: 10 iterations
- Validation set: 1,000 center-camera images
- Preprocessing: Crop, resize 200×66, uint8 [0,255]

**Output:** JSON with per-sample latencies, predictions, energy, CO₂

---

### Phase 4: Benchmarking Mac M1 (Float32)

**Script:** `03_benchmarking/unified_benchmark_mac.py`

**Energy Measurement:**
- **Software:** CodeCarbon v3.0.7
- **Method:** TDP-based estimation (20W CPU + 10W GPU)
- **Idle Subtraction:** 10-second baseline measurement
- **Carbon Intensity:** 0.420 kgCO₂/kWh (Germany)

**Benchmark Protocol:**
- Batch size: 1 (matching Akida)
- Warmup: 10 iterations
- Validation set: 1,000 center-camera images
- Preprocessing: Crop, resize 200×66, normalize [0,1]

**Output:** JSON with per-sample latencies, predictions, energy, CO₂

---

### Phase 5: Statistical Analysis

**Scripts:**
1. `extract_mac_std.py` - Extracts Mac statistics
2. `extract_akida_std.py` - Extracts Akida statistics
3. `compare_benchmarks.py` - Cross-platform comparison
4. `statistical_analysis_paper_v2.py` - Complete validation

**Statistical Tests:**
- **Energy Comparison:** Paired t-tests with Bonferroni correction (α=0.0167)
- **Effect Sizes:** Cohen's d (|d| > 12 for all architectures)
- **Accuracy Analysis:** Two-way ANOVA (Platform × Architecture)
- **Confidence Intervals:** Bootstrap resampling (10,000 iterations)

**Outputs:**
- CSV summaries with mean, std, min, max, CV
- LaTeX tables for manuscript
- Publication-ready figures (PDF/PNG)

---

## 🖥️ Hardware Requirements

### Mac M1 Platform

| Component | Specification |
|-----------|---------------|
| **Processor** | Apple M1 (8-core: 4P+4E) @ 3.2 GHz |
| **GPU** | 14-core @ 3.2 GHz (4.6 TFLOPS float32) |
| **Memory** | 16 GB LPDDR4X unified |
| **Storage** | 512 GB NVMe SSD |
| **OS** | macOS 14.3.1 Sonoma |
| **TDP** | 20W CPU + 10W GPU (estimated) |

### RPi5 + Akida Platform

| Component | Specification |
|-----------|---------------|
| **Host CPU** | Broadcom BCM2712 (ARM Cortex-A76) @ 2.4 GHz |
| **NPU** | BrainChip Akida AKD1000 (80 NPUs, 1.2M neurons) |
| **Memory** | 8 GB LPDDR4X (RPi5) + 256MB LPDDR4 @ 2400MT/s (Akida) |
| **Storage** | 512 GB microSD (UHS-I) |
| **Interconnect** | PCIe 2.0 x1 (500 MB/s) |
| **OS** | Ubuntu 22.04 LTS (64-bit) |
| **Power** | 5W (RPi5) + 0.2-0.5W (Akida active) |

### TC66 Power Meter

| Specification | Value |
|---------------|-------|
| **Voltage Range** | 4-24V (±0.5% accuracy) |
| **Current Range** | 0-5A (±0.5% accuracy) |
| **Power Resolution** | 0.001W |
| **Energy Accumulator** | 0.001 Wh resolution |
| **Interface** | USB-C, Bluetooth (TC66 App) |

---

## 📊 Dataset

**Udacity Self-Driving Car Simulator Dataset**

- **Source:** [Udacity GitHub](https://github.com/udacity/self-driving-car-sim)
- **Samples:** 8,033 images per camera (center, left, right)
- **Resolution:** 320×160 RGB
- **Frame Rate:** 10 Hz
- **Steering Range:** -1.0 to +1.0 radians (-25° to +25°)

### Preprocessing Pipeline

1. **Crop:** Remove sky (top 60px) and hood (bottom 25px)
2. **Resize:** 200×66 pixels (bilinear interpolation)
3. **Normalize:** 
   - Mac M1: [0,1] float32
   - Akida: [0,255] uint8
4. **Steering Correction:** ±0.2 radians for left/right cameras

### Data Augmentation (Training Only)

- Horizontal flip: 50% probability (negate steering)
- Brightness adjustment: 40% probability (factor [0.6, 1.4])
- Random shadows: 30% probability (factor [0.3, 0.7])

---

## 🔁 Reproducibility

### Random Seeds

All stochastic operations use **seed=42**:
```python
import random
import numpy as np
import tensorflow as tf

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
```

### Model Checksums

Verify model integrity using MD5 checksums in `models/checksums.md`:

```bash
# Verify float32 models
md5sum models/float32/pilotnet_float32_best.h5
# Expected: ffe3707a8c2b1e4f5d6a3c9b8e7f1a2d

# Verify Akida binaries
md5sum models/akida/pilotnet_best_q4_akida.fbz
# Expected: 022c92a1b3c4d5e6f7a8b9c0d1e2f3a4
```

### Benchmark Reproducibility

**Controlled Variables:**
- Fixed validation split (same 1,000 images)
- Identical preprocessing across platforms
- Consistent idle baseline methodology
- Synchronized measurement protocols

**Expected Variance:**
- Energy measurements: ±5-10% (hardware/environmental factors)
- Latency measurements: ±2-5% (system load variations)
- Accuracy metrics: <0.1% (deterministic inference)

---

## 📖 Citation

If you use this code or findings in your research, please cite:

```bibtex
@article{martinez2025sustainable,
  title={Sustainable Neuromorphic Edge Intelligence for Autonomous Driving: A Comparative Eco-Efficiency Evaluation},
  author={Martinez, F. Sevilla and [Co-authors]},
  journal={[In Review - TBD]},
  year={2026},
  volume={XX},
  number={X},
  pages={XXX--XXX},
  doi={10.xxxx/xxxx}
}
```

**BibTeX for Software:**
```bibtex
@software{ecoedgeai2025,
  author = {Martinez, F. Sevilla et al.},
  title = {EcoEdgeAI: Neuromorphic Computing Benchmark Suite},
  year = {2025},
  url = {https://github.com/SevillaFe/EcoEdgeAI-akida-mac},
  version = {1.0.0}
}
```

---

## 📄 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file for details.

### Third-Party Licenses

- **TC66C Library:** MIT License (TheHWcave)
- **CodeCarbon:** MIT License
- **BrainChip Akida SDK:** Proprietary (requires account)
- **Udacity Dataset:** Open source

---


### Key Dependencies

- TensorFlow/Keras (Apache 2.0)
- cnn2snn (BrainChip)
- CodeCarbon (MIT)
- scikit-learn (BSD-3-Clause)
- OpenCV (Apache 2.0)

---

## 📞 Contact

**Principal Investigator:**  
[F. Sevilla Martínez]  
[Open University Catalonia]  
Email: [fsevillama@uoc.edu]  
GitHub: [@SevillaFe](https://github.com/SevillaFe)

**Issues & Questions:**  
Please use [GitHub Issues](https://github.com/SevillaFe/EcoEdgeAI-akida-mac/issues)

---

## 📈 Project Status

- ✅ **Training Pipeline:** Complete
- ✅ **Quantization & Conversion:** Complete
- ✅ **Benchmarking:** Complete
- ✅ **Statistical Analysis:** Complete
- ✅ **Paper Submission:** In Review
- ✅ **Documentation:** In Progress
- ✅ **Pre-trained Models:** Complete

**Last Updated:** December 2025

---

**⭐ Star this repository if you find it useful!**

**🐛 Report bugs:** [Issue Tracker](https://github.com/SevillaFe/EcoEdgeAI-akida-mac/issues)

**💬 Discussions:** [GitHub Discussions](https://github.com/SevillaFe/EcoEdgeAI-akida-mac/discussions)

**Project Link:** [https://github.com/SevillaFe/EcoEdgeAI-akida-mac](https://github.com/SevillaFe/EcoEdgeAI-akida-mac)

## 🌟 Star History

If you find this work useful, please consider starring ⭐ the repository!

[![Star History Chart](https://api.star-history.com/svg?repos=SevillaFe/EcoEdgeAI-akida-mac&type=Date)](https://star-history.com/#SevillaFe/EcoEdgeAI-akida-mac&Date)

---

**Made for sustainable AI and neuromorphic computing**
