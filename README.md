# Sustainable Neuromorphic Edge Intelligence

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://img.shields.io/badge/paper-arXiv-red.svg)](YOUR_ARXIV_LINK)

> **Eco-Efficient Inference: Comparing Conventional vs Neuromorphic Hardware for Autonomous Driving**

This repository contains the complete experimental pipeline for comparing energy efficiency, accuracy, and eco-efficiency between conventional hardware (MacBook Pro M1) and neuromorphic hardware (Raspberry Pi 5 + BrainChip Akida v1.0 NPU) for CNN-based steering angle prediction in autonomous vehicles.

## 📄 Paper

**"Sustainable Neuromorphic Edge Intelligence for Autonomous Driving: A Comparative Eco-Efficiency Evaluation"**

*Fernando Sevilla Martínez, Jordi Casas-Roma, Laia Subirats, Raúl Parada*

Currently in Review in: eTransportation (2025)

[📖 Read the full paper soon] 

## 🎯 Key Findings

- **7.15× to 13.17× energy reduction** with neuromorphic hardware (615-1,217% savings)
- **3.4× to 7.3× throughput improvement** enabling real-time control at 233 Hz
- **Unexpected inverse-U quantization pattern**: lighter architectures (MiniNet) show best tolerance (51% MSE increase) while medium architectures (LaksNet) suffer most (113% increase)
- **8.5× eco-efficiency improvement** (Energy-Error Rate metric)
- **11.7× carbon footprint reduction** for edge AI deployment

## 📁 Repository Structure

```
EcoEdgeAI-akida-mac/
│
├── README.md                           # This file
├── LICENSE                             # MIT License
├── requirements_mac.txt                # Mac M1 dependencies
├── requirements_rpi5.txt               # Raspberry Pi 5 dependencies
│
├── 1_training/                         # Phase 1: Training on Mac M1
│   ├── train_float32_mac.py           # Main training script with CodeCarbon
│   ├── README.md                       # Training documentation
│   └── configs/
│       └── training_config.yaml        # Hyperparameters and settings
│
├── 2_quantization/                     # Phase 2: Quantization on RPi5
│   ├── quantize_and_convert_rpi5.py   # PTQ and QAT quantization + Akida conversion
│   ├── README.md                       # Quantization documentation
│   └── verify_akida_compatibility.py   # Architecture verification tool
│
├── 3_benchmarking/                     # Phase 3: Inference benchmarking
│   ├── unified_benchmark_mac.py        # Mac M1 benchmark (CodeCarbon)
│   ├── unified_benchmark_akida.py      # RPi5 + Akida benchmark (TC66)
│   ├── TC66C.py                        # TC66 USB power meter library
│   └── README.md                       # Benchmarking documentation
│
├── 4_analysis/                         # Phase 4: Results analysis
│   ├── compare_benchmarks.py           # Cross-platform comparison
│   ├── extract_mac_std.py              # Mac statistics extraction
│   ├── extract_akida_std.py            # Akida statistics extraction
│   ├── statistical_analysis_paper_v2.py # Statistical validation
│   ├── generate_all_figures_final.py   # Paper figures generation
│   └── README.md                       # Analysis documentation
│
├── scripts/                            # Utility scripts
│   ├── setup_mac.sh                    # Mac environment setup
│   ├── setup_rpi5.sh                   # RPi5 environment setup
│   ├── transfer_models.sh              # SCP transfer automation
│   ├── run_full_pipeline.sh            # End-to-end automation
│   └── README.md                       # Scripts documentation
│
├── docs/                               # Documentation
│   ├── WORKFLOW.md                     # Complete workflow guide
│   ├── HARDWARE_SETUP.md               # Hardware setup instructions
│   ├── TC66_GUIDE.md                   # Power meter usage guide
│   ├── TROUBLESHOOTING.md              # Common issues and solutions
│   ├── REPRODUCIBILITY.md              # Reproducibility checklist
│   └── CITATION.md                     # How to cite this work
│
├── data/                               # Dataset (not tracked)
│   ├── README.md                       # Dataset download instructions
│   └── .gitkeep
│
├── models/                             # Trained models (not tracked)
│   ├── README.md                       # Model format specifications
│   └── .gitkeep
│
├── results/                            # Experimental results (not tracked)
│   ├── mac/                            # Mac M1 benchmark results
│   ├── akida/                          # Akida NPU benchmark results
│   ├── comparison/                     # Cross-platform comparison
│   ├── figures/                        # Generated figures
│   └── README.md                       # Results structure
│
├── tests/                              # Unit tests
│   ├── test_models.py                  # Model architecture tests
│   ├── test_preprocessing.py           # Data pipeline tests
│   ├── test_quantization.py            # Quantization tests
│   └── README.md                       # Testing documentation
│
└── .github/                            # GitHub-specific files
    ├── workflows/
    │   ├── tests.yml                   # CI/CD for testing
    │   └── docs.yml                    # Documentation checks
    └── ISSUE_TEMPLATE/
        ├── bug_report.md
        └── feature_request.md
```

## 🚀 Quick Start

### Prerequisites

**Hardware:**
- Mac M1/M2/M3 (for training and conventional benchmarking)
- Raspberry Pi 5 with BrainChip Akida v1.0 NPU (for neuromorphic benchmarking)
- TC66/TC66C USB power meter (for hardware power measurement)

**Software:**
- Python 3.8+
- TensorFlow 2.15+ (Mac), 2.13+ (RPi5)
- CodeCarbon 3.0.7+
- Akida SDK 2.8.0+

### Installation

#### Mac M1 Setup

```bash
# Clone repository
git clone https://github.com/SevillaFe/EcoEdgeAI-akida-mac.git
cd EcoEdgeAI-akida-mac

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements_mac.txt

# Optional: Run setup script
bash scripts/setup_mac.sh
```

#### Raspberry Pi 5 Setup

```bash
# On RPi5
git clone https://github.com/SevillaFe/EcoEdgeAI-akida-mac.git
cd EcoEdgeAI-akida-mac

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies (includes Akida SDK)
pip install -r requirements_rpi5.txt

# Optional: Run setup script
bash scripts/setup_rpi5.sh
```

## 📊 Complete Workflow

### Phase 1: Training on Mac M1

Train all three CNN architectures (PilotNet, LaksNet, MiniNet) with energy tracking:

```bash
cd 1_training

# Train PilotNet
python train_float32_mac.py \
  --model pilotnet \
  --data_dir /path/to/Udacity_Dataset/Original_Images \
  --output_dir ../models/pilotnet \
  --result_dir ../results/training \
  --epochs 10 \
  --batch_size 32

# Train LaksNet
python train_float32_mac.py \
  --model laksnet \
  --data_dir /path/to/Udacity_Dataset/Original_Images \
  --output_dir ../models/laksnet \
  --result_dir ../results/training \
  --epochs 10 \
  --batch_size 32

# Train MiniNet
python train_float32_mac.py \
  --model mininet \
  --data_dir /path/to/Udacity_Dataset/Original_Images \
  --output_dir ../models/mininet \
  --result_dir ../results/training \
  --epochs 10 \
  --batch_size 32
```

**Outputs:**
- `{model}_float32.h5` - Trained float32 model
- `{model}_float32_history.csv` - Training metrics
- `training_emissions.csv` - Energy consumption and CO₂ emissions

### Phase 2: Transfer Models to RPi5

```bash
# Use automated transfer script
bash scripts/transfer_models.sh pi@raspberrypi.local

# Or manual transfer
scp models/pilotnet/pilotnet_float32.h5 pi@raspberrypi:~/models/
scp models/laksnet/laksnet_float32.h5 pi@raspberrypi:~/models/
scp models/mininet/mininet_float32.h5 pi@raspberrypi:~/models/
```

### Phase 3: Quantization and Conversion on RPi5

```bash
# SSH into RPi5
ssh pi@raspberrypi
cd EcoEdgeAI-akida-mac/2_quantization

# Quantize and convert to Akida format
python quantize_and_convert_rpi5.py \
  --model_path ../models/pilotnet_float32.h5 \
  --output_dir ../models/akida/pilotnet \
  --bits 4 \
  --qat_epochs 3 \
  --data_dir /path/to/Original_Images

# Repeat for other models...
```

**Outputs:**
- `{model}_q4_cnn2snn.h5` - PTQ quantized model
- `{model}_q4_qat_cnn2snn.h5` - QAT refined model (optional)
- `{model}_q4_akida.fbz` - Akida binary (PTQ)
- `{model}_qat_q4_akida.fbz` - Akida binary (QAT)

### Phase 4: Benchmarking

#### On RPi5 + Akida (with TC66 power meter)

```bash
cd ../3_benchmarking

# Benchmark Akida models
python unified_benchmark_akida.py \
  --akida_model ../models/akida/pilotnet/pilotnet_q4_akida.fbz \
  --data_dir /path/to/Original_Images \
  --output_dir ../results/akida/pilotnet \
  --num_samples 1000 \
  --tc66_port /dev/ttyACM0 \
  --measure_idle \
  --idle_duration 10

# Repeat for QAT and other models...
```

**Outputs:**
- `{model}_unified_benchmark_results.json` - Full results
- `{model}_unified_benchmark_summary.csv` - Summary statistics

#### On Mac M1 (with CodeCarbon)

```bash
# Back on Mac
cd 3_benchmarking

python unified_benchmark_mac.py \
  --model ../models/pilotnet/pilotnet_float32.h5 \
  --data_dir /path/to/Original_Images \
  --output_dir ../results/mac/pilotnet \
  --num_samples 1000 \
  --measure_idle \
  --idle_duration 10

# Repeat for other models...
```

### Phase 5: Analysis and Visualization

```bash
cd ../4_analysis

# Extract statistics
python extract_mac_std.py
python extract_akida_std.py

# Compare platforms
python compare_benchmarks.py \
  --mac_csv ../results/mac/pilotnet/pilotnet_float32_unified_benchmark_summary.csv \
  --akida_csv ../results/akida/pilotnet/pilotnet_q4_unified_benchmark_summary.csv \
  --output_dir ../results/comparison/pilotnet

# Generate figures
python generate_all_figures_final.py

# Statistical validation
python statistical_analysis_paper_v2.py
```

## 📈 Key Results

### Energy Efficiency

| Architecture | Mac M1 (mWh) | Akida (mWh) | Reduction | Savings |
|--------------|--------------|-------------|-----------|---------|
| PilotNet     | 35.3         | 4.9         | 7.15×     | 615%    |
| LaksNet      | 30.3         | 2.6         | 11.83×    | 1,083%  |
| MiniNet      | 33.7         | 2.6         | 13.17×    | 1,217%  |

### Accuracy Trade-offs

| Architecture | Mac MSE | Akida MSE | Degradation |
|--------------|---------|-----------|-------------|
| PilotNet     | 0.1138  | 0.2191    | +92.6%      |
| LaksNet      | 0.1187  | 0.2533    | +113.4%     |
| MiniNet      | 0.1193  | 0.1802    | +51.0%      |

### Eco-Efficiency (EER)

| Architecture | Mac EER | Akida EER | Improvement |
|--------------|---------|-----------|-------------|
| PilotNet     | 248,750 | 923,304   | 3.6×        |
| LaksNet      | 278,476 | 1,706,012 | 6.7×        |
| MiniNet      | 248,308 | 2,164,558 | **8.5×**    |

*EER = 1 / (MSE × Energy [kWh]) - Higher is better*

## 🔬 Reproducibility

All experiments use fixed random seeds (`seed=42`) for reproducibility. Complete details in:
- [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) - Full reproducibility checklist
- [WORKFLOW.md](docs/WORKFLOW.md) - Step-by-step workflow guide

### Checksum Verification

We provide MD5 checksums for all models demonstrating:
- QAT successfully modifies Keras weights (different .h5 checksums)
- Conversion produces byte-identical Akida binaries (identical .fbz checksums)
- Zero accuracy improvement from QAT (Section 4.3.4 in paper)

## 📚 Documentation

- [**WORKFLOW.md**](docs/WORKFLOW.md) - Complete experimental workflow
- [**HARDWARE_SETUP.md**](docs/HARDWARE_SETUP.md) - Hardware setup guide
- [**TC66_GUIDE.md**](docs/TC66_GUIDE.md) - Power meter usage
- [**TROUBLESHOOTING.md**](docs/TROUBLESHOOTING.md) - Common issues
- [**CITATION.md**](docs/CITATION.md) - How to cite this work

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@article{sevilla2025sustainable,
  title={Sustainable Neuromorphic Edge Intelligence for Autonomous Driving: A Comparative Eco-Efficiency Evaluation},
  author={Sevilla Mart{\'i}nez, Fernando and Casas-Roma, Jordi and Subirats, Laia and Parada, Ra{\'u}l},
  journal={[Journal Name]},
  year={2025},
  publisher={[Publisher]}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Udacity** for the Self-Driving Car Dataset
- **BrainChip** for Akida SDK and hardware access
- **CodeCarbon** team for energy tracking tools
- **Universitat Oberta de Catalunya (UOC)** for research support
- **Volkswagen AG** for industrial collaboration

## 📧 Contact

**Fernando Sevilla Martínez**
- Email: fsevillama@uoc.edu
- LinkedIn: [Your LinkedIn](YOUR_LINKEDIN)
- ResearchGate: [Your Profile](YOUR_RESEARCHGATE)

**Project Link:** [https://github.com/SevillaFe/EcoEdgeAI-akida-mac](https://github.com/SevillaFe/EcoEdgeAI-akida-mac)

## 🌟 Star History

If you find this work useful, please consider starring ⭐ the repository!

[![Star History Chart](https://api.star-history.com/svg?repos=SevillaFe/EcoEdgeAI-akida-mac&type=Date)](https://star-history.com/#SevillaFe/EcoEdgeAI-akida-mac&Date)

---

**Made for sustainable AI and neuromorphic computing**
