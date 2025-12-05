# Phase 2: Quantization & Conversion

This directory contains scripts for quantizing float32 models and converting to Akida format.

## Prerequisites

- Raspberry Pi 5 with Akida AKD1000 NPU
- Akida SDK 2.8.0 installed
- Float32 models transferred from Mac

## Usage

### Post-Training Quantization (PTQ) Only

```bash
python quantize_and_convert_rpi5.py \
    --model_path ../models/float32/pilotnet_float32_best.h5 \
    --output_dir ../models/quantized \
    --bits 4
```

### PTQ + Quantization-Aware Training (QAT)

```bash
python quantize_and_convert_rpi5.py \
    --model_path ../models/float32/pilotnet_float32_best.h5 \
    --output_dir ../models/quantized \
    --bits 4 \
    --qat_epochs 3 \
    --qat_lr 1e-6 \
    --data_dir ../data/Original_Images
```

## Arguments

- `--model_path`: Path to float32 .h5 model
- `--output_dir`: Output directory for quantized models
- `--bits`: Quantization bit-width (default: 4)
- `--qat_epochs`: QAT epochs (0=disabled, default: 0)
- `--qat_lr`: QAT learning rate (default: 1e-6)
- `--data_dir`: Dataset path (required if QAT enabled)

## Outputs

### PTQ Pipeline
1. `{model}_q4_cnn2snn.h5` - Quantized Keras model
2. `{model}_q4_akida.fbz` - Akida binary
3. Hardware mapping verification

### QAT Pipeline (if enabled)
1. `{model}_q4_qat_cnn2snn.h5` - QAT-refined Keras model
2. `{model}_qat_q4_akida.fbz` - Akida binary from QAT
3. Training history CSV

## Critical Finding

**QAT produces byte-identical Akida binaries despite modifying Keras weights.**

Verification:
```bash
md5sum {model}_q4_akida.fbz {model}_qat_q4_akida.fbz
# Both hashes are identical
```

This indicates a 4-bit quantization capacity ceiling for regression tasks.

## Hardware Mapping

All models undergo verification:
```python
model.map(device)  # Must succeed, or CPU emulation fallback
```

Failed mapping indicates architectural incompatibility with Akida v1.0.
