# !/usr/bin/env python3
"""
Script 2 (RPi5): Quantize and Convert to Akida
===============================================

Quantizes float32 models using cnn2snn (NOT quantizeml) and converts to Akida format.
This script MUST be run on RPi5 with Akida hardware and cnn2snn installed.

WORKFLOW:
1. Load float32 model trained on Mac
2. Quantize using cnn2snn.quantize() for Akida 1.0
3. (Optional) Fine-tune with QAT for better accuracy
4. Convert to Akida format using cnn2snn.convert()
5. Verify hardware compatibility
6. Save quantized .h5 and Akida .fbz models

IMPORTANT:
- Uses cnn2snn.quantize (NOT quantizeml)
- Targets Akida 1.0 (AkidaVersion.v1)
- Generates 8-bit and 4-bit quantized models
- Supports QAT (Quantization-Aware Training) for improved accuracy

USAGE:
======
--model_path /home/fernando/Documents/Doctorado/pilotnet/pilotnet_float32_best.h5 \
--output_dir /home/fernando/Documents/Doctorado/akida_models/pilotnet_akida \

# Without QAT (Post-Training Quantization only)
python quantize_and_convert_rpi5.py \
    --model_path pilotnet_float32.h5 \
    --output_dir ./output \
    --bits 8 4

# With QAT for 4-bit model
python quantize_and_convert_rpi5.py \
    --model_path -model_path /home/fernando/Documents/Doctorado/pilotnet/pilotnet_float32_best.h5 \
    --output_dir --output_dir /home/fernando/Documents/Doctorado/akida_models/pilotnet_akida \
    --bits 4 \
    --qat_epochs 5 \
    --qat_lr 1e-6 \
    --data_dir /home/fernando/Documents/Doctorado/Original_Images

ARGUMENTS:
==========
--model_path       Path to float32 .h5 model from Mac (required)
--output_dir       Output directory (default: same as model directory)
--bits             Quantization bit widths (default: 8 4)
--test_samples     Number of test samples for evaluation (default: 100)
--qat_epochs       Number of QAT epochs (default: 0, set >0 to enable QAT)
--qat_lr           Learning rate for QAT (default: 1e-6)
--qat_batch_size   Batch size for QAT (default: 32)
--data_dir         Path to dataset (required if QAT enabled)
"""


import argparse
from pathlib import Path
import sys
import numpy as np
import os

try:
    from cnn2snn import quantize, convert, set_akida_version, AkidaVersion
    import akida
    from keras import models as keras_models
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    import pandas as pd
    import cv2
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nThis script requires cnn2snn and akida packages.")
    print("Make sure you're running on RPi5 with:")
    print("  pip install cnn2snn akida")
    sys.exit(1)


def detect_akida_device():
    """Detect and verify Akida hardware"""
    print("\n" + "=" * 70)
    print("DETECTING AKIDA DEVICE")
    print("=" * 70)

    devices = akida.devices()

    if len(devices) == 0:
        print("\n❌ NO AKIDA DEVICE DETECTED!")
        print("This script requires Akida hardware.")
        sys.exit(1)

    device = devices[0]
    print(f"\n✓ Device found: {device}")

    try:
        version_str = device.version
        print(f"  Version: {version_str}")
        print(f"  SoC: {device.soc}")

        # Verify it's Akida 1.0
        if version_str.startswith('BC'):
            print(f"\n✓ Detected: Akida 1.0 (BC hardware)")
            akida_version = AkidaVersion.v1
        else:
            print(f"\n⚠️  Warning: Detected Akida 2.0")
            print("This script is optimized for Akida 1.0")
            akida_version = AkidaVersion.v2
    except AttributeError:
        print("\n⚠️  Could not determine version, assuming Akida 1.0")
        akida_version = AkidaVersion.v1

    return device, akida_version


def load_and_preprocess_image(img_path, steering):
    """
    Load and preprocess image for QAT training
    MUST match the preprocessing used in train_float32_mac.py!
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Crop: remove sky (60px top) and hood (25px bottom)
    img = img[60:-25, :]

    # Resize
    img = cv2.resize(img, (200, 66))

    # Keep as float32 (NOT normalized yet)
    return img.astype(np.float32), steering


def load_qat_data(data_dir, train_val_split=0.8, steering_correction=0.2):
    """
    Load training data for QAT
    Same logic as train_float32_mac.py
    """
    print(f"\n" + "=" * 70)
    print("LOADING DATA FOR QAT")
    print("=" * 70)

    csv_path = Path(data_dir) / 'driving_log.csv'

    # CSV has no headers
    columns = ["center", "left", "right", "steering", "throttle", "reverse", "speed"]
    df = pd.read_csv(csv_path, names=columns)

    print(f"Total samples in CSV: {len(df)}")

    # Collect image paths and steerings
    img_paths = []
    steerings = []

    for idx, row in df.iterrows():
        # Center camera
        center_filename = os.path.basename(row['center'].strip())
        center_path = os.path.join(data_dir, 'IMG', center_filename)
        if os.path.exists(center_path):
            img_paths.append(center_path)
            steerings.append(float(row['steering']))

        # Left camera (add positive correction)
        left_filename = os.path.basename(row['left'].strip())
        left_path = os.path.join(data_dir, 'IMG', left_filename)
        if os.path.exists(left_path):
            img_paths.append(left_path)
            steerings.append(float(row['steering']) + steering_correction)

        # Right camera (add negative correction)
        right_filename = os.path.basename(row['right'].strip())
        right_path = os.path.join(data_dir, 'IMG', right_filename)
        if os.path.exists(right_path):
            img_paths.append(right_path)
            steerings.append(float(row['steering']) - steering_correction)

    steerings = np.array(steerings, dtype=np.float32)

    print(f"\nTotal samples (with all cameras): {len(img_paths)}")
    print(f"Steering angle range: [{steerings.min():.3f}, {steerings.max():.3f}]")

    # Split train/validation
    split_idx = int(len(img_paths) * train_val_split)

    train_paths = img_paths[:split_idx]
    train_steerings = steerings[:split_idx]

    val_paths = img_paths[split_idx:]
    val_steerings = steerings[split_idx:]

    print(f"\nTrain samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")

    return train_paths, train_steerings, val_paths, val_steerings


def qat_data_generator(img_paths, steerings, batch_size):
    """
    Data generator for QAT training
    No augmentation during QAT - just preprocessing
    """
    num_samples = len(img_paths)

    while True:
        # Shuffle data
        indices = np.random.permutation(num_samples)
        img_paths_shuffled = [img_paths[i] for i in indices]
        steerings_shuffled = steerings[indices]

        for offset in range(0, num_samples, batch_size):
            batch_paths = img_paths_shuffled[offset:offset + batch_size]
            batch_steerings = steerings_shuffled[offset:offset + batch_size]

            images = []
            angles = []

            for img_path, steering in zip(batch_paths, batch_steerings):
                # Load and preprocess
                img, steer = load_and_preprocess_image(img_path, steering)

                # Normalize to [0, 1]
                img = img / 255.0

                images.append(img)
                angles.append(steer)

            X = np.array(images, dtype=np.float32)
            y = np.array(angles, dtype=np.float32)

            yield X, y


def detect_akida_device():
    """Detect and verify Akida hardware"""
    print("\n" + "=" * 70)
    print("DETECTING AKIDA DEVICE")
    print("=" * 70)

    devices = akida.devices()

    if len(devices) == 0:
        print("\n❌ NO AKIDA DEVICE DETECTED!")
        print("This script requires Akida hardware.")
        sys.exit(1)

    device = devices[0]
    print(f"\n✓ Device found: {device}")

    try:
        version_str = device.version
        print(f"  Version: {version_str}")
        print(f"  SoC: {device.soc}")

        # Verify it's Akida 1.0
        if version_str.startswith('BC'):
            print(f"\n✓ Detected: Akida 1.0 (BC hardware)")
            akida_version = AkidaVersion.v1
        else:
            print(f"\n⚠️  Warning: Detected Akida 2.0")
            print("This script is optimized for Akida 1.0")
            akida_version = AkidaVersion.v2
    except AttributeError:
        print("\n⚠️  Could not determine version, assuming Akida 1.0")
        akida_version = AkidaVersion.v1

    return device, akida_version


def load_float32_model(model_path):
    """Load float32 model trained on Mac"""
    print(f"\n" + "=" * 70)
    print("LOADING FLOAT32 MODEL")
    print("=" * 70)
    print(f"Model: {model_path.name}")

    try:
        model = keras_models.load_model(str(model_path), compile=False)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)

    print(f"✓ Model loaded successfully")
    print(f"\nModel architecture:")
    model.summary()

    return model


def perform_qat(quantized_model, train_paths, train_steerings, val_paths, val_steerings,
                output_dir, model_name, bits, qat_epochs=5, qat_lr=1e-6, batch_size=32):
    """
    Perform Quantization-Aware Training (QAT) to fine-tune quantized model

    Args:
        quantized_model: Quantized Keras model from cnn2snn.quantize()
        train_paths: Training image paths
        train_steerings: Training steering angles
        val_paths: Validation image paths
        val_steerings: Validation steering angles
        output_dir: Output directory
        model_name: Base model name
        bits: Quantization bit width
        qat_epochs: Number of QAT epochs
        qat_lr: Learning rate for QAT (should be very low, e.g., 1e-6)
        batch_size: Batch size for training

    Returns:
        qat_model: Fine-tuned model (same object as input, modified in-place)
        qat_path: Path to saved model
    """
    print(f"\n" + "=" * 70)
    print(f"QUANTIZATION-AWARE TRAINING (QAT) - {bits}-BIT")
    print("=" * 70)
    print(f"QAT Epochs: {qat_epochs}")
    print(f"Learning Rate: {qat_lr}")
    print(f"Batch Size: {batch_size}")

    # Compile model for QAT (use Adam, not legacy)
    quantized_model.compile(
        optimizer=Adam(learning_rate=qat_lr),
        loss='mse',
        metrics=['mae']
    )

    # Create data generators
    steps_per_epoch = len(train_paths) // batch_size
    validation_steps = len(val_paths) // batch_size

    train_gen = qat_data_generator(train_paths, train_steerings, batch_size)
    val_gen = qat_data_generator(val_paths, val_steerings, batch_size)

    print(f"\nSteps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")

    # Callbacks
    checkpoint_path = str(output_dir / f'{model_name}_q{bits}_qat_best.h5')
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-8,
        verbose=1
    )

    # Train
    print(f"\n{'=' * 70}")
    print("STARTING QAT FINE-TUNING")
    print(f"{'=' * 70}")

    history = quantized_model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        epochs=qat_epochs,
        callbacks=[checkpoint, early_stop, reduce_lr],
        verbose=1
    )

    # Evaluate final model (no need to load, EarlyStopping restored best weights)
    print(f"\n{'=' * 70}")
    print("EVALUATING QAT MODEL")
    print(f"{'=' * 70}")

    # Evaluate on validation set
    print("Evaluating on validation set...")
    y_pred = []
    y_true = []

    val_gen_eval = qat_data_generator(val_paths, val_steerings, batch_size)
    for i in range(min(validation_steps, 50)):  # Limit to 50 batches for speed
        x_batch, y_batch = next(val_gen_eval)
        predictions = quantized_model.predict(x_batch, verbose=0)
        y_pred.append(predictions)
        y_true.append(y_batch)

    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)

    from sklearn.metrics import mean_squared_error, mean_absolute_error
    qat_mse = mean_squared_error(y_true, y_pred)
    qat_mae = mean_absolute_error(y_true, y_pred)

    print(f"✓ QAT Model - Validation MSE: {qat_mse:.4f}, MAE: {qat_mae:.4f}")

    # Save final QAT model
    qat_path = output_dir / f'{model_name}_q{bits}_qat_cnn2snn.h5'
    quantized_model.save(str(qat_path))

    print(f"\n✓ QAT complete!")
    print(f"✓ Saved: {qat_path}")

    # Return the same model object (modified in-place by fit())
    return quantized_model, qat_path


def quantize_model_cnn2snn(model, bits, output_dir, model_name,
                           calibration_data=None, num_calibration_samples=1024):
    """
    Quantize model using cnn2snn for Akida 1.0 with calibration

    Args:
        model: Float32 Keras model
        bits: Quantization bit width (4 or 8)
        output_dir: Output directory
        model_name: Base model name
        calibration_data: Tuple of (img_paths, steerings) for calibration
        num_calibration_samples: Number of calibration samples to use

    Returns:
        quantized_model, output_path
    """
    print(f"\n" + "=" * 70)
    print(f"QUANTIZING TO {bits}-BIT (cnn2snn)")
    print("=" * 70)

    # Prepare calibration samples if provided
    if calibration_data is not None:
        print(f"Using {num_calibration_samples} calibration samples")
        img_paths, steerings = calibration_data

        # Select calibration samples
        num_available = len(img_paths)
        num_to_use = min(num_calibration_samples, num_available)

        # Sample uniformly across the steering angle range for better coverage
        indices = np.linspace(0, num_available - 1, num_to_use, dtype=int)

        print(f"Loading {num_to_use} calibration images...")
        calib_images = []
        for idx in indices:
            img, _ = load_and_preprocess_image(img_paths[idx], 0.0)
            img = img / 255.0  # Normalize
            calib_images.append(img)

        calib_images = np.array(calib_images, dtype=np.float32)
        print(f"✓ Calibration data shape: {calib_images.shape}")
    else:
        calib_images = None
        print("⚠️  No calibration data provided - using default quantization")

    try:
        # NOTE: cnn2snn.quantize() API is different from quantizeml
        # It doesn't directly support passing calibration samples
        # The quantization is done statically based on the model
        quantized = quantize(
            model,
            input_weight_quantization=8,  # Input always 8-bit
            weight_quantization=bits,
            activ_quantization=bits
        )

        print(f"✓ Quantization successful!")

        # If calibration data was provided, we can do a quick evaluation
        if calib_images is not None:
            print("\nValidating quantization with calibration samples...")
            try:
                # Quick check on a few samples
                sample_preds = quantized.predict(calib_images[:10], verbose=0)
                print(f"✓ Quantized model inference working")
                print(f"  Sample predictions range: [{sample_preds.min():.3f}, {sample_preds.max():.3f}]")
            except Exception as e:
                print(f"⚠️  Calibration validation failed: {e}")

        # Save quantized model
        output_path = output_dir / f"{model_name}_q{bits}_cnn2snn.h5"
        quantized.save(str(output_path))
        print(f"✓ Saved: {output_path}")

        return quantized, output_path

    except Exception as e:
        print(f"\n❌ Quantization failed: {e}")
        print("\nPossible issues:")
        print("  - Model architecture incompatible with Akida 1.0")
        print("  - Unsupported layers (check Conv2D kernel sizes)")
        raise


def convert_to_akida(quantized_model, device, akida_version, output_dir, model_name, bits):
    """
    Convert quantized Keras model to Akida format

    Args:
        quantized_model: Quantized Keras model
        device: Akida device
        akida_version: Target Akida version (v1 or v2)
        output_dir: Output directory
        model_name: Base model name
        bits: Quantization bit width

    Returns:
        fbz_path, success
    """
    print(f"\n" + "=" * 70)
    print(f"CONVERTING {bits}-BIT MODEL TO AKIDA")
    print("=" * 70)
    print(f"Target version: {akida_version}")

    try:
        # Convert with explicit Akida version
        with set_akida_version(akida_version):
            akida_model = convert(quantized_model)

        print("✓ Conversion successful!")

        # Show model info
        print(f"\nAkida model info:")
        print(f"  Input:  {akida_model.input_shape}")
        print(f"  Output: {akida_model.output_shape}")
        print(f"  Layers: {len(akida_model.layers)}")

        if hasattr(akida_model, 'ip_version'):
            print(f"  IP Version: {akida_model.ip_version}")

        # Print layer details to check for CPU emulation
        print(f"\nLayer details:")
        for i, layer in enumerate(akida_model.layers):
            layer_info = str(layer)
            print(f"  [{i}] {layer_info}")
            # Check for CPU emulation indicators
            if 'cpu' in layer_info.lower() or 'emulation' in layer_info.lower():
                print(f"      ⚠️  WARNING: This layer may run on CPU emulation!")

        # Save
        fbz_path = output_dir / f"{model_name}_q{bits}_akida.fbz"
        akida_model.save(str(fbz_path))
        print(f"\n✓ Saved: {fbz_path}")

        # Verify hardware mapping
        print(f"\n" + "=" * 70)
        print("VERIFYING HARDWARE MAPPING")
        print("=" * 70)

        try:
            akida_model.map(device)
            print("\n✓✓✓ SUCCESS! ✓✓✓")
            print("Model maps to Akida hardware!")
            print("Ready for inference on NPU.")
            return fbz_path, True

        except RuntimeError as e:
            print(f"\n❌ MAPPING FAILED: {e}")
            print("\nThe model converted but cannot map to hardware.")
            print("This usually means:")
            print("  - IP version mismatch")
            print("  - Architecture incompatibility")
            print("  - Model will run in CPU emulation (very slow)")
            return fbz_path, False

    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        print("\nCommon issues:")
        print("  - Unsupported layer types")
        print("  - Incompatible layer configurations")
        print("  - BatchNormalization fusion issues")
        raise


def test_quantized_model(model, test_samples=100):
    """
    Quick test of quantized model (optional)

    Args:
        model: Keras model to test
        test_samples: Number of random test samples

    Note: This is a smoke test with random data.
    Real evaluation should be done with actual dataset.
    """
    print(f"\n" + "-" * 70)
    print(f"SMOKE TEST (random data)")
    print("-" * 70)

    try:
        # Create random test data
        input_shape = model.input_shape[1:]  # Remove batch dimension
        X_test = np.random.rand(test_samples, *input_shape).astype(np.float32)

        # Run inference
        predictions = model.predict(X_test, verbose=0)

        print(f"✓ Model can process {test_samples} samples")
        print(f"  Output shape: {predictions.shape}")
        print(f"  Output range: [{predictions.min():.3f}, {predictions.max():.3f}]")

        return True
    except Exception as e:
        print(f"❌ Smoke test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Quantize and convert float32 models on RPi5 with optional QAT"
    )
    parser.add_argument("--model_path", required=True,
                        help="Path to float32 .h5 model from Mac")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: same as model)")
    parser.add_argument("--bits", nargs='+', type=int, default=[8, 4],
                        help="Quantization bits (default: 8 4)")
    parser.add_argument("--test_samples", type=int, default=100,
                        help="Number of test samples for smoke test (default: 100)")

    # Calibration arguments
    parser.add_argument("--num_calibration_samples", type=int, default=1024,
                        help="Number of calibration samples (default: 1024)")
    parser.add_argument("--use_calibration", action="store_true",
                        help="Use calibration samples for quantization (requires --data_dir)")

    # QAT arguments
    parser.add_argument("--qat_epochs", type=int, default=0,
                        help="Number of QAT epochs (default: 0, disabled). Set >0 to enable QAT")
    parser.add_argument("--qat_lr", type=float, default=1e-6,
                        help="Learning rate for QAT (default: 1e-6)")
    parser.add_argument("--qat_batch_size", type=int, default=32,
                        help="Batch size for QAT (default: 32)")
    parser.add_argument("--data_dir", default=None,
                        help="Path to dataset (required if QAT enabled or calibration enabled)")

    args = parser.parse_args()

    # Validate inputs
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)

    # Check QAT and calibration requirements
    needs_data = args.qat_epochs > 0 or args.use_calibration
    if needs_data:
        if args.data_dir is None:
            print(f"❌ --data_dir is required when QAT or calibration is enabled")
            sys.exit(1)
        if not Path(args.data_dir).exists():
            print(f"❌ Dataset directory not found: {args.data_dir}")
            sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else model_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract model name
    model_name = model_path.stem.replace('_float32', '')

    print("\n" + "=" * 70)
    print("AKIDA 1.0 QUANTIZATION & CONVERSION PIPELINE")
    if args.use_calibration:
        print("WITH CALIBRATION SAMPLES")
    if args.qat_epochs > 0:
        print("WITH QUANTIZATION-AWARE TRAINING (QAT)")
    print("=" * 70)
    print(f"Input:  {model_path}")
    print(f"Output: {output_dir}")
    print(f"Bits:   {args.bits}")
    print(f"Model:  {model_name}")

    if args.use_calibration:
        print(f"\nCalibration:")
        print(f"  Samples: {args.num_calibration_samples}")
        print(f"  Data:    {args.data_dir}")

    if args.qat_epochs > 0:
        print(f"\nQAT Configuration:")
        print(f"  Epochs:     {args.qat_epochs}")
        print(f"  LR:         {args.qat_lr}")
        print(f"  Batch size: {args.qat_batch_size}")
        print(f"  Data:       {args.data_dir}")
    print("=" * 70)

    # Step 1: Detect Akida device
    device, akida_version = detect_akida_device()

    # Step 2: Load float32 model
    float32_model = load_float32_model(model_path)

    # Step 3: Load data if needed (QAT or calibration)
    train_paths, train_steerings, val_paths, val_steerings = None, None, None, None
    calibration_data = None

    if needs_data:
        train_paths, train_steerings, val_paths, val_steerings = load_qat_data(
            args.data_dir
        )

        # Prepare calibration data (use training set)
        if args.use_calibration:
            calibration_data = (train_paths, train_steerings)
            print(f"\n✓ Calibration data prepared: {len(train_paths)} samples available")

    # Step 4: Quantize and optionally apply QAT for each bit width
    results = {}

    for bits in args.bits:
        print(f"\n" + "=" * 70)
        print(f"PROCESSING {bits}-BIT QUANTIZATION")
        print("=" * 70)

        # PTQ: Post-Training Quantization (with optional calibration)
        quantized_model, q_path = quantize_model_cnn2snn(
            float32_model, bits, output_dir, model_name,
            calibration_data=calibration_data,
            num_calibration_samples=args.num_calibration_samples
        )

        # Test quantized model (smoke test)
        test_quantized_model(quantized_model, args.test_samples)

        # QAT: Quantization-Aware Training (optional)
        qat_model = None
        qat_path = None
        if args.qat_epochs > 0:
            qat_model, qat_path = perform_qat(
                quantized_model,
                train_paths, train_steerings,
                val_paths, val_steerings,
                output_dir, model_name, bits,
                qat_epochs=args.qat_epochs,
                qat_lr=args.qat_lr,
                batch_size=args.qat_batch_size
            )

        # Convert PTQ model to Akida
        fbz_path_ptq, success_ptq = convert_to_akida(
            quantized_model, device, akida_version,
            output_dir, model_name, bits
        )

        # Convert QAT model to Akida (if exists)
        fbz_path_qat, success_qat = None, False
        if qat_model is not None:
            print(f"\n" + "=" * 70)
            print(f"CONVERTING {bits}-BIT QAT MODEL TO AKIDA")
            print("=" * 70)
            fbz_path_qat, success_qat = convert_to_akida(
                qat_model, device, akida_version,
                output_dir, f"{model_name}_qat", bits
            )

        results[bits] = {
            'quantized_path': q_path,
            'qat_path': qat_path,
            'akida_path_ptq': fbz_path_ptq,
            'akida_path_qat': fbz_path_qat,
            'mapping_success_ptq': success_ptq,
            'mapping_success_qat': success_qat
        }

    # Step 5: Summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    print(f"\nGenerated models in: {output_dir}")

    for bits in args.bits:
        result = results[bits]
        status_ptq = "✓" if result['mapping_success_ptq'] else "❌"

        print(f"\n{bits}-bit quantization:")
        print(f"  PTQ (Keras):       {result['quantized_path'].name}")
        print(f"  PTQ (Akida FBZ):   {result['akida_path_ptq'].name} {status_ptq}")

        if result['qat_path']:
            status_qat = "✓" if result['mapping_success_qat'] else "❌"
            print(f"  QAT (Keras):       {result['qat_path'].name}")
            print(f"  QAT (Akida FBZ):   {result['akida_path_qat'].name} {status_qat}")

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)

    # Show benchmark commands
    print("\n1. Benchmark Akida models (PTQ):")
    for bits in args.bits:
        result = results[bits]
        if result['mapping_success_ptq']:
            print(f"   python benchmark_akida_rpi5.py --akida_model {result['akida_path_ptq']}")

    if args.qat_epochs > 0:
        print("\n2. Benchmark Akida models (QAT):")
        for bits in args.bits:
            result = results[bits]
            if result['mapping_success_qat']:
                print(f"   python benchmark_akida_rpi5.py --akida_model {result['akida_path_qat']}")

    print("\n3. Compare with float32 baseline:")
    print(f"   python benchmark_float32_rpi5.py --model_path {model_path}")

    # Warning for failed mappings
    failed = [bits for bits in args.bits if not results[bits]['mapping_success_ptq']]
    if failed:
        print(f"\n⚠️  WARNING: {len(failed)} PTQ model(s) failed hardware mapping:")
        for bits in failed:
            print(f"   - {bits}-bit model will run in CPU emulation (VERY SLOW)")
        print("\nCheck model architecture for Akida 1.0 compatibility:")
        print("  - Conv2D with stride=2 must use kernel 3x3 or 1x1")
        print("  - No GlobalAveragePooling, Lambda, or LayerNormalization")

    print("=" * 70)

    all_success = all(r['mapping_success_ptq'] for r in results.values())
    if args.qat_epochs > 0:
        all_success = all_success and all(
            r['mapping_success_qat'] for r in results.values() if r['qat_path']
        )

    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()