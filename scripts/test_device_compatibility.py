#!/usr/bin/env python
"""
Test device compatibility for MPS, CUDA, and CPU.

This script checks which devices are available and tests basic operations
to ensure the training pipeline will work correctly.
"""

import torch
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def check_device_availability():
    """Check which devices are available."""
    print("=" * 60)
    print("Device Availability Check")
    print("=" * 60)

    devices = []

    # Check CUDA
    if torch.cuda.is_available():
        print("✓ CUDA available")
        print(f"  Devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  CUDA version: {torch.version.cuda}")
        devices.append("cuda")

        # Check bfloat16 support
        if torch.cuda.is_bf16_supported():
            print("  ✓ bfloat16 supported")
        else:
            print("  ✗ bfloat16 not supported (will use float16)")
    else:
        print("✗ CUDA not available")

    # Check MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        print("\n✓ MPS (Apple Silicon) available")
        print(f"  MPS backend: {torch.backends.mps.is_built()}")
        devices.append("mps")
        print("  Note: MPS only supports float16, not bfloat16")
    else:
        print("\n✗ MPS not available")

    # CPU is always available
    print("\n✓ CPU available")
    print(f"  Threads: {torch.get_num_threads()}")
    devices.append("cpu")

    print(f"\nAvailable devices: {', '.join(devices)}")
    return devices


def test_basic_operations(device_str: str):
    """Test basic tensor operations on a device."""
    print(f"\nTesting {device_str.upper()}:")
    print("-" * 40)

    try:
        device = torch.device(device_str)

        # Create tensors
        x = torch.randn(100, 100, device=device)
        y = torch.randn(100, 100, device=device)

        # Matrix multiplication
        _ = torch.mm(x, y)
        print("  ✓ Matrix multiplication works")

        # Gradient computation
        x_grad = torch.randn(100, 100, device=device, requires_grad=True)
        y_grad = torch.randn(100, 100, device=device, requires_grad=True)
        z_grad = torch.mm(x_grad, y_grad)
        loss = (z_grad * 2).sum()
        loss.backward()
        print("  ✓ Backward pass works")

        # Test mixed precision if not CPU
        if device_str != "cpu":
            try:
                # Try unified AMP (PyTorch 2.0+)
                from torch.amp import autocast

                with autocast(device_type=device_str, dtype=torch.float16):
                    _ = torch.mm(x, y)
                print("  ✓ Mixed precision (unified AMP) works")
            except ImportError:
                # Fall back to old API
                from torch.cuda.amp import autocast

                with autocast(dtype=torch.float16):
                    _ = torch.mm(x, y)
                print("  ✓ Mixed precision (CUDA AMP) works")

            # Test bfloat16 if CUDA
            if device_str == "cuda" and torch.cuda.is_bf16_supported():
                try:
                    from torch.amp import autocast

                    with autocast(device_type=device_str, dtype=torch.bfloat16):
                        _ = torch.mm(x, y)
                    print("  ✓ bfloat16 works")
                except Exception as e:
                    print(f"  ✗ bfloat16 failed: {e}")

        print(f"  ✓ All basic operations passed on {device_str}")
        return True

    except Exception as e:
        print(f"  ✗ Error on {device_str}: {e}")
        return False


def test_model_training(device_str: str):
    """Test a simple model training step."""
    print(f"\nTesting model training on {device_str.upper()}:")
    print("-" * 40)

    try:
        device = torch.device(device_str)

        # Create simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
        ).to(device)

        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Create dummy data
        x = torch.randn(32, 128, device=device)
        target = torch.randn(32, 128, device=device)

        # Training step with AMP
        if device_str != "cpu":
            try:
                from torch.amp import autocast, GradScaler

                scaler = GradScaler(device_str) if device_str == "cuda" else None
                use_scaler = scaler is not None

                with autocast(device_type=device_str, dtype=torch.float16):
                    output = model(x)
                    loss = torch.nn.functional.mse_loss(output, target)

                if use_scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                print("  ✓ Training step with AMP passed")

            except ImportError:
                # Old AMP API
                from torch.cuda.amp import autocast, GradScaler

                scaler = GradScaler() if device_str == "cuda" else None

                with autocast(dtype=torch.float16):
                    output = model(x)
                    loss = torch.nn.functional.mse_loss(output, target)

                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                print("  ✓ Training step with AMP (old API) passed")
        else:
            # CPU without AMP
            output = model(x)
            loss = torch.nn.functional.mse_loss(output, target)
            loss.backward()
            optimizer.step()
            print("  ✓ Training step (no AMP) passed")

        return True

    except Exception as e:
        print(f"  ✗ Training failed on {device_str}: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all compatibility tests."""
    print("\n" + "=" * 60)
    print("Storyteller Device Compatibility Test")
    print("=" * 60 + "\n")

    print(f"PyTorch version: {torch.__version__}\n")

    # Check availability
    available_devices = check_device_availability()

    # Test each available device
    results = {}
    for device in available_devices:
        basic_ok = test_basic_operations(device)
        training_ok = test_model_training(device)
        results[device] = basic_ok and training_ok

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    for device, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{device.upper()}: {status}")

    all_passed = all(results.values())
    if all_passed:
        print("\n✓ All devices passed compatibility tests!")
        print("\nYou can train on any of these devices:")
        for device in available_devices:
            print("  storyteller-train --config configs/base_model.yaml")
            print(f"    (set device: '{device}' in config)")
        return 0
    else:
        print("\n✗ Some devices failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
