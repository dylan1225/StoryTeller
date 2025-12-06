"""
Device selection and management utilities.

This module provides intelligent device selection for training,
automatically finding the best available compute resource.
"""

import torch
import subprocess


def get_gpu_memory_usage(device_id: int) -> float:
    """
    Get GPU memory usage percentage for a specific CUDA device.

    Args:
        device_id: CUDA device ID

    Returns:
        Memory usage as a percentage (0-100), or 100.0 if unavailable
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,nounits,noheader",
                f"--id={device_id}",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0:
            output = result.stdout.strip()
            used, total = map(float, output.split(","))
            usage_percent = (used / total) * 100 if total > 0 else 100.0
            return usage_percent
    except Exception:
        pass

    return 100.0  # Assume full if we can't check


def find_available_cuda_device(threshold: float = 20.0) -> int:
    """
    Find the first CUDA GPU with memory usage below the threshold.

    Args:
        threshold: Maximum acceptable memory usage percentage (default 20%)

    Returns:
        Device ID of available GPU, or -1 if none available
    """
    if not torch.cuda.is_available():
        return -1

    num_gpus = torch.cuda.device_count()

    for device_id in range(num_gpus):
        usage = get_gpu_memory_usage(device_id)
        if usage < threshold:
            return device_id

    return -1


def smart_select_device(verbose: bool = True) -> torch.device:
    """
    Intelligently select the best available compute device.

    Priority:
    1. MPS (Apple Silicon) if available
    2. CUDA GPU with lowest memory usage (below 20% usage)
    3. CPU as fallback

    Args:
        verbose: Whether to print device selection information

    Returns:
        torch.device object
    """
    if verbose:
        print("Smart device selection enabled...")

    # Priority 1: MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        if verbose:
            print("  ✓ MPS (Apple Silicon) available and selected")
        return torch.device("mps")

    # Priority 2: CUDA GPU (find one that's not busy)
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if verbose:
            print(f"  ✓ CUDA available with {num_gpus} GPU(s)")

        # Try to find an available GPU
        device_id = find_available_cuda_device(threshold=20.0)

        if device_id >= 0:
            device_name = torch.cuda.get_device_name(device_id)
            usage = get_gpu_memory_usage(device_id)
            if verbose:
                print(
                    f"  ✓ Selected GPU {device_id}: {device_name} (memory usage: {usage:.1f}%)"
                )
            return torch.device(f"cuda:{device_id}")
        else:
            # All GPUs busy, use the first one anyway
            device_name = torch.cuda.get_device_name(0)
            usage = get_gpu_memory_usage(0)
            if verbose:
                print(
                    f"  ⚠ All GPUs busy, using GPU 0: {device_name} (memory usage: {usage:.1f}%)"
                )
            return torch.device("cuda:0")

    # Priority 3: CPU fallback
    if verbose:
        print("  ⚠ No GPU acceleration available, using CPU")
        print("    (Training will be significantly slower)")
    return torch.device("cpu")


def custom_select_device(verbose: bool = True) -> torch.device:
    """
    Aggressively pick the CUDA GPU with the most free memory.

    This mode is intended for configs that want to maximize throughput by
    claiming the best available GPU rather than politely yielding to busy
    devices. Uses `torch.cuda.mem_get_info`/`nvidia-smi` metrics similar to
    gpu.py for visibility.

    Returns:
        torch.device object targeting the chosen GPU, or smart fallback.
    """

    if not torch.cuda.is_available():
        if verbose:
            print("Custom device requested but no CUDA GPUs detected; falling back to smart selection.")
        return smart_select_device(verbose=verbose)

    num_gpus = torch.cuda.device_count()
    gpu_stats = []

    for device_id in range(num_gpus):
        try:
            free_mem, total_mem = torch.cuda.mem_get_info(device_id)
        except Exception:
            free_mem, total_mem = 0, 0

        usage = get_gpu_memory_usage(device_id)

        gpu_stats.append(
            {
                "id": device_id,
                "free_bytes": free_mem,
                "total_bytes": total_mem,
                "utilization": usage,
                "name": torch.cuda.get_device_name(device_id),
            }
        )

    # Pick GPU with the most free memory; tie-breaker: lowest current usage
    best_gpu = sorted(
        gpu_stats,
        key=lambda g: (g["free_bytes"], -g["utilization"]),
        reverse=True,
    )[0]

    if verbose:
        print("Custom device selection (maximize GPU throughput):")
        print(f"  Detected {num_gpus} CUDA device(s)")
        print("  ID  Name                       Free / Total (GB)   Util %")
        for g in gpu_stats:
            free_gb = g["free_bytes"] / 1e9 if g["total_bytes"] else 0.0
            total_gb = g["total_bytes"] / 1e9 if g["total_bytes"] else 0.0
            print(
                f"  {g['id']:<3} {g['name'][:23]:<23} "
                f"{free_gb:5.1f} / {total_gb:5.1f}       {g['utilization']:5.1f}%"
            )
        print(
            f"  → Selecting GPU {best_gpu['id']} ({best_gpu['name']}) for maximum compute headroom"
        )

    # Lock to the chosen GPU to claim its full compute/memory budget
    torch.cuda.set_device(best_gpu["id"])
    return torch.device(f"cuda:{best_gpu['id']}")


def get_device_info(device: torch.device) -> dict:
    """
    Get detailed information about a device.

    Args:
        device: torch.device object

    Returns:
        Dictionary with device information
    """
    info = {
        "type": device.type,
        "index": device.index,
    }

    if device.type == "cuda":
        if torch.cuda.is_available():
            device_id = device.index if device.index is not None else 0
            info["name"] = torch.cuda.get_device_name(device_id)
            info["memory_allocated_gb"] = torch.cuda.memory_allocated(device_id) / 1e9
            info["memory_reserved_gb"] = torch.cuda.memory_reserved(device_id) / 1e9
            info["memory_total_gb"] = (
                torch.cuda.get_device_properties(device_id).total_memory / 1e9
            )
            info["memory_usage_percent"] = get_gpu_memory_usage(device_id)
    elif device.type == "mps":
        info["name"] = "Apple Silicon (MPS)"
    elif device.type == "cpu":
        info["name"] = "CPU"

    return info


if __name__ == "__main__":
    # Test the device selection
    print("Testing smart device selection...\n")

    device = smart_select_device()
    print(f"\nSelected device: {device}")

    info = get_device_info(device)
    print("\nDevice info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
