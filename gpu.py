import torch
import time
import subprocess
import datetime

def print_gpu_status():
    """
    Prints the status of all available GPUs using nvidia-smi.
    """
    if not torch.cuda.is_available():
        print("No CUDA GPUs available.")
        return

    # Get current timestamp for the log
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"\n{'='*75}")
    print(f"GPU Status Report | Time: {timestamp}")
    print(f"{'='*75}")

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        print(
            f"{'ID':<4} {'Name':<20} {'Mem Used':<12} {'Total Cap':<12} {'Util %':<8} {'Temp':<6}"
        )
        print(f"{'-'*75}")

        for line in result.stdout.strip().split("\n"):
            parts = line.split(",")
            if len(parts) >= 6:
                idx, name, used, total, util, temp = [p.strip() for p in parts]
                print(
                    f"{idx:<4} {name:<20} {used:>5} MB{' '*4} {total:>5} MB{' '*4} {util:>3}%{' '*4} {temp:>3}C"
                )

        print(f"{'='*75}\n")

    except Exception as e:
        print(f"Could not query GPU status: {e}")


def stress_all_gpus(duration_hours=24, memory_fraction=0.5):
    """
    Allocates 50% memory and runs compute on ALL available GPUs for 24 hours.
    """
    if not torch.cuda.is_available():
        print("No CUDA GPUs detected. Exiting.")
        return

    duration_sec = duration_hours * 3600
    num_gpus = torch.cuda.device_count()
    
    print(f"Detected {num_gpus} GPUs.")
    print(f"Target Duration: {duration_hours} hours ({duration_sec} seconds)")
    print(f"Target Memory: {memory_fraction*100}% of available per GPU")
    print("-" * 50)

    tensors = [] 

    # --- 1. Allocate 50% Memory on ALL Devices ---
    print("Allocating memory...")
    for i in range(num_gpus):
        device = torch.device(f"cuda:{i}")
        
        try:
            # Get free memory in bytes
            free_mem, total_mem = torch.cuda.mem_get_info(i)
            
            # Calculate allocation size (50% of free memory)
            alloc_size = int(free_mem * memory_fraction)
            num_elements = alloc_size // 4  # float32 is 4 bytes
            
            # Create tensor
            t = torch.empty(num_elements, dtype=torch.float32, device=device)
            tensors.append(t)
            
            print(f"  > GPU {i}: Allocated {alloc_size / (1024**3):.2f} GB "
                  f"({memory_fraction*100}% of free)")
            
        except RuntimeError as e:
            print(f"  > GPU {i} allocation failed: {e}")

    print("\nStarting 24-hour compute workload...")
    print("Log format: '.' = 1 minute passed | 'STATUS' = 1 hour passed")
    print_gpu_status()

    # --- 2. Run Compute Load ---
    start_time = time.time()
    last_print_time = start_time
    last_dot_time = start_time
    
    try:
        while (time.time() - start_time) < duration_sec:
            # A. Heavy Compute
            for t in tensors:
                dim = 4096 
                if t.numel() >= dim * dim:
                    sub_tensor = t[:dim*dim].view(dim, dim)
                    # Perform matmul to spike utilization
                    _ = torch.matmul(sub_tensor, sub_tensor)
            
            current_time = time.time()

            # B. Print a dot every minute to show liveness
            if current_time - last_dot_time > 60:
                print(".", end="", flush=True)
                last_dot_time = current_time

            # C. Print full status report every 1 hour (3600 seconds)
            if current_time - last_print_time > 3600:
                print("\n") # Newline for the report
                print_gpu_status()
                last_print_time = current_time

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")

    # --- 3. Cleanup ---
    print(f"\nTest Finished after {(time.time() - start_time)/3600:.2f} hours.")
    print("Releasing memory...")
    del tensors
    torch.cuda.empty_cache()
    print("Done.")

if __name__ == "__main__":
    # Stress test for 24 hours at 50% memory
    stress_all_gpus(duration_hours=24, memory_fraction=0.5)