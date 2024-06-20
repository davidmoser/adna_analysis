# Logging CPU and GPU memory

import psutil
import torch


def log_memory_usage():
    # RAM Usage
    ram_usage = psutil.virtual_memory().percent  # Get the system RAM usage percentage
    log = f"System RAM usage: {ram_usage}%"

    if torch.cuda.is_available():
        # GPU RAM Usage
        gpu_ram_allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)  # Convert bytes to GB
        gpu_ram_reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)  # Convert bytes to GB
        log += f", GPU RAM allocated: {gpu_ram_allocated:.2f} GB"
        log += f", GPU RAM reserved: {gpu_ram_reserved:.2f} GB"

    print(log)
