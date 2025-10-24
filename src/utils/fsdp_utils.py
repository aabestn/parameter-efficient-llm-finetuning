import os
import torch


def setup_fsdp_env():
    """Validates PyTorch Distributed / FSDP environment setups."""
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        print(f"Distributed GPU initialized on local rank {local_rank}")
    else:
        print("Running in single GPU or non distributed mode")