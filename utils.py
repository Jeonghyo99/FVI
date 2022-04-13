import os
import random
import time
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch


def find_wav_files(path_to_dir: Union[Path, str]) -> Optional[List[Path]]:
    """Find all wav files in the directory and its subtree.

    Args:
        path_to_dir: Path top directory.
    Returns:
        List containing Path objects or None (nothing found).
    """
    paths = list(sorted(Path(path_to_dir).glob("**/*.wav")))

    if len(paths) == 0:
        return None

    return paths


def set_seed_all(seed: int = 0) -> None:
    """
    Set seed for all random number generators.
    """
    if not isinstance(seed, int):
        seed = 0
    # python random
    random.seed(seed)
    # numpy random
    np.random.seed(seed)
    # torch
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    os.environ["PYTHONHASHSEED"] = str(seed)
    return None


def set_benchmark_mode() -> None:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    return None


def save_checkpoint(epoch, model, optimizer, model_kwargs, filename):
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_kwargs": model_kwargs,
    }
    time.sleep(3)
    torch.save(state, filename)
