"""
Simple auxiliary functions.
"""

import logging
import os
from json import dump, load
from datetime import datetime, timezone

_log = logging.getLogger(__name__)


def to_prefix(n: int, max_value: int):
    """
    Convert value `n` to prefix encoding.
    """
    max_value += 1
    return [1 if i < n else 0 for i in range(max_value)]


def to_onehot(n: int, max_value: int):
    """
    Convert value `n` to onehot encoding.
    """
    max_value += 1
    return [1 if i == n else 0 for i in range(max_value)]


def prefix_to_h(prefix, threshold: float = 0.01) -> int:
    """
    Convert prefix encoding to a value, respecting the given threshold value.
    """
    last_h = len(prefix) - 1
    for i in range(len(prefix)):
        if prefix[i] < threshold:
            last_h = i - 1
            break
    return last_h


def get_datetime() -> str:
    return datetime.now(timezone.utc).strftime("%d %B %Y %H:%M:%S UTC")


def get_memory_usage_mb(peak: bool = False):
    """
    Get current memory in MB.

    Peak memory if `peak` is true, otherwise current memory
    """

    field = "VmPeak:" if peak else "VmSize:"
    with open("/proc/self/status") as f:
        memusage = f.read().split(field)[1].split("\n")[0][:-3]

    return round(int(memusage.strip()) / 1024)


def get_git_commit() -> str:
    return None  # check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def get_hostname() -> str:
    return None  # check_output(["cat", "/proc/sys/kernel/hostname"]).decode("ascii").strip()


def add_train_arg(dirname: str, key, value):
    """
    Adds/updates a key-value pair from the `train_args.json` file.
    """
    with open(f"{dirname}/train_args.json", "r") as f:
        data = load(f)
    data[key] = value
    with open(f"{dirname}/train_args.json", "w") as f:
        dump(data, f, indent=4)


def get_models_from_train_folder(train_folder: str, test_model: str):
    """
    Returns the required trained network models to be used for testing, according to the
    `test_model` chosen.
    """
    models = []

    if train_folder == "":
        return models

    models_folder = f"{train_folder}/models"

    if test_model == "best":
        best_fold_path = f"{models_folder}/traced_best_val_loss.pt"
        if os.path.exists(best_fold_path):
            models.append(best_fold_path)
        else:
            _log.error(f"Best val loss model does not exists!")
    elif test_model == "all":
        i = 0
        while os.path.exists(f"{models_folder}/traced_{i}.pt"):
            models.append(f"{models_folder}/traced_{i}.pt")
            i += 1
    elif test_model == "epochs":
        i = 0
        while os.path.exists(f"{models_folder}/traced_0-epoch-{i}.pt"):
            models.append(f"{models_folder}/traced_0-epoch-{i}.pt")
            i += 1

    return models
