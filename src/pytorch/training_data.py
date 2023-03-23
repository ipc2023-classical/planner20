import torch
import logging
import numpy as np
from torch.utils.data import Dataset
from src.pytorch.utils.helpers import to_prefix, to_onehot, get_memory_usage_mb

_log = logging.getLogger(__name__)


class InstanceDataset(Dataset):
    def __init__(
        self,
        states: np.array,
        heuristics: np.array,
        sample_max_value: int,
        output_layer: str,
    ):
        self.output_layer = output_layer
        self.sample_max_value = sample_max_value

        self.states = torch.tensor(states, dtype=torch.int8)
        states = None

        if output_layer == "regression":
            self.hvalues = torch.tensor(heuristics, dtype=torch.float32).unsqueeze(1)

        elif output_layer == "prefix":
            self.hvalues = torch.tensor(
                [to_prefix(n, self.sample_max_value) for n in np.array(heuristics)],
                dtype=torch.float32,
            )

        elif output_layer == "one-hot":
            self.hvalues = torch.tensor(
                [to_onehot(n, self.sample_max_value) for n in np.array(heuristics)],
                dtype=torch.float32,
            )
        else:
            raise RuntimeError(f"Invalid output layer: {output_layer}")

        heuristics = None

    def __getitem__(self, idx: int):
        return self.states[idx], self.hvalues[idx]

    def __len__(self):
        return len(self.states)

    def x_shape(self):
        return self.states.shape

    def y_shape(self):
        if self.output_layer == "regression":
            return torch.Size([len(self.hvalues), 1])
        return self.hvalues.shape


def load_training_state_value_pairs(samples_file: str, max_training_memory: int):
    """
    Loads the data.
    """
    states, heuristics = [], []
    sample_max_value, max_h = 0, 0

    with open(samples_file) as f:
        for line in f:
            if line[0] != "#":
                line = line.split("\n")[0]

                h, state = line.split(";")
                h_int = int(h)

                st = np.fromiter(state, dtype=np.int8)
                states.append(st)
                heuristics.append(h_int)

                # Gets the domain max h value.
                if h_int > max_h:
                    max_h = h_int
                    sample_max_value = max_h

                # Check for max memory usage
                if get_memory_usage_mb() >= max_training_memory:
                    _log.info(
                        "Maximum training memory reached during loading. Some samples will not be loaded."
                    )
                    break

    return np.array(states), np.array(heuristics), sample_max_value
