import logging
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from torch.utils.data import DataLoader
from src.pytorch.model import HNN
from src.pytorch.utils.helpers import prefix_to_h, get_memory_usage_mb
from src.pytorch.utils.timer import Timer

_log = logging.getLogger(__name__)


class TrainWorkflow:
    def __init__(
        self,
        model: HNN,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: DataLoader,
        device: torch.device,
        max_epochs: int,
        save_best: bool,
        dirname: str,
        optimizer: optim.Optimizer,
        check_dead_once: bool,
        loss_fn: nn = nn.MSELoss(),
        restart_no_conv: bool = True,
        patience: int = None,
    ):
        self.model = model
        self.best_epoch_model = None
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.validation = self.val_dataloader is not None
        self.testing = self.test_dataloader is not None
        self.device = device
        self.max_epochs = max_epochs
        self.save_best = save_best
        self.dirname = dirname
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.patience = patience
        self.early_stopped = False
        self.restart_no_conv = restart_no_conv
        self.check_dead_once = check_dead_once
        self.train_y_pred_values = []  # [state, y, pred]
        self.val_y_pred_values = []  # [state, y, pred]

    def train_loop(self) -> float:
        """
        Network's train loop.
        """
        # size = len(self.train_dataloader.dataset)
        num_batches = len(self.train_dataloader)
        train_loss = 0

        for _batch, item in enumerate(self.train_dataloader):
            # Compute prediction and loss.
            X, y = item[0].to(self.device), item[1].to(self.device)
            pred = self.model(X.float())
            loss = self.loss_fn(pred, y)

            train_loss += loss.item()

            # Clear gradients for the variables it will update.
            self.optimizer.zero_grad()

            # Compute gradient of the loss.
            loss.backward()

            # Update parameters.
            self.optimizer.step()

        return train_loss / num_batches

    def val_loop(self) -> float:
        """
        Network's evaluation loop.
        """
        num_batches = len(self.val_dataloader)
        val_loss = 0
        with torch.no_grad():
            for item in self.val_dataloader:
                X, y = item[0].to(self.device), item[1].to(self.device)
                pred = self.model(X.float())
                val_loss += self.loss_fn(pred, y).item()

        return val_loss / num_batches

    def test_loop(self) -> float:
        """
        Network's testing loop.
        """
        num_batches = len(self.test_dataloader)
        test_loss = 0
        with torch.no_grad():
            for item in self.test_dataloader:
                X, y = item[0].to(self.device), item[1].to(self.device)
                pred = self.model(X.float())
                test_loss += self.loss_fn(pred, y).item()

        return test_loss / num_batches

    def dead(self) -> bool:
        """
        Checks if the network is dead.
        """
        with torch.no_grad():
            for item in self.train_dataloader:
                X = item[0] if self.device == "cpu" else item[0].to(self.device)
                for p in self.model(X.float()):
                    p_list = p.tolist()
                    if type(p_list) is float:
                        if p_list != 0.0:
                            return False
                    else:
                        if len(p) > 1:  # prefix
                            p = prefix_to_h(p_list)
                        if float(p) != 0.0:
                            return False
        return True

    def save_traced_model(self, filename: str, model="hnn"):
        """
        Saves a traced model to be used by the C++ backend.
        """
        if model == "resnet":
            example_input = self.train_dataloader.dataset[:10][0].float()
        elif model == "simple" or model == "hnn":
            example_input = self.train_dataloader.dataset[0][0].float()

        # To make testing possible (and fair), the model has to be saved while in the CPU,
        # even if training was performed in GPU.
        traced_model = torch.jit.trace(self.best_epoch_model.to("cpu"), example_input)
        traced_model.save(filename)

    def run(self, train_timer: Timer):
        """
        Network train/eval main loop.
        """
        if self.max_epochs == 0:
            self.best_epoch_model = deepcopy(self.model)
            return -1

        best_loss, best_epoch = None, None
        cur_train_loss, cur_val_loss = None, None
        born_dead = False
        check_once = False
        t = 0
        while (
            t < self.max_epochs
            and not self.early_stopped
            and not train_timer.check_timeout()
        ):
            cur_train_loss = self.train_loop()
            # Check if born dead (or died during training)
            if not (t % 10) and not born_dead and not check_once:
                if self.dead():
                    if self.restart_no_conv:
                        _log.warning(
                            "All predictions are 0 (born dead). Restarting training with a new seed..."
                        )
                        return None
                    else:
                        _log.warning(
                            "All predictions are 0 (born dead), but restart is disabled."
                        )
                        born_dead = True
                if self.check_dead_once:
                    check_once = True

            epoch_log = f"Epoch {t} | avg_train_loss={cur_train_loss:>7f}"

            if self.validation:
                cur_val_loss = self.val_loop()
                epoch_log += f" | avg_val_loss={cur_val_loss:>7f}"

            cur_loss = cur_val_loss if self.validation else cur_train_loss
            new_best = False
            if not best_loss or best_loss > cur_loss:
                best_loss, best_epoch = cur_loss, t
                self.best_epoch_model = deepcopy(self.model)
                new_best = True
            if best_epoch < t - self.patience:
                self.early_stopped = True

            if self.testing:
                cur_test_loss = self.test_loop()
                epoch_log += f" | avg_test_loss={cur_test_loss:>7f}"

            if new_best:
                epoch_log += " *"

            _log.info(epoch_log)

            if t % 10 == 0:
                _log.debug(f"Current mem usage: {get_memory_usage_mb()} MB")

            t += 1

        if self.early_stopped:
            _log.info(f"Early stop. Best epoch: {best_epoch+1}/{t}")
        if train_timer.check_timeout():
            _log.info(f"Training time reached. Best epoch: {best_epoch+1}/{t}")
        if t == self.max_epochs:
            _log.info(f"Max epoch reached. Best epoch: {best_epoch+1}/{t}")
        _log.info(f"Mem usage END: {get_memory_usage_mb()} MB")

        if not self.save_best:
            self.best_epoch_model = self.model

        return best_loss
