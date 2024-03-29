#!/usr/bin/env python3

"""
Main file used to control the training process.
"""

import os
import sys
import logging
import random
import numpy as np
import torch
import torch.nn as nn
from shutil import copyfile
from random import randint

from src.pytorch.k_fold_training_data import KFoldTrainingData
from src.pytorch.model import HNN
from src.pytorch.train_workflow import TrainWorkflow
from src.pytorch.log import setup_full_logging
from src.pytorch.utils.helpers import (
    add_train_arg,
    get_memory_usage_mb,
)
from src.pytorch.utils.file_helpers import (
    create_train_directory,
    create_fake_samplefile,
)
from src.pytorch.utils.log_helpers import logging_train_config
from src.pytorch.utils.parse_args import get_train_args
from src.pytorch.utils.timer import Timer
from argparse import Namespace

_log = logging.getLogger(__name__)


def train_main(args: Namespace):
    """
    Higher-level setup of the full training procedure.
    """
    _log.info("Starting TRAINING procedure...")

    set_seeds(args)

    dirname = create_train_directory(args)
    setup_full_logging(dirname)

    fake_samplefile = not os.path.exists(args.samples) or args.max_training_time <= 0.0
    if not fake_samplefile:
        with open(args.samples) as f:
            fake_samplefile = len(f.readlines()) < 10
    if fake_samplefile:
        _log.warning("Creating fake sample file...")
        create_fake_samplefile(args.samples, args.facts_file)
        args.training_size = 1.0
        args.max_epochs = 0

    if args.shared_timers and not fake_samplefile:
        with open(os.path.abspath(args.samples), "r") as f:
            for line in f.readlines(1024):
                if "#<Time>=" in line:
                    times = line[8:].split("/")
                    extra = int(float(times[1]) - float(times[0][:-1]))
                    args.max_training_time += extra
                    break

    if len(args.hidden_units) not in [0, 1, args.hidden_layers]:
        _log.error("Invalid hidden_units length.")
        return

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and args.use_gpu else "cpu"
    )
    if device == torch.device("cpu"):
        args.use_gpu = False

    cmd_line = " ".join(sys.argv[0:])
    logging_train_config(args, dirname, cmd_line)

    best_fold, num_retries, train_timer = train_nn(args, dirname, device)

    _log.info("Finishing training.")
    _log.info(
        f"Elapsed time: {round(train_timer.current_time(), 4)}/{args.max_training_time}s"
    )
    if num_retries:
        _log.info(f"Restarts needed: {num_retries}")

    if best_fold["fold"] != -1:
        try:
            if args.training_size != 1.0 and args.num_folds > 1:
                _log.info(
                    f"Saving traced_{best_fold['fold']}.pt as best "
                    f"model (by val loss = {best_fold['val_loss']})"
                )
                copyfile(
                    f"{dirname}/models/traced_{best_fold['fold']}.pt",
                    f"{dirname}/models/traced_best_val_loss.pt",
                )
        except:
            _log.error(f"Failed to save best fold.")

        _log.info(f"Peak memory usage: {get_memory_usage_mb(True)} MB")
        _log.info("Training complete!")
    else:
        _log.error("Training incomplete! No trained networks.")

    if args.samples.startswith("fake_") or fake_samplefile:
        os.remove(args.samples)


def train_nn(args: Namespace, dirname: str, device: torch.device):
    """
    Manages the training procedure.
    """
    num_retries = 0
    born_dead = True
    _log.warning(f"ATTENTION: Training will be performed on device '{device}'.")

    losses = {"mse": nn.MSELoss()}
    chosen_loss_function = losses[args.loss_function]

    train_timer = Timer(args.max_training_time).start()
    while born_dead:
        starting_time = train_timer.current_time()
        kfold = KFoldTrainingData(
            args.samples,
            max_training_memory=args.max_training_memory,
            device=device,
            batch_size=args.batch_size,
            num_folds=args.num_folds,
            output_layer=args.output_layer,
            shuffle=args.shuffle,
            seed=args.seed,
            shuffle_seed=args.shuffle_seed,
            training_size=args.training_size,
            data_num_workers=0,
            normalize=args.normalize_output,
            model=args.model,
        )
        _log.info(
            f"Loading the training data took {round(train_timer.current_time() - starting_time, 4)}s."
        )

        if args.normalize_output:
            # Add the reference value in train_args.json to denormalize in the test
            add_train_arg(dirname, "max_h", kfold.sample_max_value)

        best_fold = {"fold": -1, "val_loss": float("inf")}

        for fold_idx in range(args.num_folds):
            _log.info(
                f"Running training workflow for fold {fold_idx+1} out "
                f"of {args.num_folds}"
            )

            train_dataloader, val_dataloader, test_dataloader = kfold.get_fold(fold_idx)

            model = HNN(
                input_units=train_dataloader.dataset.x_shape()[1],
                hidden_units=args.hidden_units,
                output_units=train_dataloader.dataset.y_shape()[1],
                hidden_layers=args.hidden_layers,
                activation=args.activation,
                output_layer=args.output_layer,
                dropout_rate=args.dropout_rate,
                linear_output=args.linear_output,
                use_bias=args.bias,
                use_bias_output=args.bias_output,
                weights_method=args.weights_method,
                model=args.model,
            ).to(device)

            if fold_idx == 0:
                _log.info(f"\n{model}")

            train_wf = TrainWorkflow(
                model=model,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                test_dataloader=test_dataloader,
                device=device,
                max_epochs=args.max_epochs,
                save_best=args.save_best_epoch_model,
                dirname=dirname,
                optimizer=torch.optim.Adam(
                    model.parameters(),
                    lr=args.learning_rate,
                    weight_decay=args.weight_decay,
                ),
                check_dead_once=args.check_dead_once,
                loss_fn=chosen_loss_function,
                restart_no_conv=args.restart_no_conv,
                patience=args.patience,
            )

            fold_val_loss = train_wf.run(train_timer)

            born_dead = fold_val_loss is None
            if born_dead and args.num_folds == 1:
                args.seed += args.seed_increment_when_born_dead
                _log.info(f"Updated seed: {args.seed}")
                set_seeds(args, False)
                num_retries += 1
                add_train_arg(dirname, "updated_seed", args.seed)
            else:
                if fold_val_loss is not None:
                    if fold_val_loss < best_fold["val_loss"]:
                        _log.info(
                            f"New best val loss at fold {fold_idx} = {fold_val_loss}"
                        )
                        best_fold["fold"] = fold_idx
                        best_fold["val_loss"] = fold_val_loss
                    else:
                        _log.info(
                            f"Val loss at fold {fold_idx} = {fold_val_loss} (best = {best_fold['val_loss']})"
                        )
                else:  # Only using training data
                    best_fold["fold"] = fold_idx
                    best_fold["train_loss"] = train_wf.cur_train_loss

                train_wf.save_traced_model(
                    f"{dirname}/models/traced_{fold_idx}.pt", args.model
                )

        if train_timer.check_timeout():
            _log.info(f"Maximum training time reached. Stopping training.")
            break

    return best_fold, num_retries, train_timer


def set_seeds(args: Namespace, shuffle_seed: bool = True):
    """
    Sets seeds to assure program reproducibility.
    """
    if args.seed == -1:
        args.seed = randint(0, 2**32 - 1)
    if shuffle_seed and args.shuffle_seed == -1:
        args.shuffle_seed = args.seed
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    random.seed(args.seed)
    np.random.seed(args.seed)


if __name__ == "__main__":
    train_main(get_train_args())
