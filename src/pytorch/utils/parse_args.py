import argparse
from pathlib import Path

import src.pytorch.utils.default_args as default_args


def get_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "samples",
        type=str,
        help="Path to file with samples to be used in training.",
    )
    parser.add_argument(
        "-tf",
        "--train-folder",
        type=Path,
        default=None,
        help="Path to folder to save the trained model.",
    )
    parser.add_argument(
        "-mdl",
        "--model",
        choices=["hnn", "resnet"],
        default=default_args.TRAIN_MODEL,
        help="Network model to use. (default: %(default)s)",
    )
    parser.add_argument(
        "-f",
        "--num-folds",
        type=int,
        default=default_args.TRAIN_NUM_FOLDS,
        help="Number of folds to split training data. (default: %(default)s)",
    )
    parser.add_argument(
        "-hl",
        "--hidden-layers",
        type=int,
        default=default_args.TRAIN_HIDDEN_LAYERS,
        help="Number of hidden layers of the network. (default: %(default)s)",
    )
    parser.add_argument(
        "-hu",
        "--hidden-units",
        type=int,
        nargs="+",
        default=default_args.TRAIN_HIDDEN_UNITS,
        help='Number of units in each hidden layers. For all hidden layers with same size enter \
              only one value; for different size between layers enter "hidden_layers" values. \
              Use 0 to make it scalable according to the input and output units. (default: 250)',
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=default_args.TRAIN_BATCH_SIZE,
        help="Number of samples used in each step of training. (default: %(default)s)",
    )
    parser.add_argument(
        "-bi",
        "--bias",
        type=str2bool,
        default=default_args.TRAIN_BIAS,
        help="Use bias or not. (default: %(default)s)",
    )
    parser.add_argument(
        "-biout",
        "--bias-output",
        type=str2bool,
        default=default_args.TRAIN_BIAS,
        help="Use bias or not in the output layer. (default: %(default)s)",
    )
    parser.add_argument(
        "-wm",
        "--weights-method",
        choices=[
            "default",
            "sqrt_k",
            "1",
            "01",
            "xavier_uniform",
            "xavier_normal",
            "kaiming_uniform",
            "kaiming_normal",
        ],
        default=default_args.TRAIN_WEIGHTS_METHOD,
        help="Initialization of network weights. (default: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--output-layer",
        choices=["regression", "prefix", "one-hot"],
        default=default_args.TRAIN_OUTPUT_LAYER,
        help="Network output layer type. (default: %(default)s)",
    )
    parser.add_argument(
        "-lo",
        "--linear-output",
        type=str2bool,
        default=default_args.TRAIN_LINEAR_OUTPUT,
        help="Use linear output in the output layer (True) or use an activation (False). (default: %(default)s)",
    )
    parser.add_argument(
        "-no",
        "--normalize-output",
        type=str2bool,
        default=default_args.TRAIN_NORMALIZE_OUTPUT,
        help="Normalizes the output neuron. (default: %(default)s)",
    )
    parser.add_argument(
        "-a",
        "--activation",
        choices=["sigmoid", "relu", "leakyrelu"],
        default=default_args.TRAIN_ACTIVATION,
        help="Activation function for hidden layers. (default: %(default)s)",
    )
    parser.add_argument(
        "-lf",
        "--loss-function",
        choices=[
            "mse",
            "rmse",
        ],
        default=default_args.TRAIN_LOSS_FUNCTION,
        help="Loss function to be used during training. (default: %(default)s)",
    )
    parser.add_argument(
        "-w",
        "--weight-decay",
        "--regularization",
        type=float,
        default=default_args.TRAIN_WEIGHT_DECAY,
        help="Weight decay (L2 regularization) to use in network training. (default: %(default)s)",
    )
    parser.add_argument(
        "-d",
        "--dropout-rate",
        type=float,
        default=default_args.TRAIN_DROPOUT_RATE,
        help="Dropout rate for hidden layers. (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        default=default_args.TRAIN_LEARNING_RATE,
        help="Network learning rate. (default: %(default)s)",
    )
    parser.add_argument(
        "-pat",
        "--patience",
        type=int,
        default=default_args.TRAIN_PATIENCE,
        help="Early-stop patience. (default: %(default)s)",
    )
    parser.add_argument(
        "-sh",
        "--shuffle",
        type=str2bool,
        default=default_args.TRAIN_SHUFFLE,
        help="Shuffle the training data. (default: %(default)s)",
    )
    parser.add_argument(
        "-shs",
        "--shuffle-seed",
        type=int,
        default=default_args.DEFAULT_SEED,
        help="Seed to be used for separating training and validation data. Defaults to network seed. (default: %(default)s)",
    )
    parser.add_argument(
        "-tsize",
        "--training-size",
        type=float,
        default=default_args.TRAIN_TRAINING_SIZE,
        help="Training data size in relation to validation data. (default: %(default)s)",
    )
    parser.add_argument(
        "-e",
        "--max-epochs",
        type=int,
        default=default_args.TRAIN_MAX_EPOCHS,
        help="Maximum number of epochs to train each fold (or -1 for fixed value). (default: %(default)s)",
    )
    parser.add_argument(
        "-t",
        "--max-training-time",
        type=int,
        default=default_args.TRAIN_MAX_TIME,
        help="Maximum network training time (all folds). (default: %(default)ss)",
    )
    parser.add_argument(
        "-mem",
        "--max-training-memory",
        type=int,
        default=default_args.TRAIN_MAX_MEMORY,
        help="Maximum network training memory (all folds). (default: %(default)ss)",
    )
    parser.add_argument(
        "-rst",
        "--restart-no-conv",
        type=str2bool,
        default=default_args.TRAIN_RESTART_NO_CONV,
        help="Restarts the network if it won't converge. (default: %(default)s)",
    )
    parser.add_argument(
        "-cdead",
        "--check-dead-once",
        type=str2bool,
        default=default_args.TRAIN_CHECK_DEAD_ONCE,
        help="Only check if network is dead once, at the start of the first epoch. (default: %(default)s)",
    )
    parser.add_argument(
        "-sibd",
        "--seed-increment-when-born-dead",
        type=int,
        default=default_args.TRAIN_SEED_INCREMENT_WHEN_BORN_DEAD,
        help="Seed increment when the network needs to restart due to born dead. (default: %(default)s)",
    )
    parser.add_argument(
        "-sb",
        "--save-best-epoch-model",
        type=str2bool,
        default=default_args.TRAIN_SAVE_BEST_EPOCH_MODEL,
        help="Saves the best model from the best epoch instead of the last one. (default: %(default)s)",
    )
    parser.add_argument(
        "-of",
        "--output-folder",
        type=Path,
        default=default_args.TRAIN_OUTPUT_FOLDER,
        help="Path where the training folder will be saved. (default: %(default)s)",
    )
    parser.add_argument(
        "-ffile",
        "--facts-file",
        type=str,
        default="",
        help="Order of facts during sampling. (default: %(default)s)",
    )
    parser.add_argument(
        "-gpu",
        "--use-gpu",
        type=str2bool,
        default=default_args.TRAIN_USE_GPU,
        help="Use GPU during training. (default: %(default)s)",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=default_args.DEFAULT_SEED,
        help="Random seed to be used. Defaults to no seed. (default: random)",
    )
    parser.add_argument(
        "-shared-timers",
        "--shared-timers",
        type=str2bool,
        default=default_args.SHARED_TIMERS,
        help="Share timers between sampling, training and testing. (default: %(default)s)",
    )
    return parser.parse_args()


def get_test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "train_folder", type=Path, help="Path to training folder with trained model."
    )
    parser.add_argument(
        "problem_pddls", type=str, nargs="*", help="Path to problems PDDL."
    )
    parser.add_argument(
        "-p",
        "--plan-file",
        type=str,
        default=default_args.TEST_PLAN_FILE,
        help="Path to plan file.",
    )
    parser.add_argument(
        "-d",
        "--domain-pddl",
        type=str,
        default=default_args.TEST_DOMAIN_PDDL,
        help="Path to domain PDDL. (default: problem_folder/domain.pddl)",
    )
    parser.add_argument(
        "-a",
        "--search-algorithm",
        choices=["astar", "eager_greedy"],
        default=default_args.TEST_SEARCH_ALGORITHM,
        help="Algorithm to be used in the search. (default: %(default)s)",
    )
    parser.add_argument(
        "-heu",
        "--heuristic",
        choices=["nn", "add", "blind", "ff", "goalcount", "hmax", "lmcut", "hstar"],
        default=default_args.TEST_HEURISTIC,
        help="Heuristic to be used in the search. (default: %(default)s)",
    )
    parser.add_argument(
        "-hm",
        "--heuristic-multiplier",
        type=int,
        default=default_args.TEST_HEURISTIC_MULTIPLIER,
        help="Value to multiply the output heuristic with. (default: %(default)s)",
    )
    parser.add_argument(
        "-pt",
        "--test-model",
        choices=["all", "best", "epochs"],
        default=default_args.TEST_TEST_MODEL,
        help="Model(s) used for testing. (default: %(default)s)",
    )
    parser.add_argument(
        "-ffile",
        "--facts-file",
        type=str,
        default="",
        help="Order of facts during sampling. (default: %(default)s)",
    )
    parser.add_argument(
        "-dfile",
        "--defaults-file",
        type=str,
        default="",
        help="Default values for facts given with `ffile`. (default: %(default)s)",
    )
    parser.add_argument(
        "-t",
        "--max-search-time",
        type=int,
        default=default_args.TEST_MAX_SEARCH_TIME,
        help="Time limit for each search. (default: %(default)ss)",
    )
    parser.add_argument(
        "-m",
        "--max-search-memory",
        type=int,
        default=default_args.TEST_MAX_SEARCH_MEMORY,
        help="Memory limit for each search. (default: %(default)sMB)",
    )
    parser.add_argument(
        "-e",
        "--max-expansions",
        type=int,
        default=default_args.TEST_MAX_EXPANSIONS,
        help="Maximum expanded states for each search (or -1 for fixed value). (default: %(default)s)",
    )
    parser.add_argument(
        "-dlog",
        "--downward-logs",
        type=str2bool,
        default=default_args.TEST_SAVE_DOWNWARD_LOGS,
        help="Save each instance's Fast-Downward log or not. (default: %(default)s)",
    )
    parser.add_argument(
        "-unit-cost",
        "--unit-cost",
        type=str2bool,
        default=default_args.UNIT_COST,
        help="Test with unit cost instead of operator cost. (default: %(default)s)",
    )
    parser.add_argument(
        "-shared-timers",
        "--shared-timers",
        type=str2bool,
        default=default_args.SHARED_TIMERS,
        help="Share timers between sampling, training and testing. (default: %(default)s)",
    )
    return parser.parse_args()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
