#!/usr/bin/env python3

import os
from sys import argv
from time import time

from src.pytorch.utils import default_args

DOMAIN_FILE, PROBLEM_FILE, PLAN_FILE = argv[1:4]

OUTPUT_FOLDER = "results"
TEST_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    ".".join(
        [
            s.replace("/", "_").replace(".pddl", "").replace(".", "d") if s else "none"
            for s in [DOMAIN_FILE, PROBLEM_FILE, str(time())]
        ]
    ),
)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
if not os.path.exists(TEST_FOLDER):
    os.makedirs(TEST_FOLDER)

SAMPLE_FILE = os.path.join(TEST_FOLDER, f"sample_file")
FACTS_FILE = os.path.join(TEST_FOLDER, f"facts_file")
TRAIN_FOLDER = os.path.join(TEST_FOLDER, f"test_folder")

SAMPLING_COMMANDLINE = (
    f'/fast-downward.py --sas-file {os.path.join(TEST_FOLDER, "sampling-output.sas")} --plan-file {SAMPLE_FILE} '
    f"--build release {DOMAIN_FILE} {PROBLEM_FILE} "
    f'--search "sampling_search_fsm(eager_greedy([ff(transform=sampling_transform())], transform=sampling_transform()), '
    f"techniques=[gbackward_fsm("
    f"technique={default_args.SAMPLING_TECHNIQUE}, "
    f"searches={default_args.SAMPLING_SEARCHES}, "
    f"samples_per_search={default_args.SAMPLING_PER_SEARCH}, "
    f"max_samples={default_args.SAMPLING_MAX_SAMPLES}, "
    f"state_filtering={default_args.SAMPLING_STATE_FILTERING}, "
    f"allow_duplicates={default_args.SAMPLING_ALLOW_DUPLICATES}, "
    f"regression_depth={default_args.SAMPLING_REGRESSION_DEPTH}, "
    f"restart_h_when_goal_state={default_args.SAMPLING_RESTART_H_WHEN_GOAL_STATE}, "
    f"bfs_percentage={default_args.SAMPLING_BFS_PERCENTAGE}, "
    f"random_percentage={default_args.SAMPLING_RANDOM_PERCENTAGE}, "
    f"max_time={default_args.SAMPLING_MAX_TIME}, "
    f"mem_limit_mb={default_args.SAMPLING_MEM_LIMIT_MB}, "
    f"facts_file={FACTS_FILE}, "
    f"random_seed={default_args.DEFAULT_SEED})"
    f"], "
    f"sai={default_args.SAMPLING_SAI}, "
    f"sui={default_args.SAMPLING_SUI}, "
    f'random_seed={default_args.DEFAULT_SEED})"'
)
TRAIN_COMMANDLINE = (
    f"/train.py {SAMPLE_FILE} --train-folder {TRAIN_FOLDER} --facts-file {FACTS_FILE}"
)
TEST_COMMANDLINE = (
    f"/test.py {TRAIN_FOLDER} {PROBLEM_FILE} --plan-file {PLAN_FILE} --facts-file {FACTS_FILE}"
    + (f" --domain-pddl {DOMAIN_FILE}" if DOMAIN_FILE else "")
)

for commandline in [SAMPLING_COMMANDLINE, TRAIN_COMMANDLINE, TEST_COMMANDLINE]:
    os.system(commandline)
