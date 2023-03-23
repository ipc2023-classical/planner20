#!/usr/bin/env python3

import logging
import sys
import os

from src.pytorch.fast_downward_api import solve_instance_with_fd_nh
from src.pytorch.log import setup_full_logging
from src.pytorch.utils.helpers import get_models_from_train_folder
from src.pytorch.utils.file_helpers import (
    create_test_directory,
    remove_temporary_files,
)
from src.pytorch.utils.log_helpers import (
    logging_test_config,
    logging_test_statistics,
)
from src.pytorch.utils.parse_args import get_test_args

_log = logging.getLogger(__name__)


def test_main(args):
    _log.info("Starting TEST procedure...")

    dirname = create_test_directory(args)
    setup_full_logging(dirname)

    if args.shared_timers:
        nfd_log = os.path.abspath(f"{args.train_folder}/nfd.log")
        if os.path.exists(nfd_log):
            with open(nfd_log, "r") as f:
                for line in f.readlines():
                    if "Elapsed time: " in line:
                        times = line.split()[-1][:-1].split("/")
                        extra = int(float(times[1]) - float(times[0]))
                        args.max_search_time += extra
                        break

    if args.heuristic == "nn":
        models = get_models_from_train_folder(args.train_folder, args.test_model)
    else:
        models = [""]

    cmd_line = " ".join(sys.argv)
    logging_test_config(args, dirname, cmd_line)

    for model_path in models:
        output = {}
        for j, problem_pddl in enumerate(args.problem_pddls):
            _log.info(
                f'Solving instance "{problem_pddl}" ({j+1}/{len(args.problem_pddls)})'
            )
            output[problem_pddl] = solve_instance_with_fd_nh(
                domain_pddl=args.domain_pddl,
                problem_pddl=problem_pddl,
                traced_model=model_path,
                search_algorithm=args.search_algorithm,
                heuristic=args.heuristic,
                heuristic_multiplier=args.heuristic_multiplier,
                time_limit=args.max_search_time,
                memory_limit=args.max_search_memory,
                max_expansions=args.max_expansions,
                facts_file=args.facts_file,
                defaults_file=args.defaults_file,
                plan_file=args.plan_file,
                save_log_to=dirname,
                save_log_bool=args.downward_logs,
                unit_cost=args.unit_cost,
            )
            _log.info(problem_pddl)
            for var in output[problem_pddl]:
                _log.info(f" | {var}: {output[problem_pddl][var]}")

        model_file = model_path.split("/")[-1]
        logging_test_statistics(args, dirname, model_file, output)
        _log.info(f"Test on model {model_file} complete!")

    remove_temporary_files(dirname)
    _log.info("Test complete!")


if __name__ == "__main__":
    test_main(get_test_args())
