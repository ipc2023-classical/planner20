"""
Management of file creation and deletion.
"""

import os
from argparse import Namespace
from json import dump
from shutil import rmtree


def save_json(filename: str, data: list):
    """
    Saves a file with the given filename and data as a JSON file.
    """
    with open(filename, "w") as f:
        dump(data, f, indent=4)


def create_train_directory(args: Namespace) -> str:
    """
    Creates training directory according to current configuration.
    """
    sep = "."
    dirname = args.train_folder
    if not dirname:
        dirname = f"{args.output_folder}/nfd_train{sep}{args.samples.split('/')[-1]}"
        if args.seed != -1:
            dirname += f"{sep}ns{args.seed}"
    if os.path.exists(dirname):
        rmtree(dirname)
    os.makedirs(dirname)
    os.makedirs(f"{dirname}/models")
    return dirname


def create_test_directory(args):
    """
    Creates testing directory according to current configuration.
    """
    sep = "."
    tests_folder = args.train_folder / "tests"
    if not os.path.exists(tests_folder):
        os.makedirs(tests_folder)
    dirname = f"{tests_folder}/nfd_test"
    if os.path.exists(dirname):
        i = 2
        while os.path.exists(f"{dirname}{sep}{i}"):
            i += 1
        dirname = dirname + f"{sep}{i}"
    os.makedirs(dirname)
    return dirname


def remove_temporary_files(directory: str):
    """
    Removes `output.sas` and `defaults.txt` files.
    """

    def remove_file(file: str):
        if os.path.exists(file):
            os.remove(file)

    remove_file(f"{directory}/output.sas")
    remove_file(f"{directory}/sas_plan")
    remove_file(f"{directory}/defaults.txt")


def create_defaults_file(
    pddl_file: str, facts_file: str, output_folder: str = "."
) -> str:
    """
    Create defaults file for `pddl_file`.
    For all fact in facts_file, 1 if fact \in initial_state(pddl_file) else 0.
    """

    init = None
    with open(pddl_file, "r") as f:
        pddl_text = f.read().lower()
        init = pddl_text.split(":init")[1].split(":goal")[0]

    with open(facts_file, "r") as f:
        facts = f.read().strip().split(";")

    # Atom on(i, a) -> (on i a)
    modified_facts = []
    for fact in facts:
        f = fact.replace("Atom ", "")  # Atom on(i, a) -> on(i, a)
        f = f.replace(", ", ",").replace(",", " ")  # on(i, a) -> on(i a)
        f = f"({f.split('(')[0]} {f.split('(')[1]}"  # on(i a) -> (on i a)
        f = f.replace(" )", ")")  # facts without objects (handempty ) -> (handempty)
        modified_facts.append(f)

    defaults = []
    for fact in modified_facts:
        value = "1" if fact in init else "0"
        defaults.append(value)

    if not defaults:
        raise Exception("get_defaults: defaults is empty")

    output_file = output_folder + "/defaults.txt"
    with open(output_file, "w") as f:
        f.write(";".join(defaults) + "\n")
    return output_file


def create_fake_samplefile(sample_filename: str, facts_file: str):
    with open(facts_file, "r") as f:
        num_atoms = len(f.read().strip().split(";"))

    with open(sample_filename, "w") as f:
        f.write(f"{0};{num_atoms * '0'}")
