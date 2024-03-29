import logging
import os
import re
import sys
from time import time

from . import aliases
from . import arguments
from . import cleanup
from . import run_components
from . import __version__


def main():
    args = arguments.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(levelname)-8s %(message)s",
                        stream=sys.stdout)
    logging.debug("processed args: %s" % args)

    if args.version:
        print(__version__)
        sys.exit()

    if args.show_aliases:
        aliases.show_aliases()
        sys.exit()

    if args.cleanup:
        cleanup.cleanup_temporary_files(args)
        sys.exit()

    exitcode = None
    for component in args.components:
        if component == "translate":
            translate_start_time = time()
            (exitcode, continue_execution) = run_components.run_translate(args)
            translate_time = time() - translate_start_time
        elif component == "search":
            search_args_idx = next((i for i, s in enumerate(args.search_options) if "max_time=" in s), None)
            if search_args_idx:
                args.search_options[search_args_idx] = re.sub(
                    r"(max_time=)(\d+)", lambda m: m.group(1) + str(int(m.group(2)) - round(translate_time)),
                    args.search_options[search_args_idx]
                )
            (exitcode, continue_execution) = run_components.run_search(args)
            if not args.keep_sas_file:
                print("Remove intermediate file {}".format(args.sas_file))
                os.remove(args.sas_file)
        elif component == "validate":
            (exitcode, continue_execution) = run_components.run_validate(args)
        else:
            assert False, "Error: unhandled component: {}".format(component)
        print("{component} exit code: {exitcode}".format(**locals()))
        print()
        if not continue_execution:
            print("Driver aborting after {}".format(component))
            break
    # Exit with the exit code of the last component that ran successfully.
    # This means for example that if no plan was found, validate is not run,
    # and therefore the return code is that of the search.
    sys.exit(exitcode)


if __name__ == "__main__":
    main()
