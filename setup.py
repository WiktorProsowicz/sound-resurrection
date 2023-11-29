# -*- coding: utf-8 -*-
"""Script containing functions for managing the project workspace.

Available functions and corresponding arguments are described in the
main function as well as in the doc strings of the functions.
"""
import argparse
import logging
import os
import pathlib
import shutil
import subprocess
import venv
from os import environ
from os import path

HOME_PATH = pathlib.Path(__file__).absolute().parent.as_posix()


def run_tests() -> None:
    """Runs available tests from the 'tests' directory."""

    tests_path = path.join(HOME_PATH, 'tests')
    src_path = path.join(HOME_PATH, 'src')

    test_results_path = path.join(HOME_PATH, 'test_results')

    coverage_data_file = path.join(test_results_path, '.coverage')
    coverage_stats_dir = path.join(test_results_path, 'coverage_stats')

    if not os.path.exists(test_results_path):
        os.makedirs(test_results_path)

    # Cleaning test_results.
    for root, dirs, files in os.walk(test_results_path):

        for file in files:
            os.remove(os.path.join(root, file))
        for directory in dirs:
            shutil.rmtree(os.path.join(root, directory))

    current_env = environ.copy()
    current_env['PYTHONPATH'] = f"{src_path}:{current_env.get('PYTHONPATH', '')}"
    current_env['TEST_RESOURCES'] = path.join(HOME_PATH, 'tests', 'res')
    current_env['TEST_RESULTS'] = path.join(HOME_PATH, 'test_results')

    command = (f'python3 -m coverage run --data-file={coverage_data_file}'
               f' -m pytest --import-mode=prepend -s {tests_path}')

    subprocess.run(command.split(), check=False, env=current_env)

    command = (f'python3 -m coverage html --data-file={coverage_data_file}'
               f' --directory={coverage_stats_dir}')

    subprocess.run(command.split(), check=False, env=current_env)


def setup_venv() -> None:
    """Sets up the virtual environment."""

    venv_path = path.join(HOME_PATH, 'venv')

    if os.path.exists(venv_path):
        logging.warning(
            "Directory '%s' already exists. If you are sure you want to" +
            ' replace it with a new environment, delete it and run again.', venv_path)
        return

    venv.create(venv_path, with_pip=True, upgrade_deps=True, clear=False)

    logging.info(
        "Successfully created a virtual environment at directory '%s'", venv_path)
    logging.info(
        "You can now activate the environment with 'source ./venv/bin/activate'.")
    logging.info(
        "Then type 'python3 -m pip install -r requirements.txt' to install dependencies.")
    logging.info("Then type 'deactivate' to deactivate the environment.")


def run_script(script_name: str, *args) -> None:
    """Runs a script from the 'scripts' directory.

    Args:
        script_name (str): Name of the script to be run.
        args (list): List of values to be passed to the script via command line arguments.
    """

    script_path = path.join(HOME_PATH, 'scripts', script_name)

    if not os.path.exists(script_path):
        logging.error("Couldn't find the script '%s'.", script_path)
        return

    src_path = path.join(HOME_PATH, 'src')
    current_env = os.environ.copy()
    current_env['PYTHONPATH'] = f"{src_path}:{current_env.get('PYTHONPATH', '')}"

    logging.info("Running script '%s'.", script_path)

    command = f"python3 {script_path} {' '.join(args)}"

    subprocess.run(command.split(), check=False, env=current_env)


def main(function: str, *args) -> None:
    """Main function delegating the flow to other ones.

    Args:
        function (str): Name of the function to be called.
    """

    for available_func in [setup_venv, run_tests, run_script]:
        if available_func.__name__ == function:
            available_func(*args)  # type: ignore
            return

    raise RuntimeError(f"Couldn't find the function '{function}'.")


if __name__ == '__main__':

    FUNCTION_DESCRIPTIONS_LIST = [f'{func.__name__}: {func.__doc__.splitlines()[0]}'  # type: ignore
                                  for func in [setup_venv, run_tests, run_script]]

    FUNCTION_DESCRIPTIONS = '\n'.join(FUNCTION_DESCRIPTIONS_LIST)

    PROGRAM_DESC = ('Script contains functions helping with project management.\n' +
                    'Available functions:\n\n' +
                    f'{FUNCTION_DESCRIPTIONS}')

    arg_parser = argparse.ArgumentParser(
        description=PROGRAM_DESC, formatter_class=argparse.RawDescriptionHelpFormatter)

    arg_parser.add_argument(
        'function_name', help='name of the function to be used')
    arg_parser.add_argument(
        'args', nargs='*', help='positional arguments for the function')

    arguments = arg_parser.parse_args()

    logging.root.setLevel(logging.INFO)

    main(arguments.function_name, *arguments.args)
