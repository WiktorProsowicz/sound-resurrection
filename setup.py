# -*- coding: utf-8 -*-
"""Script containing functions for managing the project workspace.

Available functions and corresponding arguments are described in the
main function as well as in the doc strings of the functions.
"""
import argparse
import os
import pathlib
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

    if not path.exists(test_results_path):
        os.makedirs(test_results_path)

    current_env = environ.copy()
    current_env['PYTHONPATH'] = (src_path +
                                 ':' + current_env.get('PYTHONPATH', ''))
    current_env['TEST_RESOURCES'] = path.join(HOME_PATH, 'tests', 'res')
    current_env['TEST_RESULTS'] = path.join(HOME_PATH, 'test_results')

    command = (f'python3 -m coverage run --data-file={coverage_data_file}'
               f' -m pytest --import-mode=prepend {tests_path}')

    subprocess.run(command.split(), check=False, env=current_env)

    command = (f'python3 -m coverage html --data-file={coverage_data_file}'
               f'--directory={coverage_stats_dir}')

    subprocess.run(command.split(), check=False, env=current_env)


def setup_venv() -> None:
    """Sets up the virtual environment."""

    venv_path = path.join(HOME_PATH, 'venv')

    venv.create(venv_path, with_pip=True, upgrade_deps=True)

    print(
        f"Successfully created a virtual environment at directory '{venv_path}'")
    print("You can now activate the environment with 'source ./venv/bin/activate'.")
    print("Then type 'python3 -m pip install -r requirements.txt' to install dependencies.")
    print("Then type 'deactivate' to deactivate the environment.")


def main(function: str, *args) -> None:
    """Main function delegating the flow to other ones.

    Args:
        function (str): Name of the function to be called.
    """

    for available_func in [setup_venv, run_tests]:
        if available_func.__name__ == function:
            available_func(*args)
            return

    raise RuntimeError(f"Couldn't find the function '{function}'.")


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        'function_name', help='name of the function to be used')
    arg_parser.add_argument(
        'args', nargs='*', help='positional arguments for the function')

    arguments = arg_parser.parse_args()

    main(arguments.function_name, *arguments.args)
