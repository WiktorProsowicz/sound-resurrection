"""This file contains configuration unit tests common for existing suites."""

from typing import List
import itertools
import pathlib

import pytest


@pytest.hookimpl
def pytest_collection_modifyitems(items: List[pytest.Item]):
    """Modifies collected tests to be run in a specific order.

    Args:
        items: List of collected tests.
    """

    grouped_by_path = itertools.groupby(items, lambda item: item.path)
    grouped_by_path = {path: list(items) for path, items in grouped_by_path}

    sorted_items: List[pytest.Item] = []
    handled_paths: List[pathlib.Path] = []

    def add_path_to_items(path: pathlib.Path):
        sorted_items.extend(grouped_by_path[path])

    def get_path_by_dependency_name(name: str) -> pathlib.Path:
        for item in items:
            for mark in item.iter_markers(name='dependency'):
                if mark.kwargs.get('name', None) == name:
                    return item.path

    def handle_path(path: pathlib.Path):

        if path in handled_paths:
            return

        handled_paths.append(path)

        for item in grouped_by_path[path]:
            for mark in item.iter_markers(name='dependency'):
                if 'depends' in mark.kwargs:
                    for dependency_name in mark.kwargs['depends']:
                        dependency_path = get_path_by_dependency_name(dependency_name)
                        handle_path(dependency_path)

        add_path_to_items(path)

    if grouped_by_path:

        for path in grouped_by_path:
            handle_path(path)

        items.clear()
        items.extend(sorted_items)
