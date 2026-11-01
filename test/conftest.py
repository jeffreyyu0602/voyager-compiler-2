"""Shared pytest setup for the test modules under ``test/``.

A test marked ``slow`` runs only under ``pytest --run-slow``.
"""

import pytest
import torch


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        help="also run the tests marked slow",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: a test of minutes; runs only under --run-slow"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip = pytest.mark.skip(reason="slow: pass --run-slow to run it")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True, scope="module")
def grad_mode(request):
    """Run a module's tests with grad off if it sets ``GRAD_ENABLED = False``
    and on otherwise, so no module's grad mode reaches another's."""
    with torch.set_grad_enabled(getattr(request.module, "GRAD_ENABLED", True)):
        yield
