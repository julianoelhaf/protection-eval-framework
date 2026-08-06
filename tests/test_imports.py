"""Import smoke test for the whole ``fcl_psp`` package.

Walks every submodule and imports it. Catches breakage such as stale imports
(e.g. the historical ``dl_psp`` -> ``fcl_psp`` rename) and packaging regressions
that would otherwise only surface at runtime.
"""

import importlib
import pkgutil

import pytest

import fcl_psp

MODULES = sorted(info.name for info in pkgutil.walk_packages(fcl_psp.__path__, prefix="fcl_psp."))


def test_package_has_submodules():
    assert MODULES, "no submodules discovered under fcl_psp"


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name):
    importlib.import_module(module_name)
