"""Runtime compatibility checks. Backend requirements live in runtime-contracts.json."""

import importlib.metadata as metadata
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest


CONTRACTS = json.loads(Path(__file__).with_name("runtime-contracts.json").read_text())


@pytest.fixture(scope="module")
def contract():
    backend = os.environ.get("TINYTALK_BACKEND")
    if backend not in CONTRACTS["backends"]:
        pytest.fail(f"No runtime contract for TINYTALK_BACKEND={backend!r}")
    return CONTRACTS["backends"][backend]


def test_python_version():
    assert list(sys.version_info[:2]) == CONTRACTS["python"]


def test_distribution_versions(contract):
    expected = CONTRACTS["shared"] | contract.get("distributions", {})
    for package, wanted in expected.items():
        actual = metadata.version(package)
        print(f"{package}: {actual}")
        assert actual == wanted or (
            "+" not in wanted and actual.split("+")[0] == wanted
        ), (package, actual, wanted)


def test_direct_sources(contract):
    for package, expected_fragment in contract.get("source_fragments", {}).items():
        source_info = metadata.distribution(package).read_text("direct_url.json")
        assert source_info is not None, f"Missing direct_url.json for {package}"
        url = json.loads(source_info)["url"]
        assert expected_fragment in url, (package, url)


def test_forbidden_dependencies_absent(contract):
    for package in contract.get("absent", []):
        assert importlib.util.find_spec(package) is None, package
