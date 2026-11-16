"""Validate the installed backend against the canonical project metadata."""

import importlib.metadata as metadata
import importlib.util
import json
import os
import sys
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement


PROJECT = tomllib.loads(Path("pyproject.toml").read_text())
BACKEND = os.environ["TINYTALK_BACKEND"]


def _name(requirement: str) -> str:
    return Requirement(requirement).name


def _installed(name: str) -> bool:
    try:
        metadata.version(name)
    except metadata.PackageNotFoundError:
        return False
    return True


def _check_requirement(requirement: str, *, required: bool) -> None:
    name = _name(requirement)
    if not required and not _installed(name):
        return

    actual = metadata.version(name)
    specifier = Requirement(requirement).specifier
    assert not specifier or specifier.contains(actual), (name, actual, str(specifier))
    if "==" in requirement:
        wanted = requirement.split("==", 1)[1].strip()
        assert actual == wanted or actual.split("+", 1)[0] == wanted, (
            name,
            actual,
            wanted,
        )

    if " @ " in requirement:
        wanted_url = requirement.split(" @ ", 1)[1].strip()
        direct_url = metadata.distribution(name).read_text("direct_url.json")
        assert direct_url is not None, f"Missing direct_url.json for {name}"
        actual_url = json.loads(direct_url)["url"]
        assert actual_url == wanted_url or wanted_url in actual_url, (
            name,
            actual_url,
            wanted_url,
        )


def test_python_version():
    assert sys.version_info[:2] == (3, 13)


def test_tinytalk_distribution_is_installed():
    import tinytalk

    assert metadata.version("tinytalk") == tinytalk.__version__


def test_backend_dependencies_match_project():
    requirements = PROJECT["project"]["optional-dependencies"][BACKEND]
    for requirement in requirements:
        _check_requirement(requirement, required=True)


def test_engine_composes_speech_backend():
    from tinytalk.config import Settings
    from tinytalk.engine import SpeechBackend, SpeechEngine, create_engine

    settings = Settings(backend=BACKEND)
    engine = create_engine(settings)
    assert type(engine) is SpeechEngine
    assert isinstance(engine.backend, SpeechBackend)
    assert engine.settings is settings
    assert engine.loaded is False
    assert engine.sample_rate == 24_000
    assert engine.model_name

    for instance in (engine, engine.backend):
        declared = {
            name
            for cls in type(instance).__mro__
            for name in cls.__dict__.get("__annotations__", {})
        }
        assert vars(instance).keys() <= declared


def test_installed_overrides_match_project():
    for requirement in PROJECT["tool"]["uv"]["override-dependencies"]:
        _check_requirement(requirement, required=False)


@pytest.mark.parametrize("package", ["torchtune", "torchao"])
def test_removed_runtime_dependencies_are_absent(package):
    assert importlib.util.find_spec(package) is None
