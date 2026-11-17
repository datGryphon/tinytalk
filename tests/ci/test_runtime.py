"""Validate all backends together against the canonical project metadata."""

import importlib.metadata as metadata
import importlib.util
import json
import sys
import tomllib
from pathlib import Path
from typing import get_args

import pytest
from packaging.requirements import Requirement

from tinytalk.config import Backend, Settings
from tinytalk.engine import SpeechBackend, SpeechEngine, create_engine


PROJECT = tomllib.loads(Path("pyproject.toml").read_text())
BACKENDS = get_args(Backend)


def _check_requirement(requirement: str) -> None:
    req = Requirement(requirement)
    name = req.name
    actual = metadata.version(name)
    specifier = req.specifier
    assert not specifier or specifier.contains(actual), (name, actual, str(specifier))
    if req.url:
        wanted_url = req.url
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


@pytest.mark.parametrize("extra", PROJECT["project"]["optional-dependencies"])
def test_dependencies_match_project(extra):
    requirements = PROJECT["project"]["optional-dependencies"][extra]
    for requirement in requirements:
        _check_requirement(requirement)


@pytest.mark.parametrize("backend", BACKENDS)
def test_engine_composes_speech_backend(backend):
    settings = Settings(backend=backend)
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
        _check_requirement(requirement)


@pytest.mark.parametrize("package", ["torchtune", "torchao"])
def test_removed_runtime_dependencies_are_absent(package):
    assert importlib.util.find_spec(package) is None
