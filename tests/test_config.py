import pytest

from tinytalk.config import load_settings


def test_loads_tinytauk_backend(monkeypatch):
    monkeypatch.setenv("TINYTALK_BACKEND", "tinytauk")
    monkeypatch.setenv("TINYTALK_TINYTAUK_CHARS_PER_SECOND", "12.5")

    settings = load_settings()

    assert settings.backend == "tinytauk"
    assert settings.tinytauk_chars_per_second == 12.5


def test_rejects_unknown_backend(monkeypatch):
    monkeypatch.setenv("TINYTALK_BACKEND", "unknown")

    with pytest.raises(ValueError, match="TINYTALK_BACKEND"):
        load_settings()
