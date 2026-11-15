from pathlib import Path

import pytest

from tinytalk.config import load_settings


def test_loads_tinytauk_backend(monkeypatch):
    monkeypatch.setenv("TINYTALK_BACKEND", "tinytauk")
    monkeypatch.setenv("TINYTALK_TINYTAUK_CHARS_PER_SECOND", "12.5")

    settings = load_settings()

    assert settings.backend == "tinytauk"
    assert settings.tinytauk_chars_per_second == 12.5

    monkeypatch.setenv("TINYTALK_TINYTAUK_PROFILE", "/tmp/tinytauk.toml")
    assert load_settings().tinytauk_profile == Path("/tmp/tinytauk.toml")

    monkeypatch.setenv("TINYTALK_TINYTAUK_PROFILE", "")
    assert load_settings().tinytauk_profile is None


def test_loads_omnivoice_backend(monkeypatch):
    monkeypatch.setenv("TINYTALK_BACKEND", "omnivoice")
    monkeypatch.setenv("TINYTALK_OMNIVOICE_MODEL", "example/OmniVoice")
    monkeypatch.setenv("TINYTALK_OMNIVOICE_DEVICE", "cpu")
    monkeypatch.setenv("TINYTALK_OMNIVOICE_LANGUAGE", "en")
    monkeypatch.setenv("TINYTALK_OMNIVOICE_REF_AUDIO", "/tmp/ref.wav")
    monkeypatch.setenv("TINYTALK_OMNIVOICE_REF_TEXT", "/tmp/ref.txt")

    settings = load_settings()

    assert settings.backend == "omnivoice"
    assert settings.omnivoice_model == "example/OmniVoice"
    assert settings.omnivoice_device == "cpu"
    assert settings.omnivoice_language == "en"
    assert settings.omnivoice_ref_audio == Path("/tmp/ref.wav")
    assert settings.omnivoice_ref_text == Path("/tmp/ref.txt")


def test_empty_omnivoice_reference_env_is_unset(monkeypatch):
    monkeypatch.setenv("TINYTALK_OMNIVOICE_REF_AUDIO", "")
    monkeypatch.setenv("TINYTALK_OMNIVOICE_REF_TEXT", "   ")

    settings = load_settings()

    assert settings.omnivoice_ref_audio is None
    assert settings.omnivoice_ref_text is None


def test_rejects_unknown_backend(monkeypatch):
    monkeypatch.setenv("TINYTALK_BACKEND", "unknown")

    with pytest.raises(ValueError, match="TINYTALK_BACKEND"):
        load_settings()


def test_loads_inflect_backend(monkeypatch):
    monkeypatch.setenv("TINYTALK_BACKEND", "inflect")
    monkeypatch.setenv("TINYTALK_INFLECT_PACKAGE", "/tmp/inflect-mini-v3")
    monkeypatch.setenv("TINYTALK_INFLECT_VOICE", "james")
    monkeypatch.setenv("TINYTALK_INFLECT_PROFILE", "/tmp/bert-profile.json")

    settings = load_settings()

    assert settings.backend == "inflect"
    assert settings.inflect_package == Path("/tmp/inflect-mini-v3")
    assert settings.inflect_voice == "james"
    assert settings.inflect_profile == Path("/tmp/bert-profile.json")


def test_inflect_profile_defaults_unset(monkeypatch):
    settings = load_settings()

    assert settings.inflect_package == Path("/var/lib/tinytalk/inflect-mini-v3")
    assert settings.inflect_voice == "cassidy"
    assert settings.inflect_profile is None

    monkeypatch.setenv("TINYTALK_INFLECT_PROFILE", "")
    assert load_settings().inflect_profile is None
