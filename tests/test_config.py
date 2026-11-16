from pathlib import Path

import pytest

from tinytalk.config import Settings, load_settings


def test_loads_silencio2_backend(monkeypatch):
    monkeypatch.setenv("TINYTALK_BACKEND", "silencio2")
    monkeypatch.setenv("TINYTALK_SILENCIO2_VOICE", "/local/reference.wav")
    settings = load_settings()
    assert settings.backend == "silencio2"
    assert settings.silencio2_voice == "/local/reference.wav"


def test_loads_chatterbox_nano_backend(monkeypatch):
    monkeypatch.setenv("TINYTALK_BACKEND", "chatterbox_nano")
    monkeypatch.setenv("TINYTALK_CHATTERBOX_NANO_REF_AUDIO", "/local/reference.wav")
    settings = load_settings()
    assert settings.backend == "chatterbox_nano"
    assert settings.chatterbox_nano_ref_audio == Path("/local/reference.wav")
    monkeypatch.setenv("TINYTALK_CHATTERBOX_NANO_REF_AUDIO", "")
    assert load_settings().chatterbox_nano_ref_audio is None


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


def test_env_values_are_validated_and_typed(monkeypatch):
    monkeypatch.setenv("TINYTALK_MAX_RETRIES", "3")
    monkeypatch.setenv("TINYTALK_WATERMARK", "true")
    monkeypatch.setenv("TINYTALK_TEMPERATURE", "0.75")
    monkeypatch.setenv("TINYTALK_MAX_CHARS_PER_CHUNK", "")
    settings = load_settings()
    assert settings.max_retries == 3
    assert settings.watermark is True
    assert settings.temperature == 0.75
    assert settings.max_chars_per_chunk == Settings().max_chars_per_chunk
    assert Settings().max_retries == 2


@pytest.mark.parametrize(("name", "value"), [
    ("MAX_RETRIES", "-1"),
    ("MAX_CHARS_PER_CHUNK", "0"),
    ("TINYTAUK_CHARS_PER_SECOND", "0"),
    ("PORT", "not an integer"),
    ("WATERMARK", "not a boolean"),
])
def test_invalid_env_values_are_rejected(monkeypatch, name, value):
    env_name = f"TINYTALK_{name}"
    monkeypatch.setenv(env_name, value)
    with pytest.raises(ValueError, match=env_name):
        load_settings()
