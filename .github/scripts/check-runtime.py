"""Assert that the active CI interpreter has the qualified backend runtime."""

import importlib.metadata as metadata
import importlib.util
import json
import sys

BACKENDS = ("neutts", "omnivoice", "tinytauk")
NEUCODEC_COMMIT = "6954b1f877963e19177b43be1ef56b1818990d31"


def check_version(package: str, expected: str) -> None:
    actual = metadata.version(package)
    print(f"{package}: {actual}", flush=True)
    assert actual.split("+")[0] == expected, (package, actual, expected)
    if package in ("torch", "torchaudio"):
        assert actual.endswith("+cpu"), (package, actual)


def check_source(package: str, expected_fragment: str) -> None:
    source_info = metadata.distribution(package).read_text("direct_url.json")
    assert source_info is not None, f"Missing direct URL for {package}"
    url = json.loads(source_info)["url"]
    print(f"{package} source: {url}", flush=True)
    assert expected_fragment in url, (package, url)


def main(backend: str) -> None:
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend: {backend}")
    assert sys.version_info[:2] == (3, 13), sys.version

    expected = {
        "torch": "2.11.0",
        "torchaudio": "2.11.0",
        "transformers": "5.17.0",
    }
    if backend == "neutts":
        expected.update(neutts="1.4.1", neucodec="0.0.6")
    elif backend == "omnivoice":
        expected["omnivoice"] = "0.2.1"
    else:
        expected["tinytauk"] = "0.2.0"

    for package, wanted in expected.items():
        check_version(package, wanted)

    if backend in ("neutts", "tinytauk"):
        for removed in ("torchtune", "torchao"):
            assert importlib.util.find_spec(removed) is None, removed

    if backend == "neutts":
        check_source("neucodec", NEUCODEC_COMMIT)
    elif backend == "tinytauk":
        check_source("tinytauk", "/v0.2.0.tar.gz")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: check-runtime.py {neutts|omnivoice|tinytauk}")
    main(sys.argv[1])
