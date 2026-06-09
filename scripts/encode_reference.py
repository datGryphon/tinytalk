"""Encode a reference WAV into a NeuCodec code tensor (.pt).

Adapted from neuphonic/neutts-air examples/encode_reference.py.

The runtime server loads only the saved .pt and the ONNX int8 decoder.
Run this once on any machine with enough RAM (~500 MB). The encoder
(PyTorch NeuCodec) is not needed at serve time.

Usage:
    pip install neucodec librosa torch
    python scripts/encode_reference.py ref_audio.wav ref_codes.pt
"""

import sys
from pathlib import Path

import torch
from librosa import load
from neucodec import NeuCodec


def main(wav_path: str, out_path: str) -> None:
    wav_file = Path(wav_path)
    if not wav_file.exists():
        sys.exit(f"not found: {wav_file}")

    print("loading NeuCodec")
    codec = NeuCodec.from_pretrained("neuphonic/neucodec")
    codec.eval().to("cpu")

    print(f"reading {wav_file} at 16 kHz mono")
    wav, _ = load(str(wav_file), sr=16000, mono=True)
    wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)

    print("encoding")
    ref_codes = codec.encode_code(audio_or_path=wav_tensor).squeeze(0).squeeze(0)

    out = Path(out_path)
    print(f"saving {out}  shape={tuple(ref_codes.shape)}  dtype={ref_codes.dtype}")
    torch.save(ref_codes, out)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: encode_reference.py <ref_audio.wav> <ref_codes.pt>")
    main(sys.argv[1], sys.argv[2])
