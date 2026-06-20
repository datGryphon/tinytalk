{ self }:
{ config, lib, pkgs, ... }:

let
  cfg = config.services.tinytalk;
  python = pkgs.python312;
  pythonTarget = "/var/lib/tinytalk/python";
  runtimePackages = [
    "torch==2.11.0+cpu"
    "torchaudio==2.11.0+cpu"
    "neutts[all]"
    "fastapi"
    "uvicorn[standard]"
    "spacy"
    "librosa"
    "praat-parselmouth"
  ];
  prestart = pkgs.writeShellScript "tinytalk-prestart.sh" (
    builtins.replaceStrings
      [ "@python@" "@uv@" ]
      [ "${python}" "${pkgs.uv}" ]
      (builtins.readFile ./tinytalk-prestart.sh)
  );
in
{
  options.services.tinytalk = {
    enable = lib.mkEnableOption "OpenAI-compatible NeuTTS tinytalk server";

    model = lib.mkOption {
      type = lib.types.str;
      default = "neuphonic/neutts-nano-q4-gguf";
      description = "NeuTTS backbone repository or local GGUF path.";
    };

    codec = lib.mkOption {
      type = lib.types.str;
      default = "neuphonic/neucodec-onnx-decoder-int8";
      description = "NeuCodec repository or ONNX decoder path.";
    };

    backboneDevice = lib.mkOption {
      type = lib.types.enum [ "gpu" "cpu" ];
      default = "cpu";
      description = "NeuTTS GGUF backbone device.";
    };

    refCodes = lib.mkOption {
      type = lib.types.path;
      default = "/var/lib/tinytalk/ref_codes.pt";
      description = "Pre-encoded NeuTTS reference-code file.";
    };

    refText = lib.mkOption {
      type = lib.types.path;
      default = "/var/lib/tinytalk/ref_text.txt";
      description = "Reference transcript text file matching refCodes.";
    };

    host = lib.mkOption {
      type = lib.types.str;
      default = "0.0.0.0";
      description = "Bind host for uvicorn.";
    };

    port = lib.mkOption {
      type = lib.types.port;
      default = 9002;
      description = "Bind port for uvicorn.";
    };

    maxCharsPerChunk = lib.mkOption {
      type = lib.types.ints.positive;
      default = 180;
      description = "Maximum text characters sent to one NeuTTS infer() call.";
    };

    interChunkSilenceMs = lib.mkOption {
      type = lib.types.ints.unsigned;
      default = 60;
      description = "Zero-audio pause inserted between synthesized chunks.";
    };

    temperature = lib.mkOption {
      type = lib.types.float;
      default = 1.0;
      description = "Sampling temperature for the NeuTTS backbone (1.0 = NeuTTS default). Lower is more stable with less looping. Tune per voice or deployment.";
    };

    repeatPenalty = lib.mkOption {
      type = lib.types.float;
      default = 1.0;
      description = "Repeat penalty for the NeuTTS backbone, discouraging looped/duplicated speech.";
    };

    repeatPenaltyRerollStep = lib.mkOption {
      type = lib.types.float;
      default = 0.10;
      description = "Incremental repeat-penalty added per retry attempt in the per-chunk reroll loop.";
    };

    maxRetries = lib.mkOption {
      type = lib.types.ints.between 0 100;
      default = 2;
      description = "Maximum number of retry attempts per chunk.";
    };

    werEndpoint = lib.mkOption {
      type = lib.types.str;
      default = "";
      description = "Base URL for WER transcription evaluation. Empty string disables live transcription and falls back to chunk confidence.";
    };

    werThreshold = lib.mkOption {
      type = lib.types.float;
      default = 0.25;
      description = "WER threshold above which a chunk is accepted. Lower scores mean the chunk is kept as-is from the reroll loop.";
    };

    watermark = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = ''
        Enable perth audio watermarking. Disabled by default because the perth
        watermarker leaks ~44 MB per synthesis call (PyTorch CPU caching allocator
        retains freed tensors). Only enable if you need provenance watermarks.
      '';
    };

    memoryHigh = lib.mkOption {
      type = lib.types.str;
      default = "5000M";
      description = ''
        systemd MemoryHigh soft cap for the service. Safety net against the
        perth-watermarker memory leak; harmless headroom when watermarking is
        disabled.
      '';
    };

    memoryMax = lib.mkOption {
      type = lib.types.str;
      default = "6000M";
      description = ''
        systemd MemoryMax hard cap. With Restart=on-failure (set below) the
        service is recycled if it hits this, bounding the perth-watermarker
        leak.
      '';
    };

    runtimeIndexUrl = lib.mkOption {
      type = lib.types.str;
      default = "https://download.pytorch.org/whl/cpu";
      description = "Primary Python package index used by the runtime bootstrap.";
    };

    runtimeExtraIndexUrls = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ "https://pypi.org/simple" ];
      description = "Additional Python package indexes used by the runtime bootstrap.";
    };

    runtimePackages = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = runtimePackages;
      description = "Python package specs installed by the runtime bootstrap.";
    };
  };

  config = lib.mkIf cfg.enable {
    users.users.tinytalk = {
      isSystemUser = true;
      group = "tinytalk";
      home = "/var/lib/tinytalk";
    };
    users.groups.tinytalk = { };

    systemd.services.tinytalk = {
      description = "tinytalk OpenAI-compatible NeuTTS server";
      after = [ "network-online.target" ];
      wants = [ "network-online.target" ];
      wantedBy = [ "multi-user.target" ];

      environment = {
        TINYTALK_MODEL = cfg.model;
        TINYTALK_CODEC = cfg.codec;
        TINYTALK_BACKBONE_DEVICE = cfg.backboneDevice;
        TINYTALK_REF_CODES = toString cfg.refCodes;
        TINYTALK_REF_TEXT = toString cfg.refText;
        TINYTALK_HOST = cfg.host;
        TINYTALK_PORT = toString cfg.port;
        TINYTALK_MAX_CHARS_PER_CHUNK = toString cfg.maxCharsPerChunk;
        TINYTALK_INTER_CHUNK_SILENCE_MS = toString cfg.interChunkSilenceMs;
        TINYTALK_TEMPERATURE = toString cfg.temperature;
        TINYTALK_REPEAT_PENALTY = toString cfg.repeatPenalty;
        TINYTALK_REPEAT_PENALTY_REROLL_STEP = toString cfg.repeatPenaltyRerollStep;
        TINYTALK_MAX_RETRIES = toString cfg.maxRetries;
        TINYTALK_WER_ENDPOINT = cfg.werEndpoint;
        TINYTALK_WER_THRESHOLD = toString cfg.werThreshold;
        TINYTALK_WATERMARK = lib.boolToString cfg.watermark;
        TINYTALK_PYTHON_TARGET = pythonTarget;
        TINYTALK_PIP_INDEX_URL = cfg.runtimeIndexUrl;
        TINYTALK_PIP_EXTRA_INDEX_URLS = lib.concatStringsSep " " cfg.runtimeExtraIndexUrls;
        TINYTALK_RUNTIME_PACKAGES = lib.concatStringsSep " " cfg.runtimePackages;
        PYTHONPATH = "${self.outPath}:${pythonTarget}";
        LD_LIBRARY_PATH = lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib ];
        HOME = "/var/lib/tinytalk";
        UV_CACHE_DIR = "/var/lib/tinytalk/.cache/uv";
      };

      path = [ pkgs.coreutils pkgs.uv python pkgs.ffmpeg-headless ];

      serviceConfig = {
        ExecStartPre = prestart;
        ExecStart = "${python}/bin/python -m uvicorn tinytalk.server:app --host ${cfg.host} --port ${toString cfg.port}";
        User = "tinytalk";
        Group = "tinytalk";
        PrivateTmp = true;
        Restart = "on-failure";
        RestartSec = 3;
        TimeoutStartSec = "15min";
        MemoryHigh = cfg.memoryHigh;
        MemoryMax = cfg.memoryMax;
        StateDirectory = "tinytalk";
        StateDirectoryMode = "0750";
      };
    };
  };
}
