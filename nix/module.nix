{ self }:
{ config, lib, pkgs, ... }:

let
  cfg = config.services.tinytalk;
  python = if cfg.backend == "tinytauk" then pkgs.python313 else pkgs.python312;
  pythonTarget = "/var/lib/tinytalk/python";

  neuttsRuntimePackages = [
    "torch==2.11.0+cpu"
    "torchaudio==2.11.0+cpu"
    "neutts[all]"
    "fastapi"
    "uvicorn[standard]"
    "spacy"
    "librosa"
    "praat-parselmouth"
  ];

  tinytaukRuntimePackages = [
    "tinytauk @ https://github.com/datGryphon/tinytauk/archive/refs/tags/v0.1.0.tar.gz"
    "fastapi"
    "uvicorn[standard]"
    "numpy"
    "spacy"
    "librosa"
    "praat-parselmouth"
  ];

  runtimePackages =
    if cfg.backend == "tinytauk"
    then tinytaukRuntimePackages
    else neuttsRuntimePackages;

  defaultMemoryHigh = if cfg.backend == "tinytauk" then "17G" else "5000M";
  defaultMemoryMax = if cfg.backend == "tinytauk" then "20G" else "6000M";

  prestart = pkgs.writeShellScript "tinytalk-prestart.sh" (
    builtins.replaceStrings
      [ "@python@" "@uv@" ]
      [ "${python}" "${pkgs.uv}" ]
      (builtins.readFile ./tinytalk-prestart.sh)
  );
  requirementsFile = pkgs.writeText "tinytalk-runtime-requirements.txt" (
    lib.concatStringsSep "\n" cfg.runtimePackages + "\n"
  );
in
{
  options.services.tinytalk = {
    enable = lib.mkEnableOption "OpenAI-compatible tinytalk TTS server";

    backend = lib.mkOption {
      type = lib.types.enum [ "neutts" "tinytauk" ];
      default = "neutts";
      description = "Synthesis backend for this service instance.";
    };

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

    tinytaukModel = lib.mkOption {
      type = lib.types.str;
      default = "tencent/AuK-Flash";
      description = "AuK-Flash model repository passed to TinyTAuK.";
    };

    tinytaukQwenModel = lib.mkOption {
      type = lib.types.str;
      default = "Qwen/Qwen2.5-Omni-3B";
      description = "Qwen conditioner repository passed to TinyTAuK.";
    };

    tinytaukCharsPerSecond = lib.mkOption {
      type = lib.types.float;
      default = 14.0;
      description = "Text-duration estimate used by the TinyTAuK backend before applying request speed.";
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
      description = "Maximum text characters sent to one synthesis call.";
    };

    interChunkSilenceMs = lib.mkOption {
      type = lib.types.ints.unsigned;
      default = 60;
      description = "Zero-audio pause inserted between synthesized chunks.";
    };

    temperature = lib.mkOption {
      type = lib.types.float;
      default = 1.0;
      description = "Sampling temperature for the NeuTTS backbone.";
    };

    repeatPenalty = lib.mkOption {
      type = lib.types.float;
      default = 1.0;
      description = "Repeat penalty for the NeuTTS backbone.";
    };

    repeatPenaltyRerollStep = lib.mkOption {
      type = lib.types.float;
      default = 0.10;
      description = "Incremental repeat penalty added per NeuTTS retry attempt.";
    };

    maxRetries = lib.mkOption {
      type = lib.types.ints.between 0 100;
      default = 2;
      description = "Maximum number of NeuTTS retry attempts per chunk.";
    };

    werEndpoint = lib.mkOption {
      type = lib.types.str;
      default = "";
      description = "Base URL for NeuTTS WER transcription evaluation. Empty disables live transcription.";
    };

    werThreshold = lib.mkOption {
      type = lib.types.float;
      default = 0.25;
      description = "NeuTTS WER threshold at or below which a chunk is accepted.";
    };

    watermark = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = ''
        Enable NeuTTS perth audio watermarking. Disabled by default because the
        watermarker retains substantial PyTorch CPU allocator memory per call.
      '';
    };

    memoryHigh = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      description = "systemd MemoryHigh. Null selects the backend default (5 GB NeuTTS, 17 GB TinyTAuK).";
    };

    memoryMax = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      description = "systemd MemoryMax. Null selects the backend default (6 GB NeuTTS, 20 GB TinyTAuK).";
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
    assertions = [
      {
        assertion = cfg.backend != "tinytauk" || cfg.tinytaukCharsPerSecond > 0.0;
        message = "services.tinytalk.tinytaukCharsPerSecond must be positive";
      }
    ];

    users.users.tinytalk = {
      isSystemUser = true;
      group = "tinytalk";
      home = "/var/lib/tinytalk";
    };
    users.groups.tinytalk = { };

    systemd.services.tinytalk = {
      description = "tinytalk OpenAI-compatible TTS server";
      after = [ "network-online.target" ];
      wants = [ "network-online.target" ];
      wantedBy = [ "multi-user.target" ];

      environment = {
        TINYTALK_BACKEND = cfg.backend;
        TINYTALK_MODEL = cfg.model;
        TINYTALK_CODEC = cfg.codec;
        TINYTALK_BACKBONE_DEVICE = cfg.backboneDevice;
        TINYTALK_REF_CODES = toString cfg.refCodes;
        TINYTALK_REF_TEXT = toString cfg.refText;
        TINYTALK_TINYTAUK_MODEL = cfg.tinytaukModel;
        TINYTALK_TINYTAUK_QWEN_MODEL = cfg.tinytaukQwenModel;
        TINYTALK_TINYTAUK_CHARS_PER_SECOND = toString cfg.tinytaukCharsPerSecond;
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
        TINYTALK_RUNTIME_REQUIREMENTS = toString requirementsFile;
        PYTHONPATH = "${self.outPath}:${pythonTarget}";
        LD_LIBRARY_PATH = lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib ];
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
        TimeoutStartSec = if cfg.backend == "tinytauk" then "30min" else "15min";
        MemoryHigh = if cfg.memoryHigh != null then cfg.memoryHigh else defaultMemoryHigh;
        MemoryMax = if cfg.memoryMax != null then cfg.memoryMax else defaultMemoryMax;
        StateDirectory = "tinytalk";
        StateDirectoryMode = "0750";
      };
    };
  };
}
