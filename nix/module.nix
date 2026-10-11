{ self }:
{ config, lib, pkgs, ... }:

let
  cfg = config.services.tinytalk;
  python = pkgs.python313;
  appDirectory = "/var/lib/tinytalk/app";

  commonOptions = {
    enable = lib.mkEnableOption "OpenAI-compatible tinytalk TTS server";

    backend = lib.mkOption {
      type = lib.types.enum [ "neutts" "tinytauk" "omnivoice" ];
      default = "neutts";
      description = "Synthesis backend for this service instance.";
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
  };

  qualityOptions = {
    maxRetries = lib.mkOption {
      type = lib.types.ints.between 0 100;
      default = 2;
      description = "Maximum number of quality-gated retry attempts per chunk.";
    };

    werEndpoint = lib.mkOption {
      type = lib.types.str;
      default = "";
      description = "Base URL for transcription-based WER/CER evaluation. Empty uses the local confidence fallback.";
    };

    werThreshold = lib.mkOption {
      type = lib.types.float;
      default = 0.25;
      description = "Maximum WER/CER accepted by the shared speech-quality gate.";
    };
  };

  neuttsOptions = {
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

    llamaCppBackend = lib.mkOption {
      type = lib.types.enum [ "cpu" "vulkan" "cuda" ];
      default = "cpu";
      description = "llama-cpp-python acceleration backend for NeuTTS; select independently from the runtime device.";
    };

    llamaCppCudaArchitectures = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      description = "Optional CMake CUDA architectures (e.g. 61 for NVIDIA Pascal).";
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

    watermark = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Enable NeuTTS perth audio watermarking.";
    };
  };

  tinytaukOptions = {
    tinytaukProfile = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      description = "Optional TinyTAuK v0.2 TOML runtime profile. When set, model IDs, devices, dtypes, seed and thread count come from the profile.";
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
      description = "Text-duration estimate used by TinyTAuK before applying request speed.";
    };
  };

  omnivoiceOptions = {
    omnivoiceModel = lib.mkOption {
      type = lib.types.str;
      default = "k2-fsa/OmniVoice";
      description = "OmniVoice model repository or local checkpoint path.";
    };

    omnivoiceDevice = lib.mkOption {
      type = lib.types.str;
      default = "cpu";
      description = "Device map passed to OmniVoice.from_pretrained.";
    };

    omnivoiceLanguage = lib.mkOption {
      type = lib.types.str;
      default = "";
      description = "Optional OmniVoice language name or code. Empty enables automatic language handling.";
    };

    omnivoiceRefAudio = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      description = "Optional reference WAV for OmniVoice voice cloning.";
    };

    omnivoiceRefText = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      description = "Transcript file paired with omnivoiceRefAudio.";
    };
  };

  runtimeOptions = {
    memoryHigh = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      description = "systemd MemoryHigh. Null selects the backend default.";
    };

    memoryMax = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      description = "systemd MemoryMax. Null selects the backend default.";
    };
  };

  commonEnvironment = {
    TINYTALK_BACKEND = cfg.backend;
    TINYTALK_HOST = cfg.host;
    TINYTALK_PORT = toString cfg.port;
    TINYTALK_MAX_CHARS_PER_CHUNK = toString cfg.maxCharsPerChunk;
    TINYTALK_INTER_CHUNK_SILENCE_MS = toString cfg.interChunkSilenceMs;
  };

  qualityEnvironment = {
    TINYTALK_MAX_RETRIES = toString cfg.maxRetries;
    TINYTALK_WER_ENDPOINT = cfg.werEndpoint;
    TINYTALK_WER_THRESHOLD = toString cfg.werThreshold;
  };

  neuttsEnvironment = {
    TINYTALK_MODEL = cfg.model;
    TINYTALK_CODEC = cfg.codec;
    TINYTALK_BACKBONE_DEVICE = cfg.backboneDevice;
    TINYTALK_REF_CODES = toString cfg.refCodes;
    TINYTALK_REF_TEXT = toString cfg.refText;
    TINYTALK_TEMPERATURE = toString cfg.temperature;
    TINYTALK_REPEAT_PENALTY = toString cfg.repeatPenalty;
    TINYTALK_REPEAT_PENALTY_REROLL_STEP = toString cfg.repeatPenaltyRerollStep;
    TINYTALK_WATERMARK = lib.boolToString cfg.watermark;
  };

  tinytaukEnvironment = {
    TINYTALK_TINYTAUK_MODEL = cfg.tinytaukModel;
    TINYTALK_TINYTAUK_QWEN_MODEL = cfg.tinytaukQwenModel;
    TINYTALK_TINYTAUK_CHARS_PER_SECOND = toString cfg.tinytaukCharsPerSecond;
  }
  // lib.optionalAttrs (cfg.tinytaukProfile != null) {
    TINYTALK_TINYTAUK_PROFILE = "${cfg.tinytaukProfile}";
  };

  omnivoiceEnvironment = {
    TINYTALK_OMNIVOICE_MODEL = cfg.omnivoiceModel;
    TINYTALK_OMNIVOICE_DEVICE = cfg.omnivoiceDevice;
    TINYTALK_OMNIVOICE_LANGUAGE = cfg.omnivoiceLanguage;
  }
  // lib.optionalAttrs (cfg.omnivoiceRefAudio != null) {
    TINYTALK_OMNIVOICE_REF_AUDIO = toString cfg.omnivoiceRefAudio;
  }
  // lib.optionalAttrs (cfg.omnivoiceRefText != null) {
    TINYTALK_OMNIVOICE_REF_TEXT = toString cfg.omnivoiceRefText;
  };

  backendEnvironment =
    if cfg.backend == "tinytauk" then tinytaukEnvironment
    else if cfg.backend == "omnivoice" then omnivoiceEnvironment
    else neuttsEnvironment;

  defaultMemoryHigh =
    if cfg.backend == "tinytauk" then "17G"
    else if cfg.backend == "omnivoice" then "12G"
    else "5000M";
  defaultMemoryMax =
    if cfg.backend == "tinytauk" then "20G"
    else if cfg.backend == "omnivoice" then "16G"
    else "6000M";

  prestart = pkgs.writeShellScript "tinytalk-prestart.sh"
    (builtins.readFile ./tinytalk-prestart.sh);

  # Keep dependency installation out of ExecStartPre: compilation must not
  # block NixOS activation and deploy-rs confirmation.
  start = pkgs.writeShellScript "tinytalk-start" ''
    set -euo pipefail
    ${prestart} ${self.outPath} ${appDirectory}
    cd ${appDirectory}
    exec .venv/bin/python -m uvicorn tinytalk.server:app \
      --host "$TINYTALK_HOST" --port "$TINYTALK_PORT"
  '';

  neutts = cfg.backend == "neutts";
  vulkan = neutts && cfg.llamaCppBackend == "vulkan";
  cuda = neutts && cfg.llamaCppBackend == "cuda";
  # Use a CUDA 12 toolchain for the optional NVIDIA source build.
  cudaPackages = pkgs.cudaPackages_12;

  buildEnvironment = lib.optionalAttrs neutts ({
    CMAKE_EXECUTABLE = "${pkgs.cmake}/bin/cmake";
    CMAKE_GENERATOR = "Unix Makefiles";
  } // lib.optionalAttrs vulkan {
    # ggml-vulkan uses find_package(SPIRV-Headers CONFIG REQUIRED).
    CMAKE_PREFIX_PATH = "${pkgs.spirv-headers}";
    CMAKE_ARGS = lib.concatStringsSep " " [
      "-DGGML_VULKAN=on"
      "-DVulkan_INCLUDE_DIR=${pkgs.vulkan-headers}/include"
      "-DVulkan_LIBRARY=${pkgs.vulkan-loader}/lib/libvulkan.so"
      "-DCMAKE_CXX_FLAGS=-I${pkgs.spirv-headers}/include"
    ];
  } // lib.optionalAttrs cuda {
    CUDA_HOME = "${cudaPackages.cudatoolkit}";
    CUDAToolkit_ROOT = "${cudaPackages.cudatoolkit}";
    # Nixpkgs ships nvcc and CUDA runtime headers in separate outputs.
    # Include both during CMake's initial CUDA compiler detection.
    NVCC_PREPEND_FLAGS = lib.concatStringsSep " " [
      "-I${lib.getDev cudaPackages.cuda_cudart}/include"
      "-I${lib.getDev cudaPackages.cccl}/include"
      # nvcc implicitly links libcudadevrt.a and libcudart_static.a.
      "-L${lib.getOutput "static" cudaPackages.cuda_cudart}/lib"
    ];
    CMAKE_ARGS = lib.concatStringsSep " " (
      [ "-DGGML_CUDA=on" ]
      ++ lib.optionals (cfg.llamaCppCudaArchitectures != null) [
        "-DCMAKE_CUDA_ARCHITECTURES=${cfg.llamaCppCudaArchitectures}"
      ]
    );
  });

  runtimeEnvironment = {
    LD_LIBRARY_PATH = lib.concatStringsSep ":" (
      lib.optionals cuda [ "/run/opengl-driver/lib" ]
      ++ [
        (lib.makeLibraryPath (
          [
            pkgs.ffmpeg_8.lib
            pkgs.libsndfile
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
          ] ++ lib.optionals vulkan [ pkgs.vulkan-loader ]
            ++ lib.optionals cuda [ cudaPackages.cuda_cudart cudaPackages.libcublas ]
        ))
      ]
    );
    HOME = "/var/lib/tinytalk";
    # uv caches built native wheels: do not share Vulkan/CPU/CUDA variants.
    UV_CACHE_DIR = "/var/lib/tinytalk/.cache/uv/${if neutts then "neutts-${cfg.llamaCppBackend}" else cfg.backend}";
    TINYTALK_LLAMA_CPP_BACKEND = if neutts then cfg.llamaCppBackend else "none";
  };
in
{
  options.services.tinytalk =
    commonOptions
    // qualityOptions
    // neuttsOptions
    // tinytaukOptions
    // omnivoiceOptions
    // runtimeOptions;

  config = lib.mkIf cfg.enable {
    assertions = [
      {
        assertion = cfg.backend != "neutts"
          || ((cfg.backboneDevice == "gpu") == (cfg.llamaCppBackend != "cpu"));
        message = "services.tinytalk: NeuTTS backboneDevice=gpu requires llamaCppBackend=vulkan or cuda; cpu requires llamaCppBackend=cpu";
      }
      {
        assertion = cfg.backend == "neutts"
          || (cfg.llamaCppBackend == "cpu" && cfg.llamaCppCudaArchitectures == null);
        message = "services.tinytalk: llamaCpp options are only applicable to the neutts backend";
      }
      {
        assertion = cfg.llamaCppBackend == "cuda" || cfg.llamaCppCudaArchitectures == null;
        message = "services.tinytalk.llamaCppCudaArchitectures requires llamaCppBackend=cuda";
      }
      {
        assertion = cfg.backend != "tinytauk" || cfg.tinytaukCharsPerSecond > 0.0;
        message = "services.tinytalk.tinytaukCharsPerSecond must be positive";
      }
      {
        assertion =
          cfg.backend != "omnivoice"
          || ((cfg.omnivoiceRefAudio == null) == (cfg.omnivoiceRefText == null));
        message = "services.tinytalk.omnivoiceRefAudio and omnivoiceRefText must be configured together";
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

      environment =
        commonEnvironment
        // qualityEnvironment
        // backendEnvironment
        // runtimeEnvironment
        // buildEnvironment;

      path =
        [ pkgs.coreutils pkgs.uv python pkgs.ffmpeg-headless ]
        ++ lib.optionals neutts [ pkgs.gcc pkgs.cmake pkgs.gnumake pkgs.pkg-config ]
        ++ lib.optionals (cfg.backend == "tinytauk") [ pkgs.gcc pkgs.pkg-config ]
        ++ lib.optionals vulkan [ pkgs.shaderc ]
        ++ lib.optionals cuda [
          cudaPackages.cuda_nvcc
          cudaPackages.cuda_cudart
          cudaPackages.libcublas
          cudaPackages.cccl
        ];

      # A failed build should not retry indefinitely.
      unitConfig = {
        StartLimitIntervalSec = "10min";
        StartLimitBurst = 3;
      };

      serviceConfig = {
        Type = "exec";
        WorkingDirectory = "/var/lib/tinytalk";
        ExecStart = "${start}";
        User = "tinytalk";
        Group = "tinytalk";
        PrivateTmp = true;
        Restart = "on-failure";
        RestartSec = 3;
        TimeoutStartSec =
          if cfg.backend == "tinytauk" then "30min"
          else if cfg.backend == "omnivoice" then "20min"
          else "15min";
        MemoryHigh = if cfg.memoryHigh != null then cfg.memoryHigh else defaultMemoryHigh;
        MemoryMax = if cfg.memoryMax != null then cfg.memoryMax else defaultMemoryMax;
        StateDirectory = "tinytalk";
        StateDirectoryMode = "0750";
      };
    };
  };
}
