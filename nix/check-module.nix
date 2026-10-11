{ self, nixpkgs, pkgs, system }:

let
  lib = pkgs.lib;
  profile = ../examples/tinytauk-cpu.toml;

  evaluate = backend: options:
    (nixpkgs.lib.nixosSystem {
      inherit system;
      modules = [
        self.nixosModules.default
        {
          system.stateVersion = "25.11";
          # CUDA toolchain packages are unfree; permit evaluation in this test.
          nixpkgs.config.allowUnfree = true;
          services.tinytalk = { enable = true; inherit backend; } // options;
        }
      ];
    }).config.systemd.services.tinytalk;

  neutts = evaluate "neutts" { };
  neuttsGpu = evaluate "neutts" { backboneDevice = "gpu"; llamaCppBackend = "vulkan"; };
  neuttsCuda = evaluate "neutts" {
    backboneDevice = "gpu";
    llamaCppBackend = "cuda";
    llamaCppCudaArchitectures = "61";
  };
  omnivoice = evaluate "omnivoice" { };
  tinytauk = evaluate "tinytauk" { tinytaukProfile = profile; };

  usesAppDirectory = service:
    service.serviceConfig.WorkingDirectory == "/var/lib/tinytalk/app"
    && service.serviceConfig.Type == "exec"
    && !(service.serviceConfig ? ExecStartPre)
    && lib.hasSuffix "-tinytalk-start" service.serviceConfig.ExecStart
  && service.unitConfig.StartLimitBurst == 3;
in
assert lib.assertMsg (
  usesAppDirectory neutts
  && usesAppDirectory omnivoice
  && usesAppDirectory tinytauk
) "TinyTalk services must sync and execute the writable app directory";
assert lib.assertMsg (
  neutts.environment.TINYTALK_BACKEND == "neutts"
  && omnivoice.environment.TINYTALK_BACKEND == "omnivoice"
  && tinytauk.environment.TINYTALK_BACKEND == "tinytauk"
) "Each service must forward its selected backend";
assert lib.assertMsg (
  tinytauk.environment.TINYTALK_TINYTAUK_PROFILE == "${profile}"
  && lib.hasInfix "ffmpeg" tinytauk.environment.LD_LIBRARY_PATH
) "TinyTAuK service must forward its profile and provide FFmpeg libraries";

assert lib.assertMsg (
  neutts.environment.TINYTALK_LLAMA_CPP_BACKEND == "cpu"
  && neuttsGpu.environment.TINYTALK_LLAMA_CPP_BACKEND == "vulkan"
  && neuttsCuda.environment.TINYTALK_LLAMA_CPP_BACKEND == "cuda"
  && lib.hasInfix "cuda-merged" neuttsCuda.environment.CUDA_HOME
  && lib.elem pkgs.cudaPackages_12.cudatoolkit neuttsCuda.path
  && neutts.environment.UV_CACHE_DIR != neuttsGpu.environment.UV_CACHE_DIR
  && lib.elem pkgs.glslang neuttsGpu.path
) "TinyTalk must isolate native wheel caches by NeuTTS acceleration backend";

pkgs.runCommand "tinytalk-nixos-module-check" { } "touch $out"
