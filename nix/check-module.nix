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
          services.tinytalk = { enable = true; inherit backend; } // options;
        }
      ];
    }).config.systemd.services.tinytalk;

  neutts = evaluate "neutts" { };
  neuttsGpu = evaluate "neutts" { backboneDevice = "gpu"; };
  omnivoice = evaluate "omnivoice" { };
  tinytauk = evaluate "tinytauk" { tinytaukProfile = profile; };

  usesAppDirectory = service:
    service.serviceConfig.WorkingDirectory == "/var/lib/tinytalk"
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
  neutts.environment.CMAKE_EXECUTABLE == "${pkgs.cmake}/bin/cmake"
  && neutts.environment.CMAKE_GENERATOR == "Unix Makefiles"
  && !(neutts.environment ? CMAKE_ARGS)
  && !(neutts.environment ? CMAKE_PREFIX_PATH)
  && neuttsGpu.environment.CMAKE_PREFIX_PATH == "${pkgs.spirv-headers}"
  && lib.elem pkgs.cmake neutts.path
  && lib.elem pkgs.gnumake neutts.path
  && lib.elem pkgs.shaderc neuttsGpu.path
  && lib.hasInfix "GGML_VULKAN=on" neuttsGpu.environment.CMAKE_ARGS
  && lib.hasInfix "${pkgs.spirv-headers}/include" neuttsGpu.environment.CMAKE_ARGS
  && lib.hasInfix "${pkgs.vulkan-loader}/lib/libvulkan.so" neuttsGpu.environment.CMAKE_ARGS
  && lib.hasInfix "${pkgs.vulkan-loader}/lib" neuttsGpu.environment.LD_LIBRARY_PATH
) "TinyTalk must provide CMake/Make and NeuTTS GPU Vulkan build inputs";

pkgs.runCommand "tinytalk-nixos-module-check" { } "touch $out"
