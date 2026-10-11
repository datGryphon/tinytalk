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
  && neuttsCuda.environment.CUDA_HOME == "${pkgs.cudaPackages_12.cudatoolkit}"
  && neuttsCuda.environment.CUDAToolkit_ROOT == "${pkgs.cudaPackages_12.cudatoolkit}"
  && lib.hasInfix "-I${lib.getDev pkgs.cudaPackages_12.cuda_cudart}/include" neuttsCuda.environment.NVCC_PREPEND_FLAGS
  && lib.hasInfix "-I${lib.getDev pkgs.cudaPackages_12.cccl}/include" neuttsCuda.environment.NVCC_PREPEND_FLAGS
  && !(neuttsGpu.environment ? NVCC_PREPEND_FLAGS)
  && lib.hasInfix "GGML_CUDA=on" neuttsCuda.environment.CMAKE_ARGS
  && lib.hasInfix "CMAKE_CUDA_ARCHITECTURES=61" neuttsCuda.environment.CMAKE_ARGS
  && !(neuttsCuda.environment ? CMAKE_PREFIX_PATH)
  && lib.elem pkgs.cudaPackages_12.cuda_nvcc neuttsCuda.path
  && lib.hasInfix "/run/opengl-driver/lib" neuttsCuda.environment.LD_LIBRARY_PATH
  && neutts.environment.TINYTALK_LLAMA_CPP_BACKEND == "cpu"
  && neuttsGpu.environment.TINYTALK_LLAMA_CPP_BACKEND == "vulkan"
  && neuttsCuda.environment.TINYTALK_LLAMA_CPP_BACKEND == "cuda"
  && neutts.environment.UV_CACHE_DIR != neuttsGpu.environment.UV_CACHE_DIR
  && neuttsGpu.environment.UV_CACHE_DIR != neuttsCuda.environment.UV_CACHE_DIR
  && !(omnivoice.environment ? CMAKE_EXECUTABLE)
  && lib.elem pkgs.cmake neutts.path
  && lib.elem pkgs.gnumake neutts.path
  && lib.elem pkgs.shaderc neuttsGpu.path
  && lib.hasInfix "GGML_VULKAN=on" neuttsGpu.environment.CMAKE_ARGS
  && lib.hasInfix "${pkgs.spirv-headers}/include" neuttsGpu.environment.CMAKE_ARGS
  && lib.hasInfix "${pkgs.vulkan-loader}/lib/libvulkan.so" neuttsGpu.environment.CMAKE_ARGS
  && lib.hasInfix "${pkgs.vulkan-loader}/lib" neuttsGpu.environment.LD_LIBRARY_PATH
) "TinyTalk must provide CMake/Make and NeuTTS GPU Vulkan build inputs";

pkgs.runCommand "tinytalk-nixos-module-check" { } "touch $out"
