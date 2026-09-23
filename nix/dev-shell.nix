{ pkgs }:

let
  python = pkgs.python313;
  libPath = pkgs.lib.makeLibraryPath [
    pkgs.ffmpeg_8.lib
    pkgs.libsndfile
    pkgs.stdenv.cc.cc.lib
    pkgs.zlib
  ];

  mkBackendShell = { backend, venv, extraPackages ? [ ] }:
    pkgs.mkShell {
      packages = [ python pkgs.uv pkgs.ffmpeg-headless ] ++ extraPackages;

      shellHook = ''
        export LD_LIBRARY_PATH="${libPath}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        export TINYTALK_BACKEND="${backend}"
        export UV_PROJECT_ENVIRONMENT="$PWD/${venv}"

        uv sync \
          --frozen \
          --python ${python}/bin/python \
          --extra "${backend}" \
          --extra test || return 1

        source "$UV_PROJECT_ENVIRONMENT/bin/activate"
      '';
    };

  neutts = mkBackendShell {
    backend = "neutts";
    venv = ".venv-neutts";
  };
in
{
  default = neutts;
  inherit neutts;

  omnivoice = mkBackendShell {
    backend = "omnivoice";
    venv = ".venv-omnivoice";
  };

  tinytauk = mkBackendShell {
    backend = "tinytauk";
    venv = ".venv-tinytauk";
    # TorchInductor may compile native kernels on first VAE decode.
    extraPackages = [ pkgs.gcc pkgs.pkg-config ];
  };
}
