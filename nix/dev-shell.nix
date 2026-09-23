{ pkgs }:

let
  python = pkgs.python313;
  libPath = pkgs.lib.makeLibraryPath [
    pkgs.ffmpeg_8.lib
    pkgs.libsndfile
    pkgs.stdenv.cc.cc.lib
    pkgs.zlib
  ];

  mkBackendShell = { backend, venv, extraPackages ? [ ], override ? null }:
    pkgs.mkShell {
      packages = [ python pkgs.uv pkgs.ffmpeg-headless ] ++ extraPackages;

      shellHook = ''
        export LD_LIBRARY_PATH="${libPath}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        export TINYTALK_BACKEND="${backend}"
        export TINYTALK_PYTHON="${python}/bin/python"
        export TINYTALK_VENV="${venv}"
        export TINYTALK_OVERRIDE_FILE="${if override == null then "" else toString override}"
        source ${./dev-shell-setup.sh}
      '';
    };
in
{
  default = mkBackendShell {
    backend = "neutts";
    venv = ".venv";
    override = ./overrides/neutts.txt;
  };

  omnivoice = mkBackendShell {
    backend = "omnivoice";
    venv = ".venv-omnivoice";
    override = ./overrides/omnivoice.txt;
  };

  tinytauk = mkBackendShell {
    backend = "tinytauk";
    venv = ".venv-tinytauk";
    # TorchInductor may compile native kernels on first VAE decode.
    extraPackages = [ pkgs.gcc pkgs.pkg-config ];
  };
}
