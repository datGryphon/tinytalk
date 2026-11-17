{ pkgs }:

let
  python = pkgs.python313;
  libPath = pkgs.lib.makeLibraryPath [
    pkgs.ffmpeg_8.lib
    pkgs.libsndfile
    pkgs.stdenv.cc.cc.lib
    pkgs.zlib
  ];

in
{
  default = pkgs.mkShell {
    packages = [
      python pkgs.uv pkgs.ffmpeg-headless
      # llama.cpp builds and TinyTAuK's TorchInductor compilation.
      pkgs.gcc pkgs.cmake pkgs.gnumake pkgs.pkg-config
    ];

    shellHook = ''
      export LD_LIBRARY_PATH="${libPath}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
      export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
      uv sync --frozen --all-extras --python ${python}/bin/python || return 1
      source "$UV_PROJECT_ENVIRONMENT/bin/activate"
    '';
  };
}
