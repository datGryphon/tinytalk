{
  description = "tinytalk - OpenAI-compatible TTS server";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      libPath = pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib ];
      mkDevShell = { python, venv, extras, backend }:
        pkgs.mkShell {
          packages = [ python pkgs.uv pkgs.ffmpeg-headless ];
          shellHook = ''
            export LD_LIBRARY_PATH="${libPath}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            export TINYTALK_BACKEND="${backend}"
            if [ ! -d "${venv}" ]; then
              uv venv "${venv}" --python ${python}/bin/python --python-preference only-system
              uv pip install \
                --python "${venv}/bin/python" \
                --index-url https://download.pytorch.org/whl/cpu \
                --extra-index-url https://pypi.org/simple \
                -e '.[${extras}]'
            fi
            source "${venv}/bin/activate"
          '';
        };
    in
    {
      nixosModules.default = import ./nix/module.nix { inherit self; };

      devShells.${system}.default = mkDevShell {
        python = pkgs.python312;
        venv = ".venv";
        extras = "neutts,test";
        backend = "neutts";
      };

      devShells.${system}.tinytauk = mkDevShell {
        python = pkgs.python313;
        venv = ".venv-tinytauk";
        extras = "tinytauk,test";
        backend = "tinytauk";
      };
    };
}
