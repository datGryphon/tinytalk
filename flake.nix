{
  description = "tinytalk - OpenAI-compatible TTS server";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      libPath = pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib ];
      mkDevShell = { python, venv, extras, backend, extraPackages ? [ ] }:
        pkgs.mkShell {
          packages = [ python pkgs.uv pkgs.ffmpeg-headless ] ++ extraPackages;
          shellHook = ''
            export LD_LIBRARY_PATH="${libPath}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            export TINYTALK_BACKEND="${backend}"
            expected_python="$(${python}/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
            current_python=""
            if [ -x "${venv}/bin/python" ]; then
              current_python="$("${venv}/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
            fi
            if [ "$current_python" != "$expected_python" ]; then
              rm -rf "${venv}"
              uv venv "${venv}" --python ${python}/bin/python --python-preference only-system
            fi
            uv pip install \
              --python "${venv}/bin/python" \
              --index-url https://pypi.org/simple \
              --extra-index-url https://download.pytorch.org/whl/cpu \
              --index-strategy unsafe-best-match \
              -e '.[${extras}]'
            source "${venv}/bin/activate"
          '';
        };
    in
    {
      nixosModules.default = import ./nix/module.nix { inherit self; };

      devShells.${system} = {
        default = mkDevShell {
          python = pkgs.python313;
          venv = ".venv";
          extras = "neutts,test";
          backend = "neutts";
        };

        tinytauk = mkDevShell {
          python = pkgs.python313;
          venv = ".venv-tinytauk";
          extras = "tinytauk,test";
          backend = "tinytauk";
          # TinyTAuK's released CPU profile compiles the VAE decoder lazily
          # with TorchInductor, which invokes a native compiler on first use.
          extraPackages = [ pkgs.gcc pkgs.pkg-config ];
        };

        omnivoice = mkDevShell {
          python = pkgs.python313;
          venv = ".venv-omnivoice";
          extras = "omnivoice,test";
          backend = "omnivoice";
        };
      };
    };
}
