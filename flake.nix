{
  description = "tinytalk - OpenAI-compatible NeuTTS server";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      libPath = pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib ];
    in
    {
      nixosModules.default = import ./nix/module.nix { inherit self; };

      devShells.${system}.default = pkgs.mkShell {
        packages = [ pkgs.python312 pkgs.uv pkgs.ffmpeg-headless ];
        shellHook = ''
          export LD_LIBRARY_PATH="${libPath}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
          if [ ! -d .venv ]; then
            uv venv --python python3 --python-preference only-system
          fi
          # Both indexes are trusted upstreams. Resolve across them so packages
          # such as torchtune can come from PyPI while the higher-priority
          # PyTorch index supplies the +cpu torch/torchaudio wheels.
          uv pip install \
            --python .venv/bin/python \
            --index-url https://pypi.org/simple \
            --extra-index-url https://download.pytorch.org/whl/cpu \
            --index-strategy unsafe-best-match \
            -e '.[test]'
          source .venv/bin/activate
        '';
      };
    };
}
