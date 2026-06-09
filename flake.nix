{
  description = "tinytalk - OpenAI-compatible NeuTTS server";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      runtimeLibs = [
        pkgs.stdenv.cc.cc.lib
      ];
    in
    {
      nixosModules.default = import ./nix/module.nix { inherit self; };

      # Required for local validation with PyPI NumPy wheels on NixOS.
      devShells.${system}.default = pkgs.mkShell {
        shellHook = ''
          export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath runtimeLibs}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        '';
      };
    };
}
