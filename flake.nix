{
  description = "tinytalk - OpenAI-compatible TTS server";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in
    {
      nixosModules.default = import ./nix/module.nix { inherit self; };
      devShells.${system} = import ./nix/dev-shell.nix { inherit pkgs; };
    };
}
