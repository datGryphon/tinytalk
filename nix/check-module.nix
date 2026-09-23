{ self, nixpkgs, pkgs, system }:

let
  lib = pkgs.lib;
  profile = ../examples/tinytauk-cpu.toml;

  service = backend: options:
    (nixpkgs.lib.nixosSystem {
      inherit system;
      modules = [
        self.nixosModules.default
        {
          system.stateVersion = "25.11";
          services.tinytalk = { enable = true; inherit backend; } // options;
        }
      ];
    }).config.systemd.services.tinytalk.environment;

  packages = backend: options:
    (nixpkgs.lib.nixosSystem {
      inherit system;
      modules = [
        self.nixosModules.default
        {
          system.stateVersion = "25.11";
          services.tinytalk = { enable = true; inherit backend; } // options;
        }
      ];
    }).config.services.tinytalk.runtimePackages;

  neutts = service "neutts" { };
  omnivoice = service "omnivoice" { };
  tinytauk = service "tinytauk" { tinytaukProfile = profile; };

  neuPackages = packages "neutts" { };
  omniPackages = packages "omnivoice" { };
  aukPackages = packages "tinytauk" { tinytaukProfile = profile; };

  qualified = specs:
    lib.elem "torch==2.11.0+cpu" specs
    && lib.elem "torchaudio==2.11.0+cpu" specs
    && lib.elem "transformers==5.17.0" specs;
in
assert lib.assertMsg (qualified neuPackages && qualified omniPackages && qualified aukPackages)
  "TinyTalk NixOS module must deploy the qualified shared Torch/Transformers runtime";
assert lib.assertMsg (
  lib.any (lib.hasInfix "6954b1f877963e19177b43be1ef56b1818990d31") neuPackages
  && !lib.any (lib.hasInfix "torchtune") neuPackages
  && !lib.any (lib.hasInfix "torchao") neuPackages
  && neuPackages != [ ]
  && neutts.TINYTALK_RUNTIME_OVERRIDE != ""
) "NeuTTS service must use the qualified NeuCodec fork and metadata override";
assert lib.assertMsg (
  lib.elem "omnivoice==0.2.1" omniPackages
  && omnivoice.TINYTALK_RUNTIME_OVERRIDE != ""
) "OmniVoice service must use the qualified metadata override";
assert lib.assertMsg (
  lib.any (lib.hasInfix "/v0.2.0.tar.gz") aukPackages
  && tinytauk.TINYTALK_TINYTAUK_PROFILE == toString profile
  && tinytauk.TINYTALK_RUNTIME_OVERRIDE == ""
  && lib.hasInfix "ffmpeg" tinytauk.LD_LIBRARY_PATH
) "TinyTAuK service must use v0.2.0, forward the TOML profile, and provide FFmpeg libraries";

pkgs.runCommand "tinytalk-nixos-module-check" { } "touch $out"
