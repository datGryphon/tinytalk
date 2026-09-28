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
          services.tinytalk = { enable = true; inherit backend; } // options;
        }
      ];
    }).config.systemd.services.tinytalk;

  neutts = evaluate "neutts" { };
  omnivoice = evaluate "omnivoice" { };
  tinytauk = evaluate "tinytauk" { tinytaukProfile = profile; };

  usesAppDirectory = service:
    service.serviceConfig.WorkingDirectory == "/var/lib/tinytalk/app"
    && lib.hasPrefix "/var/lib/tinytalk/app/.venv/bin/python " service.serviceConfig.ExecStart;
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

pkgs.runCommand "tinytalk-nixos-module-check" { } "touch $out"
