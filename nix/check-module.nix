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

  usesLockedProject = service:
    service.environment.TINYTALK_PROJECT_ROOT == "${self.outPath}"
    && service.environment.TINYTALK_PYTHON_ENVIRONMENT == "/var/lib/tinytalk/python"
    && lib.hasPrefix "/var/lib/tinytalk/python/bin/python " service.serviceConfig.ExecStart
    && !(service.environment ? TINYTALK_RUNTIME_REQUIREMENTS)
    && !(service.environment ? TINYTALK_RUNTIME_OVERRIDE)
    && !(service.environment ? TINYTALK_PIP_EXTRA_INDEX_URLS);
in
assert lib.assertMsg (
  usesLockedProject neutts
  && usesLockedProject omnivoice
  && usesLockedProject tinytauk
) "TinyTalk services must sync and execute the locked project environment";
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
