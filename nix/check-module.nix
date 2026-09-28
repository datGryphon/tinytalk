{ self, nixpkgs, pkgs, system }:

let
  lib = pkgs.lib;
  profile = ../examples/tinytauk-cpu.toml;
  appDirectory = "/var/lib/tinytalk/app";

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
    }).config.systemd.services;

  neutts = evaluate "neutts" { };
  omnivoice = evaluate "omnivoice" { };
  tinytauk = evaluate "tinytauk" { tinytaukProfile = profile; };

  usesPreparedApp = services:
    let
      setup = services."tinytalk-setup";
      server = services.tinytalk;
    in
    setup.environment.TINYTALK_SOURCE == "${self.outPath}"
    && setup.environment.TINYTALK_APP_DIR == appDirectory
    && setup.environment.TINYTALK_BACKEND == server.environment.TINYTALK_BACKEND
    && setup.serviceConfig.Type == "oneshot"
    && setup.serviceConfig.RemainAfterExit
    && lib.elem "tinytalk-setup.service" server.requires
    && server.serviceConfig.WorkingDirectory == appDirectory
    && lib.hasPrefix "${appDirectory}/.venv/bin/python " server.serviceConfig.ExecStart
    && !(server.serviceConfig ? ExecStartPre);
in
assert lib.assertMsg (
  usesPreparedApp neutts
  && usesPreparedApp omnivoice
  && usesPreparedApp tinytauk
) "TinyTalk services must prepare a writable app environment before serving";
assert lib.assertMsg (
  neutts.tinytalk.environment.TINYTALK_BACKEND == "neutts"
  && omnivoice.tinytalk.environment.TINYTALK_BACKEND == "omnivoice"
  && tinytauk.tinytalk.environment.TINYTALK_BACKEND == "tinytauk"
) "Each service must forward its selected backend";
assert lib.assertMsg (
  tinytauk.tinytalk.environment.TINYTALK_TINYTAUK_PROFILE == "${profile}"
  && lib.hasInfix "ffmpeg" tinytauk.tinytalk.environment.LD_LIBRARY_PATH
) "TinyTAuK service must forward its profile and provide FFmpeg libraries";

pkgs.runCommand "tinytalk-nixos-module-check" { } "touch $out"
