# Named voices — design and verification criteria

## Ownership

A TinyTalk service owns exactly one resident backend/model and retains its existing concrete `SpeechEngine` and `SpeechBackend` strategy lifecycle. Named voices are static presets for that backend, not model or backend selectors. Preserve the existing inference lock, request finalization, and single venv.

## Configuration

Introduce an optional static JSON voice configuration file, validated with typed Pydantic models at startup, and NixOS `services.tinytalk.voices` plus `defaultVoice` options which emit that file declaratively. With no voice configuration, preserve current behavior. With configured voices, require an explicit default, reject unknown requested names, and reject backend-incompatible fields. Do not accept runtime enrollment or arbitrary overrides.

Voice presets should expose native controls only: NeuTTS reference codes plus transcript, OmniVoice reference audio plus transcript, Silencio-2 native ID or reference file, Chatterbox Nano reference audio, and TinyTAuK instruction description. Optional per-voice chunk length and inter-chunk silence are request-scoped; never mutate global settings.

## Lifecycle

Resolve the existing speech API `voice` field, pass a typed preset into `SpeechEngine.synthesize`, and preserve backend `prepare_request`/`synthesize_chunk`/`finalize_request`. Load the model once. Prepare configured reference conditioning at startup where upstream requires it. Keep NeuTTS F0 state per request. For mutable upstream conditioning (particularly Chatterbox Nano), confirm the pinned API and select within the existing inference lock rather than creating multiple models. Keep TinyTAuK text-dependent conditioning at request time.

## Verification criteria

Validate default and unknown voice behavior, per-backend reference requirements, reference preparation, per-request chunk/cadence isolation, API response behavior, and Nix-generated config. Run focused tests and full shared tests; do not equate mocked tests with actual model inference. No new service, router, caching framework, benchmarks, or unrelated cleanup.

