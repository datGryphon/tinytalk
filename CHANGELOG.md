# CHANGELOG

<!-- version list -->

## v0.2.0 (2026-11-17)

### Bug Fixes

- Omit request text from expressive backend telemetry
  ([#14](https://github.com/datGryphon/tinytalk/pull/14),
  [`a2f4b1f`](https://github.com/datGryphon/tinytalk/commit/a2f4b1f7acd890baa4cc8fd70e68cd49774b7d80))

- Preserve legacy chunk preparation call without overrides
  ([#15](https://github.com/datGryphon/tinytalk/pull/15),
  [`d0ed159`](https://github.com/datGryphon/tinytalk/commit/d0ed1590e4568fea8ce0f826f458ba00fb276041))

- Retain legacy API engine call when no named voices configured
  ([#15](https://github.com/datGryphon/tinytalk/pull/15),
  [`d0ed159`](https://github.com/datGryphon/tinytalk/commit/d0ed1590e4568fea8ce0f826f458ba00fb276041))

- Retain legacy NeuTTS chunk invocation without named preset
  ([#15](https://github.com/datGryphon/tinytalk/pull/15),
  [`d0ed159`](https://github.com/datGryphon/tinytalk/commit/d0ed1590e4568fea8ce0f826f458ba00fb276041))

### Documentation

- Align named voice design status with implementation
  ([#15](https://github.com/datGryphon/tinytalk/pull/15),
  [`d0ed159`](https://github.com/datGryphon/tinytalk/commit/d0ed1590e4568fea8ce0f826f458ba00fb276041))

- Describe named voice API and NixOS setup ([#15](https://github.com/datGryphon/tinytalk/pull/15),
  [`d0ed159`](https://github.com/datGryphon/tinytalk/commit/d0ed1590e4568fea8ce0f826f458ba00fb276041))

- Remove branch-specific implementation status from design document
  ([#15](https://github.com/datGryphon/tinytalk/pull/15),
  [`d0ed159`](https://github.com/datGryphon/tinytalk/commit/d0ed1590e4568fea8ce0f826f458ba00fb276041))

- Scope named voices to a single resident backend
  ([#15](https://github.com/datGryphon/tinytalk/pull/15),
  [`d0ed159`](https://github.com/datGryphon/tinytalk/commit/d0ed1590e4568fea8ce0f826f458ba00fb276041))

### Features

- Add Chatterbox Nano on the shared CPU runtime
  ([#14](https://github.com/datGryphon/tinytalk/pull/14),
  [`a2f4b1f`](https://github.com/datGryphon/tinytalk/commit/a2f4b1f7acd890baa4cc8fd70e68cd49774b7d80))

- Add resident Silencio-2 expressive CPU backend
  ([#14](https://github.com/datGryphon/tinytalk/pull/14),
  [`a2f4b1f`](https://github.com/datGryphon/tinytalk/commit/a2f4b1f7acd890baa4cc8fd70e68cd49774b7d80))

- Add Silencio-2 and Chatterbox Nano with shared synthesis
  ([#14](https://github.com/datGryphon/tinytalk/pull/14),
  [`a2f4b1f`](https://github.com/datGryphon/tinytalk/commit/a2f4b1f7acd890baa4cc8fd70e68cd49774b7d80))

- Emit declarative NixOS named voice configuration
  ([#15](https://github.com/datGryphon/tinytalk/pull/15),
  [`d0ed159`](https://github.com/datGryphon/tinytalk/commit/d0ed1590e4568fea8ce0f826f458ba00fb276041))

- Honor speech API voice selection and reject unknown names
  ([#15](https://github.com/datGryphon/tinytalk/pull/15),
  [`d0ed159`](https://github.com/datGryphon/tinytalk/commit/d0ed1590e4568fea8ce0f826f458ba00fb276041))

- Preload and select OmniVoice clone prompts ([#15](https://github.com/datGryphon/tinytalk/pull/15),
  [`d0ed159`](https://github.com/datGryphon/tinytalk/commit/d0ed1590e4568fea8ce0f826f458ba00fb276041))

- Preload and select Silencio-2 voice IDs ([#15](https://github.com/datGryphon/tinytalk/pull/15),
  [`d0ed159`](https://github.com/datGryphon/tinytalk/commit/d0ed1590e4568fea8ce0f826f458ba00fb276041))

- Preload NeuTTS named reference conditioning
  ([#15](https://github.com/datGryphon/tinytalk/pull/15),
  [`d0ed159`](https://github.com/datGryphon/tinytalk/commit/d0ed1590e4568fea8ce0f826f458ba00fb276041))

- Resolve named voices in existing synthesis lifecycle
  ([#15](https://github.com/datGryphon/tinytalk/pull/15),
  [`d0ed159`](https://github.com/datGryphon/tinytalk/commit/d0ed1590e4568fea8ce0f826f458ba00fb276041))

- Reuse upstream Chatterbox Nano conditioning for named voices
  ([#15](https://github.com/datGryphon/tinytalk/pull/15),
  [`d0ed159`](https://github.com/datGryphon/tinytalk/commit/d0ed1590e4568fea8ce0f826f458ba00fb276041))

- Static named voice presets for shared synthesis engine
  ([#15](https://github.com/datGryphon/tinytalk/pull/15),
  [`d0ed159`](https://github.com/datGryphon/tinytalk/commit/d0ed1590e4568fea8ce0f826f458ba00fb276041))

- Support named TinyTAuK voice descriptions ([#15](https://github.com/datGryphon/tinytalk/pull/15),
  [`d0ed159`](https://github.com/datGryphon/tinytalk/commit/d0ed1590e4568fea8ce0f826f458ba00fb276041))

- Validate static named voice presets with Pydantic
  ([#15](https://github.com/datGryphon/tinytalk/pull/15),
  [`d0ed159`](https://github.com/datGryphon/tinytalk/commit/d0ed1590e4568fea8ce0f826f458ba00fb276041))

### Refactoring

- Compose speech engine with backend strategies
  ([#14](https://github.com/datGryphon/tinytalk/pull/14),
  [`a2f4b1f`](https://github.com/datGryphon/tinytalk/commit/a2f4b1f7acd890baa4cc8fd70e68cd49774b7d80))

- Consolidate backend development and CI environments
  ([#14](https://github.com/datGryphon/tinytalk/pull/14),
  [`a2f4b1f`](https://github.com/datGryphon/tinytalk/commit/a2f4b1f7acd890baa4cc8fd70e68cd49774b7d80))

- Declare typed engine state and align adapter contracts
  ([#14](https://github.com/datGryphon/tinytalk/pull/14),
  [`a2f4b1f`](https://github.com/datGryphon/tinytalk/commit/a2f4b1f7acd890baa4cc8fd70e68cd49774b7d80))

- Define concrete synthesis and telemetry types
  ([#14](https://github.com/datGryphon/tinytalk/pull/14),
  [`a2f4b1f`](https://github.com/datGryphon/tinytalk/commit/a2f4b1f7acd890baa4cc8fd70e68cd49774b7d80))

- Flatten Chatterbox Nano conditioning preparation
  ([#15](https://github.com/datGryphon/tinytalk/pull/15),
  [`d0ed159`](https://github.com/datGryphon/tinytalk/commit/d0ed1590e4568fea8ce0f826f458ba00fb276041))

- Flatten NeuTTS voice preload and rely on validated presets
  ([#15](https://github.com/datGryphon/tinytalk/pull/15),
  [`d0ed159`](https://github.com/datGryphon/tinytalk/commit/d0ed1590e4568fea8ce0f826f458ba00fb276041))

- Flatten OmniVoice reference preparation ([#15](https://github.com/datGryphon/tinytalk/pull/15),
  [`d0ed159`](https://github.com/datGryphon/tinytalk/commit/d0ed1590e4568fea8ce0f826f458ba00fb276041))

- Flatten Silencio-2 voice preload ([#15](https://github.com/datGryphon/tinytalk/pull/15),
  [`d0ed159`](https://github.com/datGryphon/tinytalk/commit/d0ed1590e4568fea8ce0f826f458ba00fb276041))

- Keep voice configuration imports at module scope
  ([#15](https://github.com/datGryphon/tinytalk/pull/15),
  [`d0ed159`](https://github.com/datGryphon/tinytalk/commit/d0ed1590e4568fea8ce0f826f458ba00fb276041))

- Load voice config once in backend and simplify synthesis path
  ([#15](https://github.com/datGryphon/tinytalk/pull/15),
  [`d0ed159`](https://github.com/datGryphon/tinytalk/commit/d0ed1590e4568fea8ce0f826f458ba00fb276041))

- Make speech engines inherit shared synthesis
  ([#14](https://github.com/datGryphon/tinytalk/pull/14),
  [`a2f4b1f`](https://github.com/datGryphon/tinytalk/commit/a2f4b1f7acd890baa4cc8fd70e68cd49774b7d80))

- Model synthesis data with Pydantic ([#14](https://github.com/datGryphon/tinytalk/pull/14),
  [`a2f4b1f`](https://github.com/datGryphon/tinytalk/commit/a2f4b1f7acd890baa4cc8fd70e68cd49774b7d80))

- Rebuild named voice configuration and request lifecycle
  ([#15](https://github.com/datGryphon/tinytalk/pull/15),
  [`d0ed159`](https://github.com/datGryphon/tinytalk/commit/d0ed1590e4568fea8ce0f826f458ba00fb276041))

- Reuse validated voice configuration instead of reloading JSON
  ([#15](https://github.com/datGryphon/tinytalk/pull/15),
  [`d0ed159`](https://github.com/datGryphon/tinytalk/commit/d0ed1590e4568fea8ce0f826f458ba00fb276041))

- Share chunk synthesis and assembly ([#14](https://github.com/datGryphon/tinytalk/pull/14),
  [`a2f4b1f`](https://github.com/datGryphon/tinytalk/commit/a2f4b1f7acd890baa4cc8fd70e68cd49774b7d80))

- Simplify voice preset validation into required/allowed field checks
  ([#15](https://github.com/datGryphon/tinytalk/pull/15),
  [`d0ed159`](https://github.com/datGryphon/tinytalk/commit/d0ed1590e4568fea8ce0f826f458ba00fb276041))

- Use direct typed voice selection rather than dynamic keyword branching
  ([#15](https://github.com/datGryphon/tinytalk/pull/15),
  [`d0ed159`](https://github.com/datGryphon/tinytalk/commit/d0ed1590e4568fea8ce0f826f458ba00fb276041))

### Testing

- Exercise voice validation, defaults and request isolation
  ([#15](https://github.com/datGryphon/tinytalk/pull/15),
  [`d0ed159`](https://github.com/datGryphon/tinytalk/commit/d0ed1590e4568fea8ce0f826f458ba00fb276041))

- Verify named voice API selection and rejection
  ([#15](https://github.com/datGryphon/tinytalk/pull/15),
  [`d0ed159`](https://github.com/datGryphon/tinytalk/commit/d0ed1590e4568fea8ce0f826f458ba00fb276041))


## v0.1.3 (2026-10-12)

### Bug Fixes

- Allow app directory creation during prestart
  ([#12](https://github.com/datGryphon/tinytalk/pull/12),
  [`fbda249`](https://github.com/datGryphon/tinytalk/commit/fbda2490084b40253f7f064f2f8edad1d9421c19))

- Nix runtime packaging ([#12](https://github.com/datGryphon/tinytalk/pull/12),
  [`fbda249`](https://github.com/datGryphon/tinytalk/commit/fbda2490084b40253f7f064f2f8edad1d9421c19))

- **nix**: Allow CUDA module checks ([#12](https://github.com/datGryphon/tinytalk/pull/12),
  [`fbda249`](https://github.com/datGryphon/tinytalk/commit/fbda2490084b40253f7f064f2f8edad1d9421c19))

- **nix**: Assemble complete CUDA toolkit for native builds
  ([#12](https://github.com/datGryphon/tinytalk/pull/12),
  [`fbda249`](https://github.com/datGryphon/tinytalk/commit/fbda2490084b40253f7f064f2f8edad1d9421c19))

- **nix**: Expose CUDA static runtime libraries during nvcc linking
  ([#12](https://github.com/datGryphon/tinytalk/pull/12),
  [`fbda249`](https://github.com/datGryphon/tinytalk/commit/fbda2490084b40253f7f064f2f8edad1d9421c19))

- **nix**: Expose SPIRV-Headers CMake package to Vulkan builds
  ([#12](https://github.com/datGryphon/tinytalk/pull/12),
  [`fbda249`](https://github.com/datGryphon/tinytalk/commit/fbda2490084b40253f7f064f2f8edad1d9421c19))

- **nix**: Expose split CUDA development headers to nvcc
  ([#12](https://github.com/datGryphon/tinytalk/pull/12),
  [`fbda249`](https://github.com/datGryphon/tinytalk/commit/fbda2490084b40253f7f064f2f8edad1d9421c19))

- **nix**: Keep dependency builds out of system activation
  ([#12](https://github.com/datGryphon/tinytalk/pull/12),
  [`fbda249`](https://github.com/datGryphon/tinytalk/commit/fbda2490084b40253f7f064f2f8edad1d9421c19))

- **nix**: Own CMake and Vulkan build environment in TinyTalk
  ([#12](https://github.com/datGryphon/tinytalk/pull/12),
  [`fbda249`](https://github.com/datGryphon/tinytalk/commit/fbda2490084b40253f7f064f2f8edad1d9421c19))

- **nix**: Provide Vulkan shader validator ([#12](https://github.com/datGryphon/tinytalk/pull/12),
  [`fbda249`](https://github.com/datGryphon/tinytalk/commit/fbda2490084b40253f7f064f2f8edad1d9421c19))

- **nix**: Reuse successful installs across ordinary restarts
  ([#12](https://github.com/datGryphon/tinytalk/pull/12),
  [`fbda249`](https://github.com/datGryphon/tinytalk/commit/fbda2490084b40253f7f064f2f8edad1d9421c19))

- **nix**: Support explicit llama.cpp CPU, Vulkan and CUDA backends
  ([#12](https://github.com/datGryphon/tinytalk/pull/12),
  [`fbda249`](https://github.com/datGryphon/tinytalk/commit/fbda2490084b40253f7f064f2f8edad1d9421c19))

### Documentation

- Describe clean-install lifecycle and GPU build ownership
  ([#12](https://github.com/datGryphon/tinytalk/pull/12),
  [`fbda249`](https://github.com/datGryphon/tinytalk/commit/fbda2490084b40253f7f064f2f8edad1d9421c19))

- **nix**: Explain startup compilation and health readiness
  ([#12](https://github.com/datGryphon/tinytalk/pull/12),
  [`fbda249`](https://github.com/datGryphon/tinytalk/commit/fbda2490084b40253f7f064f2f8edad1d9421c19))

### Refactoring

- **nix**: Reuse persistent uv environment ([#12](https://github.com/datGryphon/tinytalk/pull/12),
  [`fbda249`](https://github.com/datGryphon/tinytalk/commit/fbda2490084b40253f7f064f2f8edad1d9421c19))

- **nix**: Use upstream CUDA toolkit ([#12](https://github.com/datGryphon/tinytalk/pull/12),
  [`fbda249`](https://github.com/datGryphon/tinytalk/commit/fbda2490084b40253f7f064f2f8edad1d9421c19))

### Testing

- Validate prestart app bootstrap wiring ([#12](https://github.com/datGryphon/tinytalk/pull/12),
  [`fbda249`](https://github.com/datGryphon/tinytalk/commit/fbda2490084b40253f7f064f2f8edad1d9421c19))

- **nix**: Assert CUDA static runtime linker path
  ([#12](https://github.com/datGryphon/tinytalk/pull/12),
  [`fbda249`](https://github.com/datGryphon/tinytalk/commit/fbda2490084b40253f7f064f2f8edad1d9421c19))

- **nix**: Assert native and Vulkan build inputs
  ([#12](https://github.com/datGryphon/tinytalk/pull/12),
  [`fbda249`](https://github.com/datGryphon/tinytalk/commit/fbda2490084b40253f7f064f2f8edad1d9421c19))

- **nix**: Assert nonblocking launcher and bounded retries
  ([#12](https://github.com/datGryphon/tinytalk/pull/12),
  [`fbda249`](https://github.com/datGryphon/tinytalk/commit/fbda2490084b40253f7f064f2f8edad1d9421c19))

- **nix**: Assert nvcc sees CUDA runtime and CCCL headers
  ([#12](https://github.com/datGryphon/tinytalk/pull/12),
  [`fbda249`](https://github.com/datGryphon/tinytalk/commit/fbda2490084b40253f7f064f2f8edad1d9421c19))

- **nix**: Check GPU CMake prefix without affecting CPU
  ([#12](https://github.com/datGryphon/tinytalk/pull/12),
  [`fbda249`](https://github.com/datGryphon/tinytalk/commit/fbda2490084b40253f7f064f2f8edad1d9421c19))

- **nix**: Preserve installed app on restart ([#12](https://github.com/datGryphon/tinytalk/pull/12),
  [`fbda249`](https://github.com/datGryphon/tinytalk/commit/fbda2490084b40253f7f064f2f8edad1d9421c19))

- **nix**: Verify unified CUDA toolkit configuration
  ([#12](https://github.com/datGryphon/tinytalk/pull/12),
  [`fbda249`](https://github.com/datGryphon/tinytalk/commit/fbda2490084b40253f7f064f2f8edad1d9421c19))


## v0.1.2 (2026-09-29)

### Bug Fixes

- Deploy TinyTalk from a writable application directory
  ([#9](https://github.com/datGryphon/tinytalk/pull/9),
  [`17d6fe8`](https://github.com/datGryphon/tinytalk/commit/17d6fe868cd26728a4c71878e4048b993dd1b77c))

### Continuous Integration

- Automate release preparation and publishing ([#9](https://github.com/datGryphon/tinytalk/pull/9),
  [`17d6fe8`](https://github.com/datGryphon/tinytalk/commit/17d6fe868cd26728a4c71878e4048b993dd1b77c))

- Use Python Semantic Release ([#9](https://github.com/datGryphon/tinytalk/pull/9),
  [`17d6fe8`](https://github.com/datGryphon/tinytalk/commit/17d6fe868cd26728a4c71878e4048b993dd1b77c))

- Validate pull request titles ([#9](https://github.com/datGryphon/tinytalk/pull/9),
  [`17d6fe8`](https://github.com/datGryphon/tinytalk/commit/17d6fe868cd26728a4c71878e4048b993dd1b77c))

### Documentation

- Document backend model licenses and Nix setup
  ([#9](https://github.com/datGryphon/tinytalk/pull/9),
  [`17d6fe8`](https://github.com/datGryphon/tinytalk/commit/17d6fe868cd26728a4c71878e4048b993dd1b77c))

- Link upstream model license sources ([#9](https://github.com/datGryphon/tinytalk/pull/9),
  [`17d6fe8`](https://github.com/datGryphon/tinytalk/commit/17d6fe868cd26728a4c71878e4048b993dd1b77c))

### Refactoring

- Remove Nix runtime path indirection ([#9](https://github.com/datGryphon/tinytalk/pull/9),
  [`17d6fe8`](https://github.com/datGryphon/tinytalk/commit/17d6fe868cd26728a4c71878e4048b993dd1b77c))

- Remove obsolete Python environment path ([#9](https://github.com/datGryphon/tinytalk/pull/9),
  [`17d6fe8`](https://github.com/datGryphon/tinytalk/commit/17d6fe868cd26728a4c71878e4048b993dd1b77c))

- Simplify Nix deployment ([#9](https://github.com/datGryphon/tinytalk/pull/9),
  [`17d6fe8`](https://github.com/datGryphon/tinytalk/commit/17d6fe868cd26728a4c71878e4048b993dd1b77c))

- Simplify Nix runtime setup ([#9](https://github.com/datGryphon/tinytalk/pull/9),
  [`17d6fe8`](https://github.com/datGryphon/tinytalk/commit/17d6fe868cd26728a4c71878e4048b993dd1b77c))

- Trim Nix runtime plumbing ([#9](https://github.com/datGryphon/tinytalk/pull/9),
  [`17d6fe8`](https://github.com/datGryphon/tinytalk/commit/17d6fe868cd26728a4c71878e4048b993dd1b77c))

### Testing

- Exercise Nix upgrade lifecycle ([#9](https://github.com/datGryphon/tinytalk/pull/9),
  [`17d6fe8`](https://github.com/datGryphon/tinytalk/commit/17d6fe868cd26728a4c71878e4048b993dd1b77c))


## v0.1.1 (2026-09-27)

### Bug Fixes

- Reuse TinyTAuK conditioning across retries
  ([`a51c7aa`](https://github.com/datGryphon/tinytalk/commit/a51c7aadf37e22d32eb7cb397bf91d6fa5b0f7e3))

- Tinytauk conditioning reuse
  ([`a51c7aa`](https://github.com/datGryphon/tinytalk/commit/a51c7aadf37e22d32eb7cb397bf91d6fa5b0f7e3))


## v0.1.0 (2026-09-24)

- Initial Release
