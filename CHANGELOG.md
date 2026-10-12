# CHANGELOG

<!-- version list -->

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
