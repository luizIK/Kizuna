# Tools

Tools for Fritchii and other subprojects. When adding new scripts please use `#!/bin/sh -e` or `#!/usr/bin/env <interpreter>` (for `.py`, `.rb`, or `.perl`). Keep scripts POSIX compliant (i.e not require hard `bash` to run, just plain old `sh`).

## Third-Party

- [CPMUtil Scripts](./cpm)

## Binaries

- `maxwell-spirv`: Converts Maxwell shaders (dumped from `.ash` files) into SPIR-V code (emitted into STDOUT).
- `maxwell-disas`: Dumb raw Maxwell disassembler.
- `maxwell-ir`: Dump generated IR of Maxwell shaders.

## Scripts

- `generate_converters.py`: Generates converters for given formats of textures (C++ helper).
- `svc_generator.py`: Generates the files `src/core/hle/kernel/svc.cpp` and `src/core/hle/kernel/svc.h` based off prototypes.
- `shellcheck.sh`: Ensure POSIX compliance (and syntax sanity) for all tools in this directory and subdirectories.
- `llvmpipe-run.sh`: Sets environment variables needed to run any command (or Fritchii) with llvmpipe.
- `optimize-assets.sh`: Optimizes PNG assets with OptiPng.
- `update-cpm.sh`: Updates CPM.cmake to the latest version.
- `dtrace-tool.sh`
- `lanczos-gen.pl`: Generates constants for the Lanczos filter.
- `clang-format.sh`: Runs `clang-format` on the entire codebase.
    * Requires: clang
- `cpp-lint.sh`: Homemade dumb C++ linter.
- `fuzzsettings.cpp`: Fuzz settings files.
