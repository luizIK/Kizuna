# Kizuna

Kizuna is a personal fork of **Fritchii v5.8.4** (itself derived from Lichii and Eden/yuzu),
tuned for a 8 GB GPU (AMD RX 6600). All credit for the emulator goes to the authors of those
projects; this fork keeps their GPLv3 license (`LICENSE.txt`) and copyright notices.

## What differs from Fritchii v5.8.4

- **Out-of-memory recovery for render targets.** When a render target cannot be allocated
  (`VK_ERROR_OUT_OF_DEVICE_MEMORY`), the texture cache frees memory right away (emergency
  collection, waits for the GPU, drops the image pool) and looks the target up again, instead of
  binding the null image view. Draws and clears whose target still cannot be made are skipped.
- `ImageView::IsRescaled()` tolerates the null view.
- **Less pressure on a card at its budget.** The texture cache keeps 1/6 of the driver's VRAM
  budget free (at most 1.25 GiB) instead of 1/8 (1 GiB), gives the image pool's idle images back
  once it is past its critical threshold, and logs a "Memory census" (what the cache holds, the
  largest images, the main formats) the first four times device memory runs out.
- **No abort for a buffer that does not fit.** Device-local buffers over the budget are made
  anyway and, failing that, in host memory (images already had this fallback), instead of
  throwing out of the GPU thread.
- **A last word before dying.** An exception nobody catches is written to the log
  (`Fatal: uncaught ...`) before the process aborts, so a crash leaves its cause behind.
- The update checker is off in the Windows build, so it never offers upstream builds over this one.

## Keeping the "Preparing shaders" cache valid

The marker of a finished preparation (`precompiled_prepared.fork-v54.txt`) holds the precompiled
file's size and last write time, the driver, the shader stamp and the GPU. The time has to be
read the way the Fritchii (MSVC) builds read it, as FILETIME ticks (100 ns since 1601):
the MinGW standard library counts nanoseconds from another epoch and drops the fractions of a
second, so a marker written by Fritchii never matched and the whole preparation ran again.
`VideoCommon::FileWriteTicks` (`pipeline_cache_files.cpp`) reads it with the Windows API on
MinGW; the marker, the warm-up stamps and the merge log use it.

Upstream stamps the shader preparation (`NX2_SHADER_STAMP`, `CMakeModules/Nx2ShaderStamp.cmake`)
with a hash of the shader recompiler and of `vk_graphics_pipeline.*`, `vk_compute_pipeline.*`,
`fixed_pipeline_state.*` and `nx2_precompile.*`. Any edit to those files changes the stamp, and
the first start then rebuilds every pipeline (1 to 2 hours, about 120,000 of them). Kizuna avoids
this in two ways:

- Prefer changes outside those files.
- When a change in one of them cannot alter a compiled pipeline (a guard in the draw path, say),
  wrap it in `// KIZUNA-STAMP-NEUTRAL-BEGIN` / `// KIZUNA-STAMP-NEUTRAL-END` (one block per
  file, each marker alone on its line). The stamp script hashes the file without that block, so
  the stamp stays equal to upstream's and a shader cache built by Fritchii v5.8.4 is reused.

Never mark code that changes shader translation or pipeline creation as neutral.

## Building

The `kizuna-windows` workflow (`.github/workflows/kizuna-windows.yml`) builds the Windows (MinGW clang, PGO)
zip on GitHub Actions. Run it from the Actions tab, or push to the `kizuna` branch. The zip is
attached to the run and published as a pre-release.
