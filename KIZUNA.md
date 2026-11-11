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
- The update checker is off in the Windows build, so it never offers upstream builds over this one.

## Keeping the "Preparing shaders" cache valid

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
