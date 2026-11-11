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
- **No abort for a buffer that does not fit.** The allocator keeps a 192 MiB block of device
  memory in reserve (taken again by the texture cache whenever it has a gigabyte of room under its
  critical threshold; none on cards with a budget under 3 GiB). A buffer the budget refuses first
  gets the reserve back and tries again, then goes over the budget, then (device-local buffers) to
  host memory, then waits up to 4 s for memory to come back, and only then throws. The log line
  says which step worked or failed (Vulkan result of each) and how much system commit was left.
  Build 6 died on a 16 MiB buffer at 7650 MiB with a 5.6 GiB budget.
- **A lower ceiling on 8 GB cards.** On cards with a budget under 9 GiB the cache's critical
  threshold is at most 4.5 GiB (5 GiB before): Windows moved the budget by 2 GB mid-game.
  `EDEN_NX2_TC_CEILING_MIB` still lowers it further.
- **Memory trace on by default.** Every 10 s the log has a "Memory trace" line (the cache's
  images and bytes next to the device's usage) plus the process, DXGI and VMA lines, to see where
  the video memory goes. `EDEN_NX2_MEM_TRACE=0` turns it off, any other value makes it every 2 s.
- **A last word before dying.** An exception nobody catches is written to the log
  (`Fatal: uncaught ...`) before the process aborts, so a crash leaves its cause behind.
- **ASTC with uneven mips: BC, with the broken levels never sampled.** On AMD the BC1/BC3 copy
  of an ASTC texture whose mip chain has uneven block counts samples wrong from its first uneven
  level on (dark grid lines on distant tiled terrain, Hyrule Field). Fritchii kept all of them as
  RGBA8 (4-8x the memory: out of memory at Ocarina's scene changes); build 9 kept them as BC and
  showed the lines. Now they stay BC and their views stop before the first uneven level (768x768:
  levels 0-6, down to 12x12), so far away they are sampled a little sharper than asked and never
  from a broken level. Those whose last even level would still be over 32 texels (683x683, uneven
  from level 1) stay RGBA8. `EDEN_NX2_ASTC_UNEVEN_RGBA8=1` keeps all of them RGBA8, `=0` all BC.
- **Name.** The window title, About, the log's first line and the visible messages say "Kizuna"
  and "vX.Y" (`Common::DisplayBuildName/Version`, from the CI release tag `kizuna-N`). Inside,
  `g_build_name` stays "Fritchii", which selects the UI and settings defaults. The links and
  credits to Fritchii (its Discord menu entry, "Fritchii had it first") are left as they are.
- **Crash reports.** A native crash writes `user\crash_dumps\kizuna-crash-<time>.txt` (exception
  code and address, the module and offset, a short stack, system commit left) and a `.dmp`
  minidump (`src/yuzu/kizuna_crash_report.cpp`).
- **Quick shader preparation.** "Preparing shaders" only translates the 120,124 records and keeps
  their SPIR-V (all threads, minutes instead of ~2 hours); the AMD driver then compiles them beside
  the game on idle threads, resuming at every boot (`precompiled_driver.fork-v54.txt`), and a draw
  whose pipeline it has not reached builds it from the kept SPIR-V on the workers.
  `EDEN_NX2_PRECOMPILE_QUICK=0` brings back the full preparation. A cache the full preparation
  already finished is left as it is.
- **Versions.** Releases are `Kizuna vX.Y`, tagged `kizuna-vX.Y`: build 4 was v1.0 and each
  release is one minor up. The workflow's Version step picks the next one from the releases and
  titles the earlier `kizuna-build-N` releases Kizuna v1.0 to v1.5 (4 -> v1.0 ... 10 -> v1.5); they
  keep their tags, which the job's token may not move. New
  releases are full releases marked Latest (GitHub lists releases by tag name as text).
- **In-app updater.** Help > Check for Updates (and the check at startup) looks for newer
  `kizuna-vX.Y` releases of luizIK/Kizuna only, verifies the zip's SHA-256 from GitHub and
  installs it over the running folder (the `user` folder is never touched; Help > Roll Back brings
  the previous build back). Fritchii's releases are never offered.
- **No Fritchii links.** The Discord and Donate entries (Help menu, home page) are gone; About
  names Kizuna, credits Fritchii, Lichii, Eden and yuzu, and links to this repository and the
  license. lichii.exe's file description is "Kizuna" (Task Manager).
- **Built for Zen 3.** `-march=x86-64-v3 -mtune=znver3` on top of PGO.
- **Look.** The "Ink" theme: the logo's navy (#1a1f28) and cream (#f0ebdf) in place of Fritchii's
  near-black and violet (palette in `fritchii_ui.cpp`, widgets in `style.qss`). The logo
  (`fritchii/kizuna_logo.png`, transparent) sits at the top of the rail, on the Preparing shaders
  screen and in About; the home page has the Kizuna name. The app icon is `dist/kizuna.ico`
  (the rounded square with transparent corners, 16 to 256 px).
- **Home screen, console style** (`fritchii/home.cpp`): a top bar (logo, name, clock), a shelf
  of game tiles with the selected one larger and ringed in cream, the selected game's name, play
  time, Play and options, and a dock (Library, Saves, Screenshots, Settings, Help, Menu) with
  keyboard hints. Left/Right choose, Enter plays, Esc opens the menus, the Menu key the game's
  options; a click picks a tile, a click on the picked one opens it. Fritchii's left rail only
  shows beside the full game list, as the way back home.
- **Scene changes and VRAM.** ASTC transcodes reuse their RGBA8 decode images as soon as the GPU
  is done with them and wait past 192 MiB pending (Ocarina's town to Hyrule Field crashed out of
  memory before the field loaded); uneven-mip ASTC stays BC when its last even level is up to
  64 texels.
- **Crash reports** chain to MinGW's exception filter, so an uncaught C++ exception still reaches
  std::terminate and the log names it; each release carries `Kizuna-symbols.zip` (the linker map)
  to read the reports' `lichii.exe+offset`.

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
