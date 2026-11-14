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
- **Shader-compile freezes ("Building: N shader(s)").** A draw into a target the game keeps
  (shadow maps, probes) waits for its pipeline; the builders run below the guest cores, so that
  build shared a core with up to ten others (0.3 to 5 s on a Ryzen 5 5600). `PipelineBuild::Wait`
  now raises the thread building it (`Common::RaiseThread`, THREAD_PRIORITY_HIGHEST) until it is
  done. v1.8 to v1.13 also kept our VkPipelineCache file on AMD (`EDEN_NX2_AMD_APP_CACHE`); AMD's
  driver puts only keys in it (~5 KiB a pipeline), nothing got faster, and it is off again.
- **Kept files read ahead.** A new pipeline's kept infos and recorded state are read on the GPU
  thread; from a cold cache on a busy, nearly full disk a few small reads took up to 1.3 s with
  the thread idle. After the kept index loads, `KeptPrefetch` reads the kept and precompiled files
  once at low I/O priority into the system's file cache (`EDEN_NX2_KEPT_PREFETCH=0` skips it).
- **No driver compiles beside the game on AMD** (`DriverWarmWanted`, `EDEN_NX2_DRIVER_WARM`): the
  driver's ~1 GiB cache cannot hold the preparation's 120k pipelines.
- **Hotkeys.** Ctrl+U switches Ocarina between 30 and 60 FPS at once (with a notice over the
  game); its high refresh mode moved to Ctrl+Shift+U; Ctrl+G turns frame generation on and off.
- **AMD FSR and frame generation.** The Graphics button has "Ocarina Upscaler (FSR)" (Game, No
  sharpening, AMD FSR 3.1, AMD FSR 4; read when the game starts) and "Frame Generation" (off, 2x,
  3x, 4x; live). Fritchii turned frame generation off for its builds (`Settings::FrameGenActive`)
  and hid its settings; Kizuna builds offer it, off by default, with the settings in Graphics.

- **Depth targets redrawn every frame skip new pipelines.** Shadow cascades (1024x1000 D16) and a
  half-resolution depth (960x540 D32) are redrawn every frame without a full clear, so the skip
  rule (`EDEN_NX2_ASYNC_DISCARD=2`) counted them as Persistent and their draws waited for every
  new pipeline: seconds of freeze entering a new area. A depth target written in two render
  frames in a row (`RedrawnEveryFrame`, `ImageBase::previous_write_frame`) now counts as
  Recurring: a new caster's shadow is missing for a few frames instead.
  `EDEN_NX2_SKIP_REDRAWN_DEPTH=0` restores the wait. Since v1.11 the same holds for colour targets
  (HDR probes, the HDR scene without a full clear) and for targets whose clear history was reset
  (FirstTime); `EDEN_NX2_SKIP_REDRAWN_COLOR=0` restores the wait for colour. Layers rendered once
  (text, probes captured once) are not written in consecutive frames and still wait.
- **Fewer first-time waits (v1.15).** Scene-format targets (depth, float, 10-bit;
  `IsSceneTargetFormat`) written twice within 8 render frames count as Recurring
  (`RedrawnPeriodically`: the reflection probes draw their 128x128 faces in turns;
  `EDEN_NX2_SKIP_PERIODIC=0` waits). A new scene-format target of at least 256x256 whose
  (format, width, height) was seen per frame before counts as Recurring from its first frame
  (`per_frame_shapes`; entering an area re-creates the scene's targets, and 21 first-time waits
  held the game 4.3 s at the Kakariko church; `EDEN_NX2_SHAPE_PRIOR=0` waits). 8-bit layers
  (text, UI) and small targets still wait. The log lists first-time targets that wait, as it did
  persistent ones.
- **Compiled pipelines kept by the emulator (v1.15, `VK_KHR_pipeline_binary`).** AMD's driver
  keeps ~1 GiB of compiled pipelines, so every boot compiled again all the pipelines earlier
  sessions used (4,858 in 449 s, more each session) and play compiled pipelines it had compiled
  before. Pipelines created through `vk::Device` are now captured
  (`VK_PIPELINE_CREATE_2_CAPTURE_DATA_BIT_KHR`) and appended to
  `shader/kizuna_pipeline_binaries.bin` under the driver's key for their create info; the next
  creation with that key builds from the binaries (`Vulkan::PipelineBinaryCache`). The file starts
  with the driver's global key and starts over for another driver; a record cut short ends it;
  at most 2 GiB. The preparation's threads look up but do not capture. On by default for AMD's
  Windows driver; `EDEN_NX2_PIPELINE_BINARIES=0` turns it off, `=1` on for any driver. Deleting
  the file resets it. The first session after updating captures, the next ones load.
- **Uneven ASTC kept as BC (v1.16).** On AMD an ASTC image with uneven block mips whose even
  levels stop too early was RGBA8 (4x BC3, 8x BC1). It is now BC at the size, up to 1/8 larger,
  with the most even levels (683 -> 768, 1365 -> 1536), resampled by the BC encoder
  (`GetPaddedBcShape`, `bc_encoder.comp`; flag `HostResized`: never copied to or from, uploaded
  whole). Ocarina's handheld textures are 2/3 of the docked sizes, so that was most of them: two
  1365x1365x90 terrain arrays held 1.8 GiB of a 5 GiB budget, the device sat over the
  collector's critical threshold and waits of 3.5 s for the host GPU froze the game.
  `EDEN_NX2_ASTC_UNEVEN_PAD=0` keeps RGBA8.
- **Calmer collector (v1.16).** Past the critical threshold the collector took images idle for
  10 frames (11,765 ASTC decodes in 450 s of Hyrule Field). With no allocation refused in the
  last 600 frames it now takes those idle for 120 (`EDEN_NX2_TC_AGGRESSIVE_AGE`).
- **From Lichii (v1.16).** #1452: mesh emulation's staging buffers are held until the submission
  that writes them (AMD draws task/mesh shaders through it). #1451: a zero-page trim pass that
  runs out of memory is skipped instead of ending the emulator.
- **v1.17.** The last piece of Lichii #1516 the base lacked: the occlusion query lists flushed at a
  fence are reused instead of allocated at each one. The rest of #1516 was already in the base
  (fencing thread wake, operation lists, master semaphore cache lines, query pool reset on the
  worker, report operation size).
- **v1.18.** `clean_words` (an optimization the base ships switched off) is on: a buffer upload
  walks only the dirty words of the memory tracker (`SynchronizeBuffer`,
  `ForEachDirtyUploadRange`). Measured on a Ryzen 5 5600 in Hyrule Field with
  `EDEN_NX2_GPU_PROFILE=1`: "bind stage buffers" 3.74 -> 1.83 us a draw, "compute synchronize"
  0.93 -> 0.37, the back thread's idle time 131 -> 209 ms/s, no wrong geometry in Kokiri Forest
  (where the family of changes it belongs to, Lichii #161, once broke). Built in as
  `EDEN_NX2_PERF_ON=clean_words`; `EDEN_NX2_PERF_OFF=clean_words` in `nx2_switches.txt` turns it
  off. What was tried on the way and gave nothing on this PC, so it stays off:
  - Lichii #1486 (list-fed grid macro calls taken without their words) was not ported. It removes
    the front's wait for GPU-written command lists (~100 ms/s here, "list GPU wait"), not the
    waits that dominate ("fence release", "macro dma read"), and Lichii reverted it in #1651 for
    white flashing in Ocarina.
  - `EDEN_NX2_EXP53=1` (the back evaluates the conditional render macro): the macro is found
    (0xe2c) but is not what waits; "macro dma read" stayed at 250-350 ms/s.
  - `EDEN_NX2_EXP12=1` (acquires wait for their writers only): "fence release" 250 -> 55 ms/s,
    but "fence writers" took the 200: the acquires need the back's results.
  - `EDEN_NX2_EXP17=1` with `EDEN_NX2_EXP18=7` (uniform reuse, binding memos): no change.
  - `EDEN_NX2_SAMPLE_PROFILE` found the `clean_words` cost, but a session crashed 5 s after the
    sampler started under load. Read its file with the linker map from `Kizuna-symbols.zip`
    (`tools/nx2/sample_report.py` expects a PDB).
  Still open: the front thread stays saturated waiting for the back (~540 ms/s); Lichii's "fast
  draws" (#1441, #1519, #1521, #1586) cut the back's cost per draw but #1519 edits files of the
  shader stamp.
- **No driver compiles beside the game on AMD.** A quick preparation used to have the driver
  compile all 120,124 pipelines beside the game on three threads (hours of it). AMD's cache keeps
  ~1 GiB, so each of those pushed out a pipeline the player had used, and the threads competed
  with the game's own builds. Off on AMD's own driver (`DriverWarmWanted`;
  `EDEN_NX2_DRIVER_WARM=1` turns it back on): a pipeline compiles when a draw first needs it.
- **Windows out of memory (commit).** Windows refuses memory past its commit limit (RAM plus
  virtual memory) whatever the card has free, and device memory counts against it too. Under
  3 GiB left a notice over the game says so (`MainWindow::WarnLowSystemMemory`). The log lists
  the programs holding the commit at the start and every minute while under 8 GiB is left
  (`Common::TopCommitPrograms`, on a thread of its own), and the crash report carries the last
  list. A staging buffer that still does not fit after the idle ones are freed waits for the GPU
  to finish, frees the rest and is tried three more times before the emulator gives up.

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
