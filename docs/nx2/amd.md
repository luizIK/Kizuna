# AMD GPUs and Switch 2 titles

Switch 2 games are written for an NVIDIA GPU. Their native shaders and command streams rely on
NVIDIA behaviour in places where Vulkan leaves the result undefined, and the translated Vulkan
usually does the same thing on an NVIDIA host by luck. AMD drivers return something else, so a
missing or wrong emulation detail tends to show up on AMD first: NaN haze, garbage pixels, black
scenes, driver timeouts. This page lists the differences found so far, what Fritchii does about
each, and how to investigate a new one.

Ocarina of Time (`0400486024DEC000`) is the main test case, but the rules apply to every title.

## Differences and how they are handled

"Open" in the last column means the fix is not in yet: the emulator still behaves
the old way there.

| Guest (NVIDIA) behaviour | Vulkan, and what AMD does | What Fritchii does | Status |
|---|---|---|---|
| `FMNMX` returns the operand that is not NaN, and saturation turns NaN into 0 | `FMin`, `FMax` and `FClamp` leave NaN undefined; AMD returns the NaN, and blurred or accumulated targets spread it over the screen | Min, max, clamp and saturate are emitted as `NMin`/`NMax`, clamp as `NMin(NMax(x, lo), hi)` | fixed |
| A bound render target that the fragment shader doesn't export keeps its old value | The attachment gets an undefined value; AMD writes whatever its export registers hold (format maximum, NaN) | The colour write mask, in the pipeline and in the dynamic mask, leaves out the components the fragment shader doesn't write | fixed |
| Storage buffers start at the guest's alignment, so 64- and 128-bit accesses are aligned | Bindings start at `minStorageBufferOffsetAlignment`, which is 4 bytes on AMD; vector views of the binding read the wrong words | Misaligned 64/128-bit accesses are split into 32-bit words | fixed |
| Demoted fragments stop writing where the guest's control flow says | Moving demotes to the end of the program over nested returns let helper lanes keep writing; AMD drew a black scene | Nested returns are counted before demotes move | fixed |
| Fragment subgroup shuffles | AMD's proprietary driver gets them wrong | Broadcast fallback on that driver | fixed |
| Depth-compare samples | Undefined on colour or integer images; AMD returned green | Depth-compare images stay floating point. Binding the null image for depth-compare samples of uncomparable images is proposed | partly fixed, rest open |
| Task and mesh programs | Native task/mesh draws hang AMD GPUs. RADV resets the GPU within minutes on RX 6000, 7000 and 9000 cards, often taking the desktop session down with it; Mesa 26.2.6 still hangs, though it recovers more often. AMD's Windows driver times out (TDR) on the Radeon 780M | On AMD drivers (RADV and AMD's own), task/mesh draws take the compute fallback. It can be slower in scenes full of foliage, and it skips the mesh programs it can't run (per-primitive outputs, primitives other than triangles, layered output); Ocarina of Time testers report all geometry drawn. The Mesh Shaders setting (GraphicsExtra tab, Vulkan Extensions) picks Native or Compute Fallback instead of Automatic, also per game | worked around |
| Native FP16 arithmetic | Turned off on AMD as a driver workaround | Turned off, with no override yet. An `EDEN_AMD_FP16=1` switch to keep it for testing is proposed | open |

Not specific to AMD, but easy to mistake for an AMD bug:

- Ocarina's volumetric fog reads its sky-light probes 104-122 MiB past a pointer it loads from a
  table. With the 64 MiB cap for read-only table pointers every probe read returned 0, the fog lost
  its sky ambient and outdoor scenes turned into a white haze, on every vendor. A ported fix binds
  read-only table pointers to the end of their mapping instead.
- Two of Ocarina's compute shaders take minutes to optimise in AMD's driver; a ported fix builds
  them unoptimised on that driver.

## Known open AMD issues

Check these before filing a new one.

| Symptom |
|---|
| Neon-green canopy or tint in Kokiri Forest |
| Lighting flicker |
| Driver timeout or device lost at gameplay start |
| Out of device memory on 8 GB cards |
| Frame rate (one GPU thread at 99%) |

## Settings for AMD users

- Keep the split GPU thread on (`EDEN_NX2_SPLIT=2`, the default in `nx2_switches.txt`) and GPU
  accuracy below High, which turns the split off. Textures flickering in that setup on an RX 9070
  XT were fixed by a ported change.
- Logging at `*:Info` costs CPU time in busy scenes. Use `*:Error` when you compare frame rates.
- The first run compiles every shader and can stall for minutes; later runs load the pipeline
  cache.

## Investigating an AMD-only bug

1. Reproduce it with the live debug server (`EDEN_NX2_DEBUG_SERVER=127.0.0.1:6545`, see
   [debug-mcp.md](./debug-mcp.md)) and take a `capture_frame`. Find the first pass whose output
   holds NaN, values at the format maximum, or zeros where there should be data. The manifest's
   per-channel statistics show this without opening every image, but only for formats the dump
   decodes: other targets are listed with `"skipped":"format"` and multisampled ones with
   `"skipped":"msaa"`, so check those passes separately.
2. Bisect that pass's shaders with `reject_shaders`, one screenshot per step.
3. For the draw or dispatch you find, ask what Vulkan leaves undefined there: NaN in min, max or
   clamp, outputs the shader doesn't write, misaligned or out-of-bounds storage access, values
   read before they are written, helper invocations, image views of the wrong kind. That is where
   AMD and NVIDIA differ.
4. Check the bindings as well as the shader. Reads past the end of a buffer bound too small return
   zeros or other values from inside the buffer under robust buffer access, so a plausible value
   doesn't prove the binding is large enough. Ocarina's fog read zeros this way.
5. If you can, compare with an NVIDIA run. A difference that also shows up there is an emulation
   bug, not an AMD one.
