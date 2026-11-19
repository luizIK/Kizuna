# Fritchii branding

Fritchii's own look (name, logo) is kept apart from emulator work so
other builds can take its fixes without it.

## Telling branding from fixes

Every branding-only commit carries the trailer `Fritchii-Branding: yes` and touches nothing that
changes emulation. To list or skip them:

```sh
git log --grep='^Fritchii-Branding: yes' --oneline           # branding commits
git log --invert-grep --grep='^Fritchii-Branding: yes' ...    # everything else (fixes, ports)
```

When cherry-picking fixes out of Fritchii, leave the branding commits out. A Claude doing the
merge can be told: "skip commits with the trailer Fritchii-Branding: yes".

## What branding changes

- `dist/fritchii.ico`, `dist/fritchii.png`: the logo, the pear (art by Freshdyno, made for Fritchii; `dist/lichii.*` hold the same art until the file rename).
- `src/yuzu/yuzu.rc`, `src/yuzu/yuzu.qrc`, `dist/qt_themes/default/default.qrc`: point the existing icon names at the Fritchii files.
- Version string and window title "Fritchii vN" (CMakeModules/GenerateSCMRev.cmake,
  src/qt_common/qt_common.cpp, src/yuzu/bootmanager.cpp), only when dist/frizti_version.txt exists.
- This folder.

The executable is still `lichii.exe` (it becomes `fritchii.exe` later); user data lives in
`%APPDATA%\fritchii` since v5.8.
