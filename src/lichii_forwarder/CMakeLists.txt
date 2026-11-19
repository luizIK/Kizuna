# SPDX-License-Identifier: GPL-3.0-or-later

# lichii.exe: starts fritchii.exe next to it. Fritchii v5.x updaters require lichii.exe in the
# package and start it after updating, and old shortcuts point to it.
add_executable(lichii_forwarder WIN32 main.cpp forwarder.rc)
set_target_properties(lichii_forwarder PROPERTIES OUTPUT_NAME "lichii")
target_compile_definitions(lichii_forwarder PRIVATE UNICODE _UNICODE WIN32_LEAN_AND_MEAN NOMINMAX)
# Built with the app, so a build of the app always refreshes it.
add_dependencies(yuzu lichii_forwarder)
