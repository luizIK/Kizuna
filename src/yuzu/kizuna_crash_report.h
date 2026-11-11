// SPDX-FileCopyrightText: Copyright 2026 Kizuna contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

namespace KizunaCrash {

/// Windows: a native crash (access violation, driver fault, out of commit...) ends the process
/// without a word in the log. This writes, into the crash dumps folder (user\crash_dumps), a
/// kizuna-crash-<time>.txt with the exception code and address, the module and offset it is in, a
/// short stack and how much of the system's commit was left, and a kizuna-crash-<time>.dmp
/// minidump. Does nothing on other systems.
void Install();

} // namespace KizunaCrash
