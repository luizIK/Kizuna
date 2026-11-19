// SPDX-License-Identifier: GPL-3.0-or-later

// FRITCHII_JIT_CHEAP_BARRIERS (boot, default 0): cheaper ordered loads and memory barriers on
// x86-64 hosts, ported from zortos293/Lichii #1327
// (https://github.com/zortos293/Lichii/pull/1327, ProphetWild's commit from #161 by 999sian).
//   0 = off: ordered loads are a locked xadd, every DMB is an mfence (the old code, unchanged).
//   1 = all of #1327: plain loads for LDAR/LDAPR/LDAXR (and A32 LDA/LDAEX) up to 64 bits, and
//       DMB *LD / DMB *ST emit nothing.
//   2 = only the plain ordered loads.
//   3 = only the DMB *LD / DMB *ST change.
// x86-64 keeps loads in order with later loads and stores, and stores in order with later stores
// (TSO), so an acquire load is a plain load and a load-only or store-only barrier has nothing to
// add. Ordered stores stay xchg (a full barrier), so STLR then LDAR stays ordered; a full DMB and
// every DSB keep their mfence; 128-bit ordered loads keep lock cmpxchg16b.

#pragma once

#include <cstdlib>

#include "common/logging.h"

namespace Dynarmic::Fritchii {

/// FRITCHII_JIT_CHEAP_BARRIERS, read once per process (0..3, anything else is 0).
inline int CheapBarriersMode() {
    static const int mode = [] {
        const char* text = std::getenv("FRITCHII_JIT_CHEAP_BARRIERS");
        const int value = text == nullptr ? 0 : std::atoi(text);
        if (value < 0 || value > 3) {
            LOG_WARNING(Core_ARM, "FRITCHII_JIT_CHEAP_BARRIERS={} is outside 0..3, using 0", value);
            return 0;
        }
        if (value != 0) {
            LOG_INFO(Core_ARM, "FRITCHII_JIT_CHEAP_BARRIERS={}: {}", value,
                     value == 1   ? "plain ordered loads, no barrier for DMB LD/ST"
                     : value == 2 ? "plain ordered loads"
                                  : "no barrier for DMB LD/ST");
        }
        return value;
    }();
    return mode;
}

/// Ordered and exclusive loads of up to 64 bits are a plain mov (modes 1 and 2).
inline bool PlainOrderedLoads() {
    const int mode = CheapBarriersMode();
    return mode == 1 || mode == 2;
}

/// DMB with CRm<1:0> = 01 (loads) or 10 (stores) emits nothing (modes 1 and 3).
inline bool SkipLoadStoreDmb() {
    const int mode = CheapBarriersMode();
    return mode == 1 || mode == 3;
}

}  // namespace Dynarmic::Fritchii
