// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <type_traits>
#include <utility>
#include <vector>

#include "common/common_types.h"

namespace Common {

template <typename BaseAddr>
class MultiLevelPageTable final {
public:
    constexpr MultiLevelPageTable() = default;
    explicit MultiLevelPageTable(std::size_t address_space_bits, std::size_t first_level_bits,
                                 std::size_t page_bits);

    ~MultiLevelPageTable() noexcept;

    MultiLevelPageTable(const MultiLevelPageTable&) = delete;
    MultiLevelPageTable& operator=(const MultiLevelPageTable&) = delete;

    MultiLevelPageTable(MultiLevelPageTable&& other) noexcept
        : address_space_bits{std::exchange(other.address_space_bits, 0)},
          first_level_bits{std::exchange(other.first_level_bits, 0)}, page_bits{std::exchange(
                                                                          other.page_bits, 0)},
          first_level_shift{std::exchange(other.first_level_shift, 0)},
          first_level_chunk_size{std::exchange(other.first_level_chunk_size, 0)},
          first_level_map{std::move(other.first_level_map)}, base_ptr{std::exchange(other.base_ptr,
                                                                                    nullptr)},
          census_name{std::exchange(other.census_name, nullptr)} {}

    MultiLevelPageTable& operator=(MultiLevelPageTable&& other) noexcept {
        address_space_bits = std::exchange(other.address_space_bits, 0);
        first_level_bits = std::exchange(other.first_level_bits, 0);
        page_bits = std::exchange(other.page_bits, 0);
        first_level_shift = std::exchange(other.first_level_shift, 0);
        first_level_chunk_size = std::exchange(other.first_level_chunk_size, 0);
        alloc_size = std::exchange(other.alloc_size, 0);
        first_level_map = std::move(other.first_level_map);
        base_ptr = std::exchange(other.base_ptr, nullptr);
        census_name = std::exchange(other.census_name, nullptr);
        return *this;
    }

    void ReserveRange(u64 start, std::size_t size);

    /// FRITCHII_MEM_CENSUS: names the table (a string literal); ReserveRange then logs calls that
    /// commit 1 MiB or more of it.
    void SetCensusName(const char* name);

    /// Whether the entry at `index` lies in a committed part of the table.
    [[nodiscard]] bool IsCommittedAt(std::size_t index) const noexcept {
        if (page_bits == 0 || first_level_map.empty()) {
            return false;
        }
        const std::size_t level = (index << page_bits) >> first_level_shift;
        return level < first_level_map.size() && first_level_map[level] != nullptr;
    }

    /// Bytes reserved for the entries.
    [[nodiscard]] std::size_t ReservedBytes() const noexcept {
        return alloc_size;
    }

    /// FRITCHII_RAM_PT_TRIM: the whole 4 KiB pages of entries in [first, end) leave RAM (they stay
    /// committed; their contents are undefined until written again). Only for entries nobody
    /// reads until they are written again (unmapped), under the lock their writers take.
    void DiscardEntries(std::size_t first, std::size_t end) noexcept;

    [[nodiscard]] const BaseAddr& operator[](std::size_t index) const {
        return base_ptr[index];
    }

    [[nodiscard]] BaseAddr& operator[](std::size_t index) {
        return base_ptr[index];
    }

    [[nodiscard]] BaseAddr* data() {
        return base_ptr;
    }

    [[nodiscard]] const BaseAddr* data() const {
        return base_ptr;
    }

private:
    void AllocateLevel(u64 level);

    std::size_t address_space_bits{};
    std::size_t first_level_bits{};
    std::size_t page_bits{};
    std::size_t first_level_shift{};
    std::size_t first_level_chunk_size{};
    std::size_t alloc_size{};
    std::vector<void*> first_level_map{};
    BaseAddr* base_ptr{};
    const char* census_name{};
};

} // namespace Common
