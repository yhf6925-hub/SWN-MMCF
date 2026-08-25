// SPDX-License-Identifier: GPL-3.0-only
#include "swn_mmcf/core_api.hpp"
#include "swn_mmcf/openvins_bridge.hpp"
#include "swn_mmcf/version.hpp"

#include <iostream>
#include <type_traits>

static_assert(std::is_standard_layout_v<swn_mmcf::ImuSample>);
static_assert(SWN_MMCF_PUBLIC_API_VERSION_MAJOR == 0);

int main() {
  std::cout << "SWN-MMCF public API: OK\n";
  return 0;
}

