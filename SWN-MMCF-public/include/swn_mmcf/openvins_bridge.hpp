// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "swn_mmcf/types.hpp"

#include <memory>
#include <vector>

namespace ov_msckf {
class State;
class VioManager;
}  // namespace ov_msckf

namespace swn_mmcf {

// Converts OpenVINS-owned objects into the stable, read-only public data model.
// Implementation is project/version-specific and intentionally not included.
class OpenVinsSnapshotAdapter {
 public:
  [[nodiscard]] OpenVinsSnapshot capture(
      const std::shared_ptr<ov_msckf::State>& state) const;
};

// Applies an already-vetted packet at the estimator update boundary.
// Ownership of the state and covariance remains with OpenVINS.
class OpenVinsUpdateAdapter {
 public:
  void apply(
      const UpdatePacket& packet,
      const std::shared_ptr<ov_msckf::State>& state) const;
};

// Non-owning orchestration surface used by a ROS1/ROS2 wrapper or dataset runner.
class SwnMmcfPipeline {
 public:
  explicit SwnMmcfPipeline(SwnMmcfConfig config);

  void pushImu(const ImuSample& sample);

  [[nodiscard]] Diagnostics processAfterOpenVinsCameraUpdate(
      const std::shared_ptr<ov_msckf::State>& state);

 private:
  SwnMmcfConfig config_;
  std::vector<ImuSample> pending_imu_;
};

}  // namespace swn_mmcf

