// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "swn_mmcf/types.hpp"

#include <vector>

namespace swn_mmcf {

// Protected core operator A: SWN window construction and node weighting.
// Only the public contract is disclosed; no implementation is distributed.
class SwnWindowSelector {
 public:
  explicit SwnWindowSelector(SwnMmcfConfig config);

  [[nodiscard]] SelectedWindow select(
      const OpenVinsSnapshot& state,
      const std::vector<ImuSample>& imu_segment) const;

 private:
  [[nodiscard]] double scoreNode(
      const PoseClone& clone,
      const OpenVinsSnapshot& state,
      const std::vector<ImuSample>& imu_segment) const;

  SwnMmcfConfig config_;
};

// Protected core operator B: MMCF residual/Jacobian/covariance construction.
// Only signatures, types, and behavioral contracts are disclosed.
class MmcfConstraintBuilder {
 public:
  explicit MmcfConstraintBuilder(SwnMmcfConfig config);

  [[nodiscard]] std::vector<FusionConstraint> build(
      const OpenVinsSnapshot& state,
      const SelectedWindow& window,
      const std::vector<ImuSample>& imu_segment) const;

 private:
  [[nodiscard]] FusionConstraint buildVisualConstraint(
      const OpenVinsSnapshot& state,
      const SelectedWindow& window) const;

  [[nodiscard]] FusionConstraint buildInertialConstraint(
      const OpenVinsSnapshot& state,
      const SelectedWindow& window,
      const std::vector<ImuSample>& imu_segment) const;

  [[nodiscard]] FusionConstraint buildCrossModalConstraint(
      const FusionConstraint& visual,
      const FusionConstraint& inertial) const;

  SwnMmcfConfig config_;
};

// Protected core operator C: robust gating, weighting, and update synthesis.
// The returned packet is consumed by the OpenVINS adapter.
class RobustUpdateSynthesizer {
 public:
  explicit RobustUpdateSynthesizer(SwnMmcfConfig config);

  [[nodiscard]] UpdatePacket synthesize(
      Timestamp timestamp_sec,
      const std::vector<FusionConstraint>& candidates,
      Diagnostics* diagnostics) const;

 private:
  [[nodiscard]] bool passesInnovationGate(
      const FusionConstraint& constraint,
      double* normalized_innovation) const;

  [[nodiscard]] double computeRobustWeight(
      const FusionConstraint& constraint,
      double normalized_innovation) const;

  SwnMmcfConfig config_;
};

}  // namespace swn_mmcf

