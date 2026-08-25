// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace swn_mmcf {

using Timestamp = double;
using Vector3 = std::array<double, 3>;
using Quaternion = std::array<double, 4>;

struct ImuSample {
  Timestamp timestamp_sec{};
  Vector3 angular_velocity_rad_s{};
  Vector3 linear_acceleration_m_s2{};
};

struct FeatureObservation {
  std::uint64_t track_id{};
  std::size_t camera_id{};
  Timestamp timestamp_sec{};
  std::array<double, 2> normalized_uv{};
  double tracking_quality{};
};

struct PoseClone {
  Timestamp timestamp_sec{};
  Quaternion q_global_to_imu{};
  Vector3 p_imu_in_global_m{};
};

// A deliberately minimal, OpenVINS-independent view of the active estimator state.
struct OpenVinsSnapshot {
  Timestamp timestamp_sec{};
  Quaternion q_global_to_imu{};
  Vector3 p_imu_in_global_m{};
  Vector3 v_imu_in_global_m_s{};
  Vector3 gyro_bias{};
  Vector3 accel_bias{};
  std::vector<PoseClone> clones{};
  std::vector<FeatureObservation> feature_observations{};
  std::vector<double> covariance_diagonal{};
};

struct WindowNode {
  Timestamp timestamp_sec{};
  std::size_t clone_index{};
  double visual_quality{};
  double inertial_quality{};
  double temporal_weight{};
};

struct SelectedWindow {
  std::vector<WindowNode> nodes{};
  Timestamp start_time_sec{};
  Timestamp end_time_sec{};
};

enum class ConstraintKind : std::uint8_t {
  kVisual = 0,
  kInertial = 1,
  kCrossModal = 2,
};

struct FusionConstraint {
  ConstraintKind kind{ConstraintKind::kVisual};
  std::vector<double> residual{};
  std::vector<double> jacobian_row_major{};
  std::size_t jacobian_rows{};
  std::size_t jacobian_cols{};
  std::vector<double> covariance_row_major{};
  double confidence{};
};

struct UpdatePacket {
  Timestamp timestamp_sec{};
  std::vector<FusionConstraint> accepted_constraints{};
  double aggregate_confidence{};
  bool request_update{};
};

struct Diagnostics {
  std::size_t input_clones{};
  std::size_t selected_nodes{};
  std::size_t candidate_constraints{};
  std::size_t accepted_constraints{};
  double normalized_innovation{};
  std::string status{};
};

struct SwnMmcfConfig {
  std::size_t max_window_clones{10};
  std::size_t min_window_clones{3};
  std::size_t max_constraints_per_update{40};
  double min_track_quality{0.25};
  double min_constraint_confidence{0.50};
  double innovation_gate_probability{0.95};
  double numerical_epsilon{1e-9};
  bool enable_visual_term{true};
  bool enable_inertial_term{true};
  bool enable_cross_modal_term{true};
};

}  // namespace swn_mmcf

