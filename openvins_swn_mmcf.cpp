// SPDX-License-Identifier: GPL-3.0-or-later
// Core SWN-MMCF inference path inserted into the OpenVINS MSCKF update.

#include "state/State.h"
#include "state/StateHelper.h"
#include "types/IMU.h"
#include "types/PoseJPL.h"

#include <Eigen/Dense>
#include <onnxruntime_c_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ov_msckf {
namespace swn_mmcf {

using ov_type::Type;

static const OrtApi *ort_api() {
  const OrtApi *api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  if (api == nullptr)
    throw std::runtime_error("ONNX Runtime API is unavailable");
  return api;
}

static void check_ort(OrtStatus *status) {
  if (status == nullptr)
    return;
  const OrtApi *api = ort_api();
  const std::string message = api->GetErrorMessage(status);
  api->ReleaseStatus(status);
  throw std::runtime_error("ONNX Runtime: " + message);
}

struct PackingSpec {
  int measurement_capacity;
  int state_capacity;
  double measurement_scale;
  Eigen::VectorXd kernel_scales;
};

struct MmcfSpec {
  double q;
  double minimum_factor;
  double relative_tolerance;
  int iteration_limit;
};

struct CanonicalSystem {
  Eigen::MatrixXd H;
  Eigen::VectorXd residual;
  Eigen::MatrixXd R;
  Eigen::MatrixXd prior_covariance;
  Eigen::VectorXd prior_error;
};

struct RobustFactors {
  double prior;
  double measurement;
};

class OnnxWeightNetwork {
public:
  explicit OnnxWeightNetwork(const std::string &model_path) {
    if (model_path.empty())
      throw std::invalid_argument("ONNX model path is empty");
    const OrtApi *api = ort_api();
    OrtEnv *env = nullptr;
    OrtSessionOptions *options = nullptr;
    OrtSession *session = nullptr;
    OrtMemoryInfo *memory = nullptr;
    try {
      check_ort(api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "openvins_swn_mmcf", &env));
      check_ort(api->CreateSessionOptions(&options));
      check_ort(api->SetSessionGraphOptimizationLevel(options, ORT_ENABLE_ALL));
      check_ort(api->CreateSession(env, model_path.c_str(), options, &session));
      check_ort(api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory));

      size_t input_count = 0;
      size_t output_count = 0;
      check_ort(api->SessionGetInputCount(session, &input_count));
      check_ort(api->SessionGetOutputCount(session, &output_count));
      if (input_count != 1 || output_count != 1)
        throw std::runtime_error("SWN ONNX graph must have one input and one output");

      OrtAllocator *allocator = nullptr;
      char *name = nullptr;
      check_ort(api->GetAllocatorWithDefaultOptions(&allocator));
      check_ort(api->SessionGetInputName(session, 0, allocator, &name));
      input_name_ = name;
      check_ort(api->AllocatorFree(allocator, name));
      name = nullptr;
      check_ort(api->SessionGetOutputName(session, 0, allocator, &name));
      output_name_ = name;
      check_ort(api->AllocatorFree(allocator, name));

      OrtTypeInfo *type_info = nullptr;
      check_ort(api->SessionGetInputTypeInfo(session, 0, &type_info));
      const OrtTensorTypeAndShapeInfo *tensor_info = nullptr;
      check_ort(api->CastTypeInfoToTensorInfo(type_info, &tensor_info));
      size_t dimension_count = 0;
      check_ort(api->GetDimensionsCount(tensor_info, &dimension_count));
      std::vector<int64_t> dimensions(dimension_count);
      check_ort(api->GetDimensions(tensor_info, dimensions.data(), dimensions.size()));
      api->ReleaseTypeInfo(type_info);
      if (dimensions.size() != 2 || dimensions[1] <= 0)
        throw std::runtime_error("unexpected SWN ONNX input shape");
      input_size_ = static_cast<std::size_t>(dimensions[1]);
    } catch (...) {
      if (memory != nullptr)
        api->ReleaseMemoryInfo(memory);
      if (session != nullptr)
        api->ReleaseSession(session);
      if (options != nullptr)
        api->ReleaseSessionOptions(options);
      if (env != nullptr)
        api->ReleaseEnv(env);
      throw;
    }
    env_ = env;
    options_ = options;
    session_ = session;
    memory_ = memory;
  }

  ~OnnxWeightNetwork() {
    const OrtApi *api = ort_api();
    if (memory_ != nullptr)
      api->ReleaseMemoryInfo(static_cast<OrtMemoryInfo *>(memory_));
    if (session_ != nullptr)
      api->ReleaseSession(static_cast<OrtSession *>(session_));
    if (options_ != nullptr)
      api->ReleaseSessionOptions(static_cast<OrtSessionOptions *>(options_));
    if (env_ != nullptr)
      api->ReleaseEnv(static_cast<OrtEnv *>(env_));
  }

  OnnxWeightNetwork(const OnnxWeightNetwork &) = delete;
  OnnxWeightNetwork &operator=(const OnnxWeightNetwork &) = delete;

  std::size_t input_size() const { return input_size_; }

  Eigen::VectorXd infer(std::vector<float> feature) const {
    if (feature.size() != input_size_)
      throw std::runtime_error("packed feature does not match the ONNX input");

    const OrtApi *api = ort_api();
    const int64_t shape[] = {1, static_cast<int64_t>(feature.size())};
    const char *input_names[] = {input_name_.c_str()};
    const char *output_names[] = {output_name_.c_str()};
    OrtValue *input = nullptr;
    OrtValue *output = nullptr;
    OrtTensorTypeAndShapeInfo *output_info = nullptr;
    try {
      check_ort(api->CreateTensorWithDataAsOrtValue(
          static_cast<OrtMemoryInfo *>(memory_), feature.data(), feature.size() * sizeof(float), shape, 2,
          ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input));
      check_ort(api->Run(static_cast<OrtSession *>(session_), nullptr, input_names, &input, 1, output_names, 1, &output));
      check_ort(api->GetTensorTypeAndShape(output, &output_info));
      size_t count = 0;
      check_ort(api->GetTensorShapeElementCount(output_info, &count));
      float *raw = nullptr;
      check_ort(api->GetTensorMutableData(output, reinterpret_cast<void **>(&raw)));
      Eigen::VectorXd alpha(static_cast<Eigen::Index>(count));
      for (std::size_t i = 0; i < count; ++i)
        alpha(static_cast<Eigen::Index>(i)) = static_cast<double>(raw[i]);

      const double sum = alpha.sum();
      if (!alpha.allFinite() || sum <= 0.0)
        throw std::runtime_error("invalid SWN ONNX output");
      alpha /= sum;
      api->ReleaseTensorTypeAndShapeInfo(output_info);
      api->ReleaseValue(output);
      api->ReleaseValue(input);
      return alpha;
    } catch (...) {
      if (output_info != nullptr)
        api->ReleaseTensorTypeAndShapeInfo(output_info);
      if (output != nullptr)
        api->ReleaseValue(output);
      if (input != nullptr)
        api->ReleaseValue(input);
      throw;
    }
  }

private:
  void *env_ = nullptr;
  void *options_ = nullptr;
  void *session_ = nullptr;
  void *memory_ = nullptr;
  std::string input_name_;
  std::string output_name_;
  std::size_t input_size_ = 0;
};

static CanonicalSystem map_openvins_system(
    const std::shared_ptr<State> &state,
    const std::vector<std::shared_ptr<Type>> &H_order,
    const Eigen::MatrixXd &H,
    const Eigen::VectorXd &residual,
    const Eigen::MatrixXd &R,
    const PackingSpec &packing) {
  if (H.rows() != residual.rows() || R.rows() != residual.rows() || R.cols() != residual.rows())
    throw std::invalid_argument("inconsistent OpenVINS measurement system");
  if (packing.measurement_capacity <= 0 || packing.state_capacity <= 0 || packing.measurement_scale <= 0.0 ||
      packing.kernel_scales.size() == 0 || (packing.kernel_scales.array() <= 0.0).any())
    throw std::invalid_argument("invalid SWN packing specification");
  // Paper order: attitude, gyro bias, velocity, accelerometer bias, position,
  // camera-0 extrinsics and the active pose clones.
  constexpr int kImuSize = 15;
  constexpr int kPoseSize = 6;
  const int base_size = kImuSize + kPoseSize;
  const int clone_limit = std::max(0, (packing.state_capacity - base_size) / kPoseSize);
  const int clone_count = std::min<int>(state->_clones_IMU.size(), clone_limit);
  const int state_size = base_size + kPoseSize * clone_count;

  Eigen::MatrixXd H_canonical = Eigen::MatrixXd::Zero(H.rows(), state_size);
  std::vector<std::shared_ptr<Type>> covariance_order{state->_imu};
  std::vector<int> source_ids(state_size, -1);
  for (int i = 0; i < 3; ++i) {
    source_ids[i] = i;
    source_ids[3 + i] = 9 + i;
    source_ids[6 + i] = 6 + i;
    source_ids[9 + i] = 12 + i;
    source_ids[12 + i] = 3 + i;
  }

  int covariance_offset = state->_imu->size();
  if (state->_calib_IMUtoCAM.count(0) > 0) {
    const auto &camera = state->_calib_IMUtoCAM.at(0);
    if (camera->id() >= 0) {
      covariance_order.push_back(camera);
      for (int i = 0; i < kPoseSize; ++i)
        source_ids[15 + i] = covariance_offset + i;
      covariance_offset += camera->size();
    }
  }

  int clone_index = 0;
  for (const auto &clone : state->_clones_IMU) {
    if (clone_index == clone_count)
      break;
    covariance_order.push_back(clone.second);
    for (int i = 0; i < kPoseSize; ++i)
      source_ids[base_size + kPoseSize * clone_index + i] = covariance_offset + i;
    covariance_offset += clone.second->size();
    ++clone_index;
  }

  int source_column = 0;
  for (const auto &variable : H_order) {
    if (variable == state->_imu) {
      if (variable->size() != kImuSize)
        throw std::runtime_error("unexpected OpenVINS IMU error-state size");
      H_canonical.block(0, 0, H.rows(), 3) = H.block(0, source_column, H.rows(), 3);
      H_canonical.block(0, 3, H.rows(), 3) = H.block(0, source_column + 9, H.rows(), 3);
      H_canonical.block(0, 6, H.rows(), 3) = H.block(0, source_column + 6, H.rows(), 3);
      H_canonical.block(0, 9, H.rows(), 3) = H.block(0, source_column + 12, H.rows(), 3);
      H_canonical.block(0, 12, H.rows(), 3) = H.block(0, source_column + 3, H.rows(), 3);
    } else if (state->_calib_IMUtoCAM.count(0) > 0 && variable == state->_calib_IMUtoCAM.at(0)) {
      if (variable->size() != kPoseSize)
        throw std::runtime_error("unexpected camera extrinsic error-state size");
      H_canonical.block(0, 15, H.rows(), kPoseSize) = H.block(0, source_column, H.rows(), kPoseSize);
    } else {
      int index = 0;
      for (const auto &clone : state->_clones_IMU) {
        if (index == clone_count)
          break;
        if (variable == clone.second) {
          if (variable->size() != kPoseSize)
            throw std::runtime_error("unexpected clone error-state size");
          H_canonical.block(0, base_size + kPoseSize * index, H.rows(), kPoseSize) =
              H.block(0, source_column, H.rows(), kPoseSize);
          break;
        }
        ++index;
      }
    }
    source_column += variable->size();
  }
  if (source_column != H.cols())
    throw std::runtime_error("OpenVINS Jacobian ordering mismatch");

  const Eigen::MatrixXd P_source = StateHelper::get_marginal_covariance(state, covariance_order);
  Eigen::MatrixXd P = Eigen::MatrixXd::Zero(state_size, state_size);
  for (int row = 0; row < state_size; ++row)
    for (int col = 0; col < state_size; ++col)
      if (source_ids[row] >= 0 && source_ids[col] >= 0)
        P(row, col) = P_source(source_ids[row], source_ids[col]);

  const int measurement_size = std::min<int>(residual.rows(), packing.measurement_capacity);
  const double scale = packing.measurement_scale;
  CanonicalSystem system;
  system.H = H_canonical.topRows(measurement_size) * scale;
  system.residual = residual.head(measurement_size) * scale;
  system.R = R.topLeftCorner(measurement_size, measurement_size) * (scale * scale);
  system.prior_covariance = std::move(P);
  system.prior_error = Eigen::VectorXd::Zero(state_size);
  return system;
}

static std::vector<float> pack_network_input(const CanonicalSystem &system, const PackingSpec &spec) {
  const int m = static_cast<int>(system.residual.rows());
  const int n = static_cast<int>(system.prior_error.rows());
  if (m > spec.measurement_capacity || n > spec.state_capacity)
    throw std::runtime_error("SWN system exceeds its padded input capacity");

  std::vector<float> feature;
  const auto append = [&feature](double value) { feature.push_back(static_cast<float>(value)); };
  for (Eigen::Index i = 0; i < spec.kernel_scales.rows(); ++i)
    append(spec.kernel_scales(i));
  for (int i = 0; i < spec.state_capacity; ++i)
    append(i < n ? system.prior_error(i) : 0.0);
  for (int i = 0; i < spec.measurement_capacity; ++i)
    append(i < m ? system.residual(i) : 0.0);
  for (int row = 0; row < spec.measurement_capacity; ++row)
    for (int col = 0; col < spec.state_capacity; ++col)
      append(row < m && col < n ? system.H(row, col) : 0.0);
  for (int i = 0; i < spec.measurement_capacity; ++i)
    append(i < m ? system.R(i, i) : 0.0);
  for (int row = 0; row < spec.state_capacity; ++row)
    for (int col = 0; col < spec.state_capacity; ++col)
      append(row < n && col < n ? system.prior_covariance(row, col) : 0.0);
  for (int i = 0; i < spec.measurement_capacity; ++i)
    append(i < m ? 1.0 : 0.0);
  for (int i = 0; i < spec.state_capacity; ++i)
    append(i < n ? 1.0 : 0.0);
  for (int i = 0; i < spec.measurement_capacity; ++i)
    append(i < m ? system.residual(i) : 0.0);
  Eigen::LDLT<Eigen::MatrixXd> noise_solver(system.R);
  if (noise_solver.info() != Eigen::Success)
    throw std::runtime_error("measurement covariance factorization failed");
  append(system.residual.dot(noise_solver.solve(system.residual)));
  return feature;
}

static RobustFactors solve_mmcf_factors(
    const CanonicalSystem &system,
    const Eigen::VectorXd &alpha,
    const PackingSpec &packing,
    const MmcfSpec &spec) {
  if (alpha.rows() != packing.kernel_scales.rows())
    throw std::runtime_error("weight and kernel dimensions do not match");
  if (3.0 * spec.q <= 1.0 || spec.q >= 1.0 || spec.minimum_factor <= 0.0 ||
      spec.relative_tolerance <= 0.0 || spec.iteration_limit <= 0)
    throw std::invalid_argument("invalid MMCF specification");

  const double kernel_a = (spec.q - 1.0) / (3.0 * spec.q - 1.0);
  const double kernel_b = (2.0 - spec.q) / (spec.q - 1.0);
  Eigen::JacobiSVD<Eigen::MatrixXd> prior_solver(
      system.prior_covariance, Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::VectorXd correction = system.prior_error;
  RobustFactors factors{1.0, 1.0};

  for (int iteration = 0; iteration < spec.iteration_limit; ++iteration) {
    const Eigen::VectorXd prior_residual = system.prior_error - correction;
    const Eigen::VectorXd innovation = system.residual - system.H * correction;
    const double prior_distance = std::max(0.0, prior_residual.dot(prior_solver.solve(prior_residual)));
    const double measurement_distance = std::max(
        0.0, innovation.dot(system.R.ldlt().solve(innovation)));

    const Eigen::ArrayXd prior_kernel =
        (1.0 - kernel_a * prior_distance / packing.kernel_scales.array().square()).max(0.0).pow(kernel_b);
    const Eigen::ArrayXd measurement_kernel =
        (1.0 - kernel_a * measurement_distance / packing.kernel_scales.array().square()).max(0.0).pow(kernel_b);
    factors.prior = std::max((alpha.array() * prior_kernel).sum(), spec.minimum_factor);
    factors.measurement = std::max((alpha.array() * measurement_kernel).sum(), spec.minimum_factor);

    const Eigen::MatrixXd effective_prior = system.prior_covariance / factors.prior;
    Eigen::MatrixXd innovation_covariance =
        system.H * effective_prior * system.H.transpose() + system.R / factors.measurement;
    innovation_covariance = 0.5 * (innovation_covariance + innovation_covariance.transpose());
    Eigen::LDLT<Eigen::MatrixXd> innovation_solver(innovation_covariance);
    if (innovation_solver.info() != Eigen::Success)
      throw std::runtime_error("MMCF innovation covariance factorization failed");
    const Eigen::MatrixXd gain = effective_prior * system.H.transpose() *
        innovation_solver.solve(Eigen::MatrixXd::Identity(
            innovation_covariance.rows(), innovation_covariance.cols()));
    const Eigen::VectorXd next = gain * system.residual;
    const double relative_step = (next - correction).norm() /
        std::max(correction.norm(), std::numeric_limits<double>::epsilon());
    correction = next;
    if (relative_step <= spec.relative_tolerance)
      break;
  }
  return factors;
}

static void apply_full_openvins_update(
    const std::shared_ptr<State> &state,
    const std::vector<std::shared_ptr<Type>> &H_order,
    const Eigen::MatrixXd &H,
    const Eigen::VectorXd &residual,
    const Eigen::MatrixXd &R,
    const RobustFactors &factors) {
  // (P/prior)H'[H(P/prior)H' + R/measurement]^-1 is algebraically
  // identical to PH'[HPH' + (prior/measurement)R]^-1. The latter lets
  // OpenVINS perform its normal complete-state covariance and mean update.
  const Eigen::MatrixXd robust_R = (factors.prior / factors.measurement) * R;
  StateHelper::EKFUpdate(state, H_order, H, residual, robust_R);
}

void run_swn_mmcf_update(
    const std::shared_ptr<State> &state,
    const std::vector<std::shared_ptr<Type>> &H_order,
    const Eigen::MatrixXd &H,
    const Eigen::VectorXd &residual,
    const Eigen::MatrixXd &R,
    const PackingSpec &packing,
    const MmcfSpec &mmcf,
    const OnnxWeightNetwork &network) {
  // UpdaterMSCKF reaches this point after feature triangulation, null-space
  // projection, chi-square rejection and measurement compression.
  const CanonicalSystem canonical =
      map_openvins_system(state, H_order, H, residual, R, packing);
  const std::vector<float> feature = pack_network_input(canonical, packing);
  const Eigen::VectorXd alpha = network.infer(feature);
  const RobustFactors factors = solve_mmcf_factors(canonical, alpha, packing, mmcf);

  // Use the learned robust factors with the complete OpenVINS Jacobian so all
  // active navigation, clone and calibration states remain coupled.
  apply_full_openvins_update(state, H_order, H, residual, R, factors);
}

/*
OpenVINS camera-update execution order:

  trackFEATS->feed_new_camera(message);
  propagator->propagate_and_clone(state, message.timestamp);
  auto features = collect_lost_and_marginalized_tracks();
  updaterMSCKF->build_nullspace_projected_system(features, H_order, H, residual, R);
  run_swn_mmcf_update(state, H_order, H, residual, R, packing, mmcf, network);
  updaterSLAM->update(state, active_slam_features);
  updaterSLAM->delayed_init(state, new_slam_features);
  StateHelper::marginalize_old_clone(state);
  publish_state_and_diagnostics(state);
*/

} // namespace swn_mmcf
} // namespace ov_msckf
