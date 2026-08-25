// REVIEWER PSEUDOCODE — intentionally not compiled.
// Capitalized operations map to declarations in include/swn_mmcf/core_api.hpp.

procedure HANDLE_IMU(sample):
    OpenVINS.feed_measurement_imu(sample)
    SWN_MMCF.imu_buffer.append(sample)

procedure HANDLE_CAMERA(frame):
    # Baseline OpenVINS responsibilities remain unchanged.
    OpenVINS.feed_measurement_camera(frame)
    if OpenVINS.is_not_initialized():
        return

    snapshot <- CAPTURE_READ_ONLY_OPENVINS_STATE()
    imu_segment <- TAKE_IMU_BETWEEN_LAST_AND_CURRENT_CAMERA_TIME()

    # Protected operator A. The function body is not disclosed.
    window <- SWN_SELECT(snapshot, imu_segment)
    if size(window.nodes) < config.min_window_clones:
        REPORT("insufficient_window")
        return

    # Protected operator B. The function body is not disclosed.
    candidates <- MMCF_BUILD_CONSTRAINTS(snapshot, window, imu_segment)

    # Protected operator C. The function body is not disclosed.
    packet <- ROBUST_SYNTHESIZE_UPDATE(
        timestamp = snapshot.timestamp,
        candidates = candidates)

    if packet.request_update:
        # Adapter converts the public packet to the OpenVINS state ordering.
        OPENVINS_APPLY_UPDATE(packet)

    RETAIN_ONLY_IMU_NEWER_THAN(snapshot.timestamp)
    REPORT_DIAGNOSTICS(window, candidates, packet)

function SWN_SELECT(snapshot, imu_segment) -> SelectedWindow:
    REQUIRE snapshot.clones are time ordered
    REQUIRE covariance summary is finite
    # 1. Form candidate nodes from active OpenVINS clones.
    # 2. Compute visual, inertial, and temporal quality descriptors.
    # 3. Evaluate the manuscript-defined SWN score for each candidate.
    # 4. Enforce temporal coverage and maximum-window constraints.
    # 5. Normalize selected node weights and return the ordered window.
    BODY OMITTED

function MMCF_BUILD_CONSTRAINTS(snapshot, window, imu_segment)
        -> list<FusionConstraint>:
    REQUIRE window contains at least min_window_clones nodes
    # 1. Assemble visual residual blocks from valid multi-clone tracks.
    # 2. Assemble inertial consistency blocks over matching intervals.
    # 3. Project nuisance variables out of the visual block.
    # 4. Align residuals/Jacobians with the active OpenVINS error state.
    # 5. Construct cross-modal constraints and calibrated confidence.
    BODY OMITTED

function ROBUST_SYNTHESIZE_UPDATE(timestamp, candidates) -> UpdatePacket:
    # 1. Reject non-finite and dimensionally inconsistent candidates.
    # 2. Evaluate normalized innovation and chi-square gate.
    # 3. Compute manuscript-defined robust/confidence weights.
    # 4. Bound the update size and preserve temporal ordering.
    # 5. Return residual, Jacobian, covariance, and diagnostics.
    BODY OMITTED

