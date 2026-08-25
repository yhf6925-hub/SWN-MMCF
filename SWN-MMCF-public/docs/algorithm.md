# Algorithm logic

This document describes the observable behavior of SWN-MMCF without disclosing the protected numerical kernels. Symbols should be synchronized with the final manuscript before publication.

## Inputs and outputs

At camera time \(t_k\), the method reads an immutable snapshot of the current OpenVINS estimate, the active cloned poses, tracked feature observations, a covariance summary, and the IMU samples spanning the latest camera interval. It returns either no update or an `UpdatePacket` containing accepted residual blocks, Jacobians, covariances, and confidence values.

SWN-MMCF never owns or independently propagates the OpenVINS state. The OpenVINS adapter is the only component allowed to map a returned packet into the estimator's internal error-state ordering.

## Stage table

| Stage | Inputs | Processing logic | Outputs | Failure behavior |
|---|---|---|---|---|
| 0. Baseline propagation | IMU stream | OpenVINS propagates the nominal state and covariance | Propagated state | Defer to OpenVINS |
| 1. Snapshot | Active state, clones, tracks | Copy only the fields used by SWN-MMCF; preserve timestamps and ordering | `OpenVinsSnapshot` | Reject non-finite or non-monotonic data |
| 2. SWN selection | Snapshot, interval IMU | Score clone nodes from visual quality, inertial excitation, uncertainty, and temporal coverage; select a bounded window | `SelectedWindow` | Skip update if fewer than the minimum nodes survive |
| 3. Visual construction | Selected window, tracks | Form multi-view residuals; remove invalid tracks and nuisance feature variables | Visual candidates | Drop degenerate tracks |
| 4. Inertial construction | Selected window, IMU | Form consistency terms over the same time support and align dimensions with the estimator error state | Inertial candidates | Drop intervals with gaps or invalid timing |
| 5. MMCF coupling | Visual and inertial candidates | Build cross-modal constraints and confidence using the manuscript-defined fusion rule | Candidate set | Fall back to enabled unimodal terms |
| 6. Robust synthesis | Candidate set, covariance | Perform finite-value checks, innovation gating, robust weighting, and update-size limiting | `UpdatePacket` | Return `request_update = false` if none survive |
| 7. Estimator update | Vetted packet | Adapter maps columns to OpenVINS state variables and invokes the update boundary | Corrected state/covariance | Preserve the baseline state on rejection |
| 8. Cleanup | Current timestamp | Trim consumed IMU samples; let OpenVINS perform its normal clone/feature marginalization | Bounded buffers | Never marginalize independently |

## Protected function contracts

### `SwnWindowSelector::select`

- Preconditions: clone timestamps are strictly ordered; IMU timestamps are monotonic; all numeric fields are finite.
- Postconditions: returned nodes are a subset of active clones, time ordered, and no larger than `max_window_clones`.
- Invariant: selection cannot mutate the OpenVINS state.

### `MmcfConstraintBuilder::build`

- Preconditions: the window satisfies the configured minimum size.
- Postconditions: every returned block has mutually consistent residual, Jacobian, and covariance dimensions.
- Invariant: every Jacobian column has a documented mapping to the active estimator error state.

### `RobustUpdateSynthesizer::synthesize`

- Preconditions: candidates use the same state snapshot and timestamp.
- Postconditions: accepted constraints pass finite-value, dimensional, confidence, and innovation checks.
- Invariant: an empty accepted set produces `request_update = false`.

## Complexity envelope

Let \(N\) be the number of active pose clones, \(M\) the number of candidate feature tracks, and \(C\) the number of candidate constraint blocks. Window scoring is bounded by the configured clone window. Constraint assembly scales with retained observations, while robust gating is linear in \(C\) excluding the small dense linear-algebra operations within each block. The public artifact intentionally makes no unsupported claim about exact asymptotic complexity of the protected kernels.

