# OpenVINS integration map

The integration follows OpenVINS ownership rules: OpenVINS ingests sensors, propagates state, maintains cloned poses and feature tracks, and performs its filter update. SWN-MMCF observes a snapshot and proposes additional vetted constraints.

## Hook map

| SWN-MMCF event | OpenVINS-side concept | Direction | Notes |
|---|---|---|---|
| IMU arrival | `VioManager` IMU measurement path | Into OpenVINS and local interval buffer | Do not alter timestamps or units |
| Camera arrival | `VioManager` camera measurement path | Into OpenVINS | Run the baseline tracking/propagation path first |
| State capture | `State` and active IMU clones | OpenVINS → snapshot | Copy under the same synchronization policy used by the host application |
| Track capture | Active feature database / update candidates | OpenVINS → snapshot | Keep track IDs, camera IDs, normalized coordinates, and timestamps |
| SWN-MMCF update | MSCKF-compatible residual update boundary | Packet → OpenVINS | Map packet columns to the exact active error-state order |
| Marginalization | OpenVINS state helper / normal camera update flow | OpenVINS-owned | SWN-MMCF must not remove clones itself |

The exact C++ symbols can vary across OpenVINS versions. Pin and record the upstream commit, then implement `OpenVinsSnapshotAdapter` and `OpenVinsUpdateAdapter` against that revision. The public headers use forward declarations so this artifact does not vendor or hard-code upstream internals.

## Recommended insertion order

```text
IMU callback ──► OpenVINS propagation path ──► local interval buffer

Camera callback
  ├─► OpenVINS tracking / propagation / clone augmentation
  ├─► capture immutable snapshot
  ├─► SWN window selection
  ├─► MMCF constraint construction
  ├─► robust gating and packet synthesis
  ├─► adapter applies accepted update
  └─► OpenVINS normal cleanup and marginalization
```

## Integration assertions

An implementation should fail closed when any of the following is false:

- Camera and IMU timestamps use the same time convention, including the configured camera–IMU offset.
- Clone order in the snapshot matches the state order used to construct Jacobian columns.
- Residual length equals Jacobian row count.
- Covariance dimensions equal the residual dimension.
- All packet values are finite and covariance blocks are symmetric positive semidefinite within numerical tolerance.
- The active OpenVINS state has not changed between snapshot capture and packet application.

## Upstream references

- OpenVINS repository: <https://github.com/rpng/open_vins>
- OpenVINS documentation: <https://docs.openvins.com/>
- OpenVINS paper: P. Geneva et al., “OpenVINS: A Research Platform for Visual-Inertial Estimation,” ICRA 2020.

