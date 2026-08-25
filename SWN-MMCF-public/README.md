# SWN-MMCF: public algorithm interface for OpenVINS

This repository is the compact public artifact accompanying the SWN-MMCF manuscript. It documents the method's interfaces, data flow, configuration surface, and integration points with OpenVINS. The implementation bodies of the manuscript's protected core functions are intentionally omitted; their contracts and pseudocode are provided instead.

> **Naming note:** `SWN-MMCF` is kept as the manuscript-defined method name. Replace this sentence with the full expansion used in the accepted manuscript before release.

## What is included

| Item | Disclosure level | Location |
|---|---|---|
| Input/output data structures | Full declarations | `include/swn_mmcf/types.hpp` |
| SWN and MMCF core APIs | Signatures and contracts only | `include/swn_mmcf/core_api.hpp` |
| OpenVINS integration boundary | Full declarations | `include/swn_mmcf/openvins_bridge.hpp` |
| End-to-end processing flow | Pseudocode | `pseudocode/swn_mmcf_pipeline.pseudo.cpp` |
| Algorithm stages and equations-to-code map | Logic-level description | `docs/algorithm.md` |
| OpenVINS hook map | Integration guide | `docs/openvins_integration.md` |
| Parameters | Example values and meanings | `config/swn_mmcf.example.yaml` |

This artifact does **not** contain trained weights, dataset files, private calibration, or the bodies of the three protected operators: window scoring, multi-modal constraint construction, and robust update synthesis. See `docs/disclosure_scope.md` for the exact boundary.

## Repository layout

```text
SWN-MMCF-public/
├── include/swn_mmcf/          # C++17 public contracts
├── pseudocode/                # reviewer-readable pipeline
├── config/                    # non-dataset-specific parameters
├── docs/                      # method and integration notes
├── tests/                     # header syntax check only
├── third_party/               # attribution and dependency notes
├── CMakeLists.txt
├── CITATION.cff
└── LICENSE
```

## Relationship to OpenVINS

SWN-MMCF is designed as an extension layer around OpenVINS. OpenVINS remains responsible for sensor ingestion, inertial propagation, camera-state cloning, feature tracking/triangulation, and the baseline MSCKF state update. SWN-MMCF consumes a read-only snapshot of the active window and returns a vetted update packet to the estimator adapter.

No OpenVINS source file is copied into this repository. Use the upstream project separately:

```bash
git clone --branch v2.7 https://github.com/rpng/open_vins.git
```

The tested OpenVINS commit, ROS distribution, compiler, dataset split, and calibration file hashes must be filled in under `docs/reproducibility.md` before the public release.

## Validate the public headers

The artifact is an interface release, not an executable estimator. The following command verifies that the public C++ declarations are self-consistent:

```bash
cmake -S . -B build
cmake --build build
./build/swn_mmcf_header_check
```

A successful run prints `SWN-MMCF public API: OK`. It does not execute the protected algorithm.

## How to read the method

1. Start with `docs/algorithm.md` for the stage-by-stage logic.
2. Read `pseudocode/swn_mmcf_pipeline.pseudo.cpp` for control flow.
3. Inspect `include/swn_mmcf/core_api.hpp` for exact input/output contracts.
4. Use `docs/openvins_integration.md` to locate the OpenVINS-side hooks.

## Citation

Please cite both the SWN-MMCF manuscript and OpenVINS. Replace the placeholder author and publication fields in `CITATION.cff` before publishing this repository.

## License

This interface artifact is released under GPL-3.0-only to remain compatible with the OpenVINS integration context. OpenVINS is a separate upstream project and retains its original copyright notices. See `LICENSE` and `third_party/NOTICE.md`.

