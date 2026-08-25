# Contributing

Issues and documentation corrections are welcome. Do not submit OpenVINS source copied from upstream or any private dataset/calibration material.

Changes to a public contract must update all of the following:

- the corresponding declaration under `include/swn_mmcf/`;
- the stage description in `docs/algorithm.md`;
- the control flow in `pseudocode/swn_mmcf_pipeline.pseudo.cpp`;
- the API version in `include/swn_mmcf/version.hpp` when compatibility changes.

Run the header check before opening a pull request:

```bash
cmake -S . -B build
cmake --build build
./build/swn_mmcf_header_check
```

