# Disclosure scope

This file makes the public/non-public boundary explicit for reviewers and downstream users.

## Public in this repository

- Data structures and units used at the SWN-MMCF boundary.
- Class names, function signatures, preconditions, postconditions, and invariants.
- End-to-end control flow and failure behavior.
- OpenVINS integration points and state-ownership rules.
- Configuration keys needed to discuss ablations and deployment limits.
- A syntax-only build check for the interface.

## Intentionally omitted

- The numerical expression used by `SwnWindowSelector::scoreNode`.
- The implementation of visual, inertial, and cross-modal MMCF constraint construction.
- The robust weighting and confidence calibration formulae beyond those already printed in the manuscript.
- Learned weights, training scripts, private datasets, and device-specific calibration.
- The adapter implementation that depends on the authors' private OpenVINS fork, if any.

## Consequence

This is a logic-level research artifact. It is suitable for reviewing architecture, inputs/outputs, control flow, ablation boundaries, and integration assumptions. It is not a claim of full numerical reproducibility and must not be described as a complete runnable release.

Before publication, verify that the wording matches the journal or conference code-availability statement. If the venue requires complete reproducibility or the applicable GPL obligations require corresponding source for a distributed derivative executable, declarations alone may not satisfy that requirement.

