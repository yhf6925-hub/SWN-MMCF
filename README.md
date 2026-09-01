# SWN-MMCF

Compact public training core for the SWN-MMCF kernel-weight network used with [OpenVINS](https://github.com/rpng/open_vins). The repository is intentionally flat: the complete public code consists of two Python files.

## Files

- `model.py` defines the normalized encoder, training-only physics decoder, and inference-only ONNX graph.
- `train.py` provides the NPZ dataset loader, training loop, checkpoint writer, and ONNX export.


## Confidential objective boundary

The exact loss terms, their weights, and their epoch-dependent coordination are research parameters and are not disclosed. The public trainer calls one external function:

```python
def compose_loss(outputs, batch, epoch):
    # Private implementation; return one scalar torch.Tensor.
    ...
```

Pass it at runtime as `--objective package.module:compose_loss`. The private module and JSON configuration should be stored outside this repository; matching `private_*.py` and `private_*.json` files are ignored by Git.

## Data contract

Training data is a private NumPy `.npz` file. It must contain aligned rank-2 arrays named `feature` and `physics_target`. Additional numeric arrays are forwarded unchanged to the private objective function. This keeps the public training loop useful without exposing the loss construction or dataset-specific supervision.

For the current OpenVINS adapter, the deployment graph uses input name `x`, output name `alpha`, and a dynamic batch dimension. The deployed Stage-6 contract is a 40,661-element padded feature vector and 52 normalized kernel weights. Normalization is embedded in the exported ONNX graph, so the OpenVINS runtime passes raw packed features directly.

## Training

Install a recent Python, NumPy, PyTorch, and ONNX environment, then run:

```bash
python train.py \
  --data /secure/path/stage6_train.npz \
  --config /secure/path/private_training.json \
  --objective private_losses:compose_loss \
  --checkpoint /secure/path/stage6_encoder.pth \
  --onnx /secure/path/swn_stage6.onnx
```

The private JSON has `network` and `training` objects. The code validates all required keys at startup; no publishable defaults are included because default values would disclose the training setup. The trainer logs only the aggregate objective and does not print individual loss weights.

## OpenVINS relationship and acknowledgment

SWN-MMCF is an extension around OpenVINS, not a replacement or a fork published here. OpenVINS supplies sensor ingestion, inertial propagation, camera-state cloning, feature tracking, MSCKF residual/Jacobian construction, and the estimator state. This network predicts the normalized MMCF kernel-mixture coefficients consumed at the robust update boundary. No OpenVINS source code is copied into this repository.

Please obtain OpenVINS from its official repository and retain its GPL-3.0 notices:

```bash
git clone --branch v2.7 https://github.com/rpng/open_vins.git
```


Also cite the SWN-MMCF paper after its final bibliographic information is available.

## Demo video

The local screen recording is about 125 MB, above GitHub's normal 100 MB per-file limit, so it is deliberately not committed to the source history. Publish a compressed copy as a GitHub Release asset and link it here when ready.

## License

This repository is released under GPL-3.0-only. OpenVINS is a separate upstream project and retains its original copyrights and license notices. See `LICENSE`.
