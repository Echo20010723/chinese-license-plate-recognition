# Included model checkpoints

| Checkpoint | Purpose |
| --- | --- |
| `detection/yolov8n.pt` | Ultralytics YOLOv8n starting weights for detector training |
| `detection/runs/ccpd2019_yolov8n_gpu_30ep/weights/best.pt` | Team detector trained on the selected CCPD2019 training split, 30-epoch run |
| `recognition/checkpoints/seven_head_ccpd2019.pt` | Team seven-head spatial residual recogniser, selected by validation performance |

Checkpoints are included as ordinary Git files; Git LFS is not required. The recogniser
filename was simplified without changing checkpoint bytes. Original detector training
arguments, history and figures remain beside its weights. Recognition history is under
`recognition/results/`.

The recogniser takes a 48 x 192 RGB plate crop, uses ImageNet normalisation and predicts
seven position-specific vocabularies from `ccpd_annotations.py`. Its architecture is in
`recognition/model.py`. The Python loader checks the vocabulary and input-size metadata
before loading weights. No eight-character recognition checkpoint is included.

## SHA-256 checksums

Use these to verify model downloads. In PowerShell use `Get-FileHash path/to/model.pt -Algorithm SHA256`;
on Linux use `sha256sum path/to/model.pt`.

| File (relative to repository root) | SHA-256 |
| --- | --- |
| `detection/yolov8n.pt` | `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` |
| `detection/runs/ccpd2019_yolov8n_gpu_30ep/weights/best.pt` | `2030793219f337a30a6934e1642101decc786d55cacff27ddd54b07f640fed01` |
| `recognition/checkpoints/seven_head_ccpd2019.pt` | `c13233fd79f8c10aa912b39881a042011158ae5c26aac403cd3de9e9c3ed3f8b` |

See [evaluation](evaluation.md) for recorded metrics and [third-party notices](../THIRD_PARTY_NOTICES.md)
for attribution and distribution scope. Loading custom checkpoints requires compatible
model definitions and a checkpoint source you trust.
