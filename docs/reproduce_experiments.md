# Reproducing experiments

Use [inference](getting_started.md) if you only want to process your own images.
Reproduction requires the original CCPD images and substantially more time and storage.
The included checkpoints and results are from the original team experiment, with seed 9444.

## Dataset

Obtain CCPD2019 from the [original project](https://github.com/detectRecog/CCPD),
following its distribution terms. Put it in this layout:

```text
chinese-license-plate-recognition/
  config.py
  CCPD2019/
    splits/
    ccpd_base/
    ccpd_blur/
    ccpd_challenge/
    ...
```

Or set `CCPD_DATA_ROOT` before starting Python/Jupyter:

```powershell
# Windows PowerShell
$env:CCPD_DATA_ROOT = 'D:\datasets\CCPD2019'
```

```bash
# macOS / Linux
export CCPD_DATA_ROOT=/path/to/CCPD2019
```

`data_preparation/` preserves the exact 30k/5k/10k selected split and metadata.
The 3k development set is drawn from training, not an independent test set.
Raw images are needed even when metadata is bundled: preparation reads and links/copies them.

## Detection

From the repository root:

```bash
python detection/prepare_yolo_dataset.py --mode auto
python detection/evaluate_detector.py --weights detection/runs/ccpd2019_yolov8n_gpu_30ep/weights/best.pt --subsets all --with-map --device cpu
```

`auto` tries hardlinks, symlinks, then copies. To train a new run without replacing
the included final checkpoint, use a new name:

```bash
python detection/train_detector.py --data detection/yolo_dataset/ccpd_full.yaml --model detection/yolov8n.pt --epochs 30 --imgsz 640 --batch 16 --device 0 --name ccpd2019_reproduction
```

Use the new run's `weights/best.pt` for subsequent evaluation and `predict.py --detector`.
Detector scripts use `--device 0` for the first CUDA GPU and `--device cpu` for CPU.
The top-level pipeline CLI uses PyTorch-style `--device cuda:0`.

## Recognition and combined evaluation

Install the notebook kernel in the same environment:

```bash
python -m ipykernel install --user --name plate-project --display-name "Plate Project"
```

Open `notebooks/ccpd_workflow.ipynb` with that kernel and run cells in order.
The notebook locates the repository from either the root or `notebooks/` directory.
Parts 3 and 4 contain recognition training, recognition-only evaluation and combined evaluation.
Existing trained checkpoints are reused when the notebook's reuse checks pass.
To perform fresh training, work in a separate clone and change the checkpoint/run
destinations or reuse conditions explicitly. Evaluation regenerates result CSVs;
keep the tracked original records if you need to compare them.

## CCPD2020-Green scope

Detector preparation also accepts `CCPD2020/ccpd_green/` with `train/val/test`
subdirectories, or that directory via `CCPD_DATA_ROOT`. It creates separate Green YOLO
folders and split files. The full original notebook, seven-head recogniser and
recorded metrics are CCPD2019-specific; Green detection support does not imply
eight-character recognition or reproduction of the recorded results on Green data.

## Validate saved records without a dataset

```bash
python validate_results.py
```

This standard-library command checks split disjointness, metadata and recorded
prediction/summary consistency. It does not rerun model evaluation or speed measurements.
Read [evaluation definitions](evaluation.md) before comparing accuracy or FPS.
