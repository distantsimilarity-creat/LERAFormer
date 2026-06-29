# LERAFormer

PyTorch implementation of **LERAFormer**, a boundary-aware dual-branch network for landslide semantic segmentation from multi-source remote sensing data.

LERAFormer couples a Swin Transformer semantic branch with a lightweight CNN edge-detail branch. The framework is designed to improve landslide region consistency and boundary delineation under blurred boundary transitions, background interference, and semantic-detail misalignment.

> This repository is intended to accompany the manuscript:  
> **Boundary-aware dual-branch network with explicit region correction for landslide segmentation from multi-source remote sensing data**

## Main features

- **Dual-branch architecture**: Transformer semantic branch + CNN edge-detail branch.
- **LERA module**: Local Enhancement and Rectified Alignment for enhancing local structures and correcting inconsistent cross-branch responses.
- **LCAF module**: Local Cross-Attention Fusion for boundary-guided local semantic interaction.
- **Ablation models**: variants without LERA, LCAF, DGER, CECA, or LSK are included.
- **Baseline models**: U-Net, DeepLabV3+, SegFormer, STUNet, TransUNet, SwinUNet, LEFormer, and LandslideNet are included for comparison.
- **Validation threshold selection**: the best foreground threshold is saved during validation.
- **Best checkpoint selection**: `best.pth` is selected according to the lowest validation loss.

## Repository structure

```text
LERAFormer/
├── configs/
│   └── config_case_luding.py          # Main configuration file
├── data/
│   ├── compute_mean_std.py            # Utility for computing normalization statistics
│   ├── generate_txt.py                # Utility for generating split text files
│   └── landslide_dataset/             # Expected dataset root, data not included
├── dataloader/
│   ├── Dataset_luding.py              # Dataset loading and online augmentation
│   └── bucket_hard_sampler.py         # Bucket-based hard-sample batch sampler
├── model/
│   ├── LERAFormer.py                  # Proposed model
│   ├── no_LERA.py                     # Ablation variant
│   ├── noLCAF.py                      # Ablation variant
│   ├── noCECA.py                      # Ablation variant
│   ├── noDGER.py                      # Ablation variant
│   ├── NOLSK.py                       # Ablation variant
│   ├── unet.py                        # Baseline model
│   ├── deeplabv3plus.py               # Baseline model
│   ├── segformer_b5.py                # Baseline model
│   ├── STUnet.py                      # Baseline model
│   ├── transunet.py                   # Baseline model
│   ├── swinunet.py                    # Baseline model
│   ├── LEFormer.py                    # Baseline model
│   └── _build_model.py                # Model builder
├── tools/
│   └── compute_norm_stats.py          # Split-wise normalization utility
├── utils/
│   ├── ema.py                         # Exponential moving average
│   ├── utils_loop.py                  # Training and validation loop
│   ├── utils_loss.py                  # Composite loss and metrics
│   └── loss/lovasz.py                 # Lovasz-Softmax loss
├── train.py                           # Training script
├── test.py                            # Testing script
├── requirements.txt                   # Python dependencies
└── README.md
```

## Environment

Recommended environment:

- Python 3.8+
- PyTorch 1.12+
- CUDA-enabled GPU for training

Install dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies are:

```text
torch
torchvision
numpy
h5py
einops
tqdm
matplotlib
```

## Dataset preparation

The code was developed for the **Landslide4Sense** benchmark dataset. The dataset itself is not included in this repository.

The expected dataset structure is:

```text
data/landslide_dataset/
├── TrainData/
│   ├── img/
│   ├── mask/
│   └── config/
│       ├── train.txt
│       ├── val.txt
│       └── test.txt
├── ValidData/
│   ├── img/
│   └── mask/
├── TestData/
│   ├── img/
│   └── mask/
└── norm_stats/
    ├── train.pt
    ├── val.pt
    └── test.pt
```

Each line in the split text files should contain one image-mask pair:

```text
image_name.h5,mask_name.h5
```

The default setting uses 14 input channels and 128 × 128 image patches.

## Configuration

The main configuration file is:

```text
configs/config_case_luding.py
```

Important default settings:

```python
total_epoch = 200
batch_size = 16
input_shape = (128, 128)
in_channels = 14
optimizer = "adamw"
base_lr = 2e-4
weight_decay = 1e-2
scheduler = "cosine"
warmup_epochs = 3
min_lr = 1e-6
ema_decay = 0.9999
```

To change the model, edit:

```python
cfg.model.model_type = "LERAFormer"
```

Available model names depend on `model/_build_model.py`.

## Training

Run:

```bash
python train.py
```

Training outputs are saved under:

```text
checkpoints/landslide4sense/loss_<timestamp>/
```

The following files are generated during training:

```text
epcXXX-trlossX.XXX-valossX.XXX-iouX.XXX-f1-X.XXX.pth
best.pth
best_val_loss.txt
best_thr.txt
log.txt
```

`best.pth` is saved when the validation loss reaches a new minimum. The foreground threshold selected on the validation set is saved in `best_thr.txt`.

## Testing

First set the checkpoint path in `configs/config_case_luding.py`:

```python
ckpt_test = "loss_YYYY_MM_DD_HH_MM_SS/best.pth"
```

Then run:

```bash
python test.py
```

## Loss function

The training objective is a weighted composite loss:

```text
0.2 × CrossEntropy
+ 0.25 × Dice
+ 0.25 × Focal
+ 0.2 × Lovasz-Softmax
+ 0.1 × Boundary loss
```

The focal loss uses `alpha = 0.75` and `gamma = 4.0`. The boundary loss is implemented using Sobel-gradient differences between predicted foreground probabilities and ground-truth masks.

## Online augmentation

The dataloader applies online augmentations including:

- random horizontal and vertical flipping;
- random 90-degree rotation;
- spectral perturbation;
- random cropping;
- random erasing.

## Checkpoints and large files

Large files are not recommended for direct GitHub upload. The following files should usually be excluded from the repository:

- trained weights: `*.pth`, `*.pt`, `*.ckpt`;
- dataset files: `*.h5`, `*.tif`, `*.tiff`;
- checkpoint folders: `checkpoints/`;
- cache folders: `__pycache__/`, `.idea/`.

If trained weights are released, it is recommended to provide them through Zenodo, Google Drive, OneDrive, or another external storage service, and then add the download link here.

## Reproducibility notes

The training script sets a fixed random seed. However, exact reproducibility may still vary across GPU models, CUDA versions, PyTorch versions, and low-level library implementations.

For better reproducibility, please keep the following files consistent:

- train/validation/test split files;
- normalization statistics;
- configuration file;
- checkpoint file;
- selected validation threshold;
- software and CUDA versions.

## Citation

If you use this code, please cite the corresponding paper after publication.

```bibtex
@article{liao_leraformer,
  title   = {Boundary-aware dual-branch network with explicit region correction for landslide segmentation from multi-source remote sensing data},
  author  = {Liao, Jun and Hao, Lina and Liu, Xi and Xu, Qiang and Sajinkumar, K. S.},
  journal = {Applied Computing and Geosciences},
  year    = {2026},
  note    = {Manuscript submitted}
}
```
