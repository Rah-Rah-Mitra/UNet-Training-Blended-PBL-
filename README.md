# Simplified U-Net super-resolution (UnetSR / UnetSR+): PBL edition

This repository is a fork of the authors' code for

> Z. Lu and Y. Chen, **"Single Image Super Resolution based on a Modified U-net with Mixed Gradient Loss"**,
> arXiv:1911.09428 (2019); *Signal, Image and Video Processing* 16, 1143–1151 (2022).

It adds [`PBL/`](PBL/): a reproducible, single-notebook PyTorch re-implementation of the paper. It trains and scores
the models against the paper's Table 2, and studies how they cope with blurred inputs. The original code is kept
unchanged next to it.

## Start here

| you want to … | go to |
|---|---|
| run it | [`PBL/README.md` §1](PBL/README.md#1-quick-start-uv): four commands with uv, on any NVIDIA GPU, Apple Silicon or CPU |
| read the code | [`PBL/SimplifiedUNetSR.md`](PBL/SimplifiedUNetSR.md): all of the code, in 13 sections, with the outputs of the last run. It is a Markdown copy of [`PBL/SimplifiedUNetSR.ipynb`](PBL/SimplifiedUNetSR.ipynb), which GitHub cannot display |
| see the results | [`PBL/results/final_results.md`](PBL/results/final_results.md): Table A (ours vs the paper), Table B (robustness to blur) |
| understand the results | [`PBL/README.md` §7](PBL/README.md#7-does-it-perform-like-the-paper) |
| explain the latest changes (meeting of 1 Oct 2026) | [`PBL/README.md` §10](PBL/README.md#10-code-guide-what-was-added-for-the-1-oct-feedback): code guide and a ten-minute talk outline |

## Quick start

```bash
cd PBL
uv sync --extra cu130                       # or cu126 (older NVIDIA GPUs) or cpu (no NVIDIA GPU / Apple Silicon)
uv run python download_datasets.py          # BSD300, SET14 (and ICDAR2003, if its HTTP servers answer)
uv run papermill SimplifiedUNetSR.ipynb runs/smoke.ipynb -p SMOKE_TEST True    # 20-second check
uv run papermill SimplifiedUNetSR.ipynb runs/BSD300_x4_mixge.ipynb -p SCALE 4 -p LOSS mixge   # ~20 min on a laptop GPU
```

## Results at a glance

These are final results: 300 epochs, BSD300 test set (100 images), PSNR in dB on RGB, a single run each, on an
RTX 2070 Max-Q.

| scale | bicubic (paper) | UnetSR (MSE), paper / ours | UnetSR+ (MixGE), paper / ours |
|---|---|---|---|
| ×2 | 26.65 | 29.42 / 29.04 | 29.84 / 29.06 |
| ×4 | 23.51 | 24.83 / 24.58 | 24.95 / 24.55 |
| ×8 | 21.31 | 21.99 / 22.03 | 22.04 / 22.04 |

* **×8 matches the paper.** ×4 and ×2 are 0.25–0.78 dB below it, and the MixGE loss gives no measurable gain over
  MSE here.
* **SET14**: models trained on its 11 remaining images barely beat bicubic, while our BSD300 models reach or beat the
  paper's SET14 numbers. So the paper's SET14 models were most likely trained on more data.
* **Blur**: models trained on the paper's single degradation lose most of their advantage when the input is blurred
  differently. Fine-tuning with random blur recovers 0.2–0.3 dB at ×2.

Details and caveats: [`PBL/README.md` §7](PBL/README.md#7-does-it-perform-like-the-paper).

## What is where

| path | what it is | origin |
|---|---|---|
| [`PBL/`](PBL/) | everything new: the notebook, the dataset downloader, the uv environment, committed results and the guides | this project |
| [`Unet/`](Unet/) | the authors' modified U-net (`Umodel.py`, `unet_parts.py`), its solver and the unused `GraLoss.py` | authors |
| `SRCNN/`, `VDSR/`, `EDSR/`, `FSRCNN/`, `DRCN/`, `DBPN/`, `SRGAN/`, `SubPixelCNN/`, `bicubic/` | the baseline models of the paper's Table 2 | [`icpm/super-resolution`](https://github.com/icpm/super-resolution) |
| `main.py`, `super_resolve.py`, `dataset/`, `pytorch_ssim/`, `progress_bar.py` | the original training entry point, data loader and helpers | authors / icpm |

The original code does not run as published: its dataset folder was never released, and `CenterCrop` is commented out.
[`PBL/README.md` §8](PBL/README.md#8-notes-on-the-original-code) lists every issue and how the notebook handles it.
The notebook's `UNetSR` reproduces the authors' `UNet2/4/8` exactly: the same weights give identical outputs.

## The authors' original usage notes

Kept as published. These flags drive `main.py`, which needs the unpublished `dataset/output2/images/{train,test}`
folders.

```
python main.py -m unet -uf 2 -lr 0.001 -n 300

-b [batchSize]
-t [testBatchSize]
-seed [random seed]
-m [model]
-uf [upscale_factor]
-lr [learning rate]
-n [epochs]

-m [model]
SimUnet ---> simunet
ESPCN   ---> sub
SRCNN   ---> srcnn
VDSR    ---> vdsr
EDSR    ---> edsr
FSRCNN  ---> fsrcnn
DRCN    ---> drcn
SRGAN   ---> srgan
Bicubic ---> bi
```

(`-t` and `-seed` are listed here but not defined in `main.py`.)

## Citation

```bibtex
@article{lu2022unetsr,
  title   = {Single image super-resolution based on a modified U-net with mixed gradient loss},
  author  = {Lu, Zhengyang and Chen, Ying},
  journal = {Signal, Image and Video Processing},
  volume  = {16},
  pages   = {1143--1151},
  year    = {2022},
  note    = {arXiv:1911.09428}
}
```
