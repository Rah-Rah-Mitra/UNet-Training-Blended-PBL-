# GPU session: verify the notebook on a real GPU

Start a new local Claude Code session on the GPU machine and give it this file, for example:
*"Read `PBL/GPU_SESSION.md` and do what it says."*

The session has three jobs:

1. **Set up.** Follow `README.md` on a real GPU machine.
2. **Test.** Run the checklist below. These changes were written in a CPU-only cloud sandbox, and most are untested.
3. **Train.** Do **one training run of at most 30 minutes**, then record and push the results.

---

## Context

* **Repository and branch:** `rah-rah-mitra/unet-training-blended-pbl-`, branch **`pbl/unet-sr-notebook`**. The
  repository is a fork of the authors' code, `Mnster00/simplifiedUnetSR`.
* **Scope:** everything new lives in `PBL/`. Do not modify the original files outside `PBL/`.
* **What is in `PBL/`:**
  * `SimplifiedUNetSR.ipynb` holds all the code: data, model, losses, metrics, training, evaluation, figures.
  * `download_datasets.py` fetches the data.
  * `README.md` explains everything; read §1, §5 and §7 first.
  * `pyproject.toml` and `requirements.txt` set up the environment.
  * `results/` holds small committed evidence: the protocol study, the SET14 subset search, and two 40-epoch CPU runs.
* **Where it was built:** a CPU-only Linux sandbox that could reach only GitHub and PyPI.
* **Verified in that sandbox:**
  * the environment from PyPI;
  * the dataset download from the pinned GitHub mirrors;
  * the notebook on CPU;
  * the protocol study.
* **Never verified:** CUDA and MPS, the download.pytorch.org wheels (`uv sync --extra cu130/cu126`), Windows and
  macOS, and ICDAR2003 downloads (plain HTTP).
* **The last commit before this file** added fixes from an independent review, and most of them have never been run.
  On CPU, two things were checked. The default smoke run gives exactly the earlier results. `MODE=calibrate`
  reproduces the committed CSVs. Everything else in the checklist is new.

## Rules

* Work and push on **`pbl/unet-sr-notebook`** (`git push origin pbl/unet-sr-notebook`). Do not push to `master`, and
  do not open a pull request unless the user asks for one.
* Never commit `data/`, `runs/` (except `runs/.gitkeep`), `.venv/`, `uv.lock` or `*.pt`. The `.gitignore` already
  excludes them.
* **The notebook is the only copy of the code.** Edit its cells directly. Clear the outputs before you commit:
  `uv run jupyter nbconvert --clear-output --inplace SimplifiedUNetSR.ipynb`.
* **Keep the README reproducible.** If a README command or claim does not do what the README says, fix the code or the
  README.
* **GPU time budget: 30 minutes of training in total.** The tests below need only a few minutes; most are 2-epoch
  smoke runs.
* **Some choices are deliberate.** Do not "fix" the decisions listed in the appendix unless you can show they are wrong.

---

## Step 0: record the machine

Run these and keep the output for the report:

```bash
nvidia-smi                   # GPU name, driver version, CUDA version
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
uv --version                 # uv >= 0.9.4 is needed only for the requirements.txt --torch-backend route
git log --oneline -3         # the commit you are testing
```

## Step 1: set up exactly as README §1 and §3 say

Run every command from `PBL/`:

```bash
uv sync --extra cu130        # driver >= 580; otherwise cu126 (see the README §1 table; RTX 50xx needs cu130)
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.backends.mps.is_available())"
uv run python download_datasets.py --datasets bsd300 set14
uv run python download_datasets.py --verify-only --datasets bsd300 set14
uv run python download_datasets.py --datasets icdar2003 --timeout 10   # plain HTTP: may fail, which is fine (README §3)
```

Expected results:

* The version check prints `2.x+cu130 True False` (or `+cu126`).
* The download ends with BSD300 `train=200 test=100` and SET14 `train=11 test=3`.
* `--verify-only` says OK.
* If ICDAR2003 downloads, that is a bonus: test T12 can then check its protocol, which nobody has done yet.

Write down every place where the output differs from what README §1 or §3 says.

## Step 2: test checklist

Run every command from `PBL/`. Each papermill call writes an executed copy to `runs/` (keep these until you write
the report). Add `--log-output` to watch the cell output live. Record PASS or FAIL and a note for each test.

Commands are written for bash. In PowerShell they are the same, except `CUDA_VISIBLE_DEVICES=-1 cmd` becomes
`$env:CUDA_VISIBLE_DEVICES="-1"; cmd` (and `Remove-Item Env:CUDA_VISIBLE_DEVICES` afterwards).

`NB` below stands for `uv run papermill SimplifiedUNetSR.ipynb`.

| # | what it checks | command (from `PBL/`) | expected |
|---|---|---|---|
| T1 | smoke run on the GPU | `NB runs/t1.ipynb -p SMOKE_TEST True` | Exit 0. Section 2 prints the GPU with no WARNING. Section 4 prints "same outputs as the repo's UNet2/UNet4/UNet8". `runs/BSD300_x4_mixge_smoke/metrics.json` exists. (The CPU result was PSNR 12.2555 dB; the GPU differs slightly, which is fine.) |
| T2 | resuming a finished run on the GPU (was a crash: "RNG state must be a torch.ByteTensor") | run T1's command again (`runs/t2.ipynb`) | Prints `resuming BSD300_x4_mixge_smoke after epoch 2` and `... already has 2 epochs (EPOCHS = 2) - nothing to train`. Exit 0. |
| T3 | resuming in the middle of training on the GPU (optimiser state on the GPU) | `NB runs/t3a.ipynb -p SCALE 8 -p EPOCHS 2 -p EVAL_EVERY 1 -p RUN_NAME resume_gpu`, then the same with `-p EPOCHS 3` (`runs/t3b.ipynb`) | The second run prints `resuming resume_gpu after epoch 2` and trains only epoch 3. `runs/resume_gpu/history.csv` has epochs 1, 2, 3 once each. |
| T4 | a changed setting never overwrites a run | (a) `NB runs/t4a.ipynb -p SMOKE_TEST True -p LAMBDA_G 0.01`; (b) `NB runs/t4b.ipynb -p SMOKE_TEST True -p LAMBDA_G 0.01 -p RUN_NAME BSD300_x4_mixge_smoke`; (c) as (b) plus `-p RESUME False` | (a) A new folder, `runs/BSD300_x4_mixge_lg0.01_smoke`. (b) Fails with `RuntimeError: ... was trained with other settings (LAMBDA_G: 0.1 there, 0.01 now)`, and T1's `metrics.json` is unchanged. (c) Prints `RESUME = False: training BSD300_x4_mixge_smoke from scratch ...`. Exit 0. |
| T5 | booleans written as text | `NB runs/t5a.ipynb -p SMOKE_TEST true -p SOBEL_NORM false`, then `NB runs/t5b.ipynb -p SMOKE_TEST True -p AMP maybe` | (a) The run folder is `BSD300_x4_mixge_sobelraw_smoke`, and its `config.json` has `"SOBEL_NORM": false`. (b) Fails with `ValueError: AMP must be True or False, not 'maybe'`. |
| T6 | bf16 AMP gate | `NB runs/t6.ipynb -p SMOKE_TEST True -p AMP True` | Compute capability ≥ 8.0: the folder is `BSD300_x4_mixge_bf16_smoke`, `"AMP": true`, and the loss is finite. Older GPU: prints `AMP requested, but bfloat16 is only fast on ... - training in float32`, and there is no `_bf16` tag. |
| T7 | hold-out set | `NB runs/t7.ipynb -p SMOKE_TEST True -p VAL_HOLDOUT 5` | The folder is `BSD300_x4_mixge_val5_smoke`. It has test sets `BSD300 4 (repo_crop)` and `BSD300_VAL 4 (repo_crop)`. In `metrics.json`, `BSD300_VAL` has the verdict `no paper value`. |
| T8 | cross-dataset scores and SET14_ALL | `NB runs/t8.ipynb -p SMOKE_TEST True -p DATASET SET14 -p EVAL_SETS "[SET14, SET14_ALL, BSD300]"` | Prints `WARNING: SET14_ALL includes the 11 images this model is trained on ...`. In `metrics.json`: SET14 has a normal verdict; SET14_ALL has `no paper value` plus a note; BSD300 has `none: model trained on SET14`. |
| T9 | odd parameter forms (not a smoke run, because smoke mode forces `EVAL_EVERY=1`) | `NB runs/t9.ipynb -p DATASET SET14 -p EPOCHS 1 -p EVAL_EVERY 0 -p RUN_NAME 2024 -p EVAL_SETS "['SET14']"` | Exit 0 (this used to be a ZeroDivisionError); `runs/2024/metrics.json` exists. |
| T10 | the repo's loss has no paper row | `NB runs/t10.ipynb -p SMOKE_TEST True -p LOSS l1_ssim` | The BSD300 verdict is `no paper value`. The curves show no paper lines. |
| T11 | data paths and data problems | (a) `NB runs/t11a.ipynb -p SMOKE_TEST True -p DATASET SET14 -p DATA_DIR "~/pbl_data_test"`; (b) `cp data/SET14/test/comic.png data/SET14/test/._comic.png`, then `uv run python download_datasets.py --verify-only --datasets set14` and `NB runs/t11b.ipynb -p SMOKE_TEST True -p DATASET SET14`, then delete `._comic.png`; (c) `mkdir empty_data`, then `NB runs/t11c.ipynb -p SMOKE_TEST True -p DOWNLOAD False -p DATA_DIR empty_data`, then remove `empty_data` | (a) Downloads into `~/pbl_data_test`, trains, exit 0; delete that folder afterwards. (b) verify-only says OK; the notebook prints `test SET14 3 (repo_crop)` and exits 0. (c) Fails with `RuntimeError: Training set BSD300 is not complete in .../empty_data/BSD300: found {'train': 0, 'test': 0} images ... DOWNLOAD is False ...`. |
| T12 | protocol study | `NB runs/t12.ipynb -p MODE calibrate` | Exit 0 within a few minutes. `git status` shows `results/bicubic_calibration.csv` and `results/set14_subset_search.csv` unchanged. If ICDAR2003 is present, new ICDAR2003 rows appear; commit them, and README §5 can then say which ICDAR protocol matches the paper (`icdar_resize` is the unverified default). If it is absent, the notebook prints one line saying how to include it. |
| T13 | Jupyter use | `uv run jupyter lab SimplifiedUNetSR.ipynb`. Set `SCALE = 8`, `EPOCHS = 4`, `RUN_NAME = "jupyter_test"`, then Restart Kernel and Run All. During epoch 3, interrupt the kernel and re-run only the training-loop cell (section 8, second cell). | Training continues at epoch 3, not epoch 1. `runs/jupyter_test/history.csv` has epochs 1–4 once each. |
| T14 | portable weights | `CUDA_VISIBLE_DEVICES=-1 uv run python -c "import torch; sd = torch.load('runs/BSD300_x4_mixge_smoke/unetsr_x4_mixge.pt', weights_only=True); print(next(iter(sd.values())).device)"` | Prints `cpu` with no error. |
| T15 | README commands, literally | Run the commands of README §1 (papermill block), §3 (downloader block) and §7 exactly as written, except the long training grid. | No command fails. Every printed claim matches the README. |

If a test fails: find the cause, fix the notebook (or the README), re-run that test, and note the fix in the report.

## Step 3: one training run (at most 30 minutes)

Run **×8 on BSD300 with MixGE (UnetSR+)**, the paper's settings for up to 300 epochs. Two reasons for this choice:

* ×8 is the fastest scale.
* The repository already has 40-epoch CPU results for this configuration, so the GPU run extends them:

  |  | PSNR / SSIM |
  |---|---|
  | 40 epochs on CPU | 21.746 dB / 0.5201 |
  | paper, UnetSR+ | 22.0368 dB / 0.5235 |
  | bicubic | 21.3115 dB / 0.4933 |

1. **Start it.** Evaluating every 10 epochs instead of every epoch saves time:

   ```bash
   uv run papermill SimplifiedUNetSR.ipynb runs/x8_mixge_gpu.ipynb -p SCALE 8 -p LOSS mixge -p EVAL_EVERY 10 --log-output
   ```

2. **Check the time estimate.** After the first epoch the notebook prints `epoch    1 ... (X s/epoch, ETA Y min)`.
   * If `Y` ≤ 27 minutes, let it finish.
   * Otherwise press Ctrl-C, set `N = floor(1500 / X)` rounded down to a multiple of 25 (the learning rate halves every
     25 epochs), and re-run the same command with `-p EPOCHS N`. The run name does not include `EPOCHS`, so the run
     resumes where it stopped.
3. **Optional second run.** If the run takes less than 13 minutes, use the rest of the 30-minute budget for the UnetSR
   counterpart: the same command with `-p LOSS mse` (output `runs/x8_mse_gpu.ipynb`).
4. **Read the results.** Look at `runs/BSD300_x8_mixge/metrics.json`: PSNR, SSIM, `verdict_absolute`, `verdict_gain`.
   Also look at `runs/BSD300_x8_mixge/figures/curves.png`.
5. **Save the evidence.** Copy `config.json`, `history.csv`, `metrics.json` and `figures/curves.png` into
   `results/gpu_x8_BSD300_<N>ep_mixge/` (and `..._mse/` if you ran it), the same way as
   `results/sanity_x8_BSD300_40ep_*`.
6. **Update README §7.**
   * Add the GPU result next to the 40-epoch CPU table: GPU name, epochs, minutes, PSNR / SSIM, gain over bicubic,
     the paper's values and both verdicts.
   * Update the short answer and the check table.
   * Keep the wording honest: it is a single run, and the paper does not state its number of epochs.
   * Also update "What was tested where" in README §1.

## Step 4: report and push

1. Write a short report for the user:
   * the machine (Step 0);
   * the setup deviations (Step 1);
   * the T1–T15 table (PASS / FAIL, fixes made);
   * the training result against the paper.
2. Commit the fixes, `results/gpu_*` and the README updates, with clear messages (outputs cleared in the notebook).
   Then run `git push origin pbl/unet-sr-notebook`.
3. Delete scratch folders you created (`empty_data`, `~/pbl_data_test`). `runs/` is git-ignored, so it can stay.

---

## Appendix A: deliberate design decisions (do not "fix" these)

1. **Data protocol.**
   * `PROTOCOL="auto"` uses `repo_crop` for BSD300 and SET14. That is a 256×256 centre crop with torchvision-0.4
     rounding and zero padding, plus a PIL bilinear LR image. It reproduces the paper's SET14 bicubic row exactly and
     BSD300 within 0.035 dB.
   * ICDAR2003 uses `icdar_resize` (224 bicubic resize + bilinear LR), which is unverified.
   * The paper's text (224 bicubic + bicubic LR) does not reproduce its own numbers.
   * Under `auto`, every test set gets its own protocol.
2. **SET14 split.** Train 11 / test 3, with test = comic, monarch, zebra. The subset search proves the 3 test images;
   the 11 training images are an assumption. `SET14_ALL` is all 14 images, for evaluation only.
3. **Metrics.**
   * RGB PSNR per image, then averaged.
   * SSIM identical to the repo's `pytorch_ssim` (Gaussian 11, σ 1.5, zero "same" padding).
   * Model outputs are not clamped, as in the repo.
   * The Y-channel metrics are extra.
4. **MixGE = MSE + 0.1 · MGE.**
   * MGE is computed on Sobel gradient magnitudes per RGB channel, with a valid convolution and ε 1e-6.
   * The kernels are divided by 8 by default (`SOBEL_NORM=True`). With raw kernels, MGE is 2.5–3× the MSE, which
     contradicts the paper calling it "auxiliary" and hurt training in an ablation (README §2).
5. **Optimisation.**
   * Adam (0.9, 0.999, 1e-8), lr 1e-3, `StepLR(25, 0.5)`, batch 1, weight decay 1e-6, 300 epochs, seed 123.
   * The model is seeded before it is built.
   * The last epoch is reported. Test sets are used for monitoring only.
6. **Model.** `UNetSR(scale)` reproduces the repo's `UNet2/4/8` exactly: the same module names, so the authors' weights
   load strictly, and identical outputs. That includes the right/bottom-biased `F.pad`, ×8 bilinear upsampling with
   `align_corners=True`, and sigmoid outputs for ×4 and ×8 only.
7. **Data loading.** Pairs are kept in memory as uint8 tensors, with `num_workers=0` (safe for notebooks on Windows and
   macOS).
8. **AMP.** bf16 autocast for the forward pass only, used only on CPUs and GPUs with compute capability ≥ 8.0. Losses
   and metrics are always fp32.
9. **Checkpoints.**
   * `last.pt` is written atomically after every epoch, loads with `weights_only=True`, and is always loaded on the CPU.
   * One configuration per run folder: the default `RUN_NAME` tags changed settings, and resuming checks all training
     settings.
10. **Environment.**
    * uv extras `cpu` / `cu126` / `cu130` pull torch from explicit download.pytorch.org indexes.
    * `requirements.txt` serves the `uv pip --torch-backend=auto` route.
    * `uv.lock` is not committed.

## Appendix B: untested changes from the last review round, and the test that covers each

| change | test |
|---|---|
| checkpoints loaded on the CPU (resume crashed on CUDA/MPS) | T2, T3 |
| default `RUN_NAME` tags changed settings; a settings mismatch raises an error instead of overwriting; `config.json` written after the check | T4, T5 |
| WEIGHT_DECAY, AUGMENT, AMP and VAL_HOLDOUT are part of the resume check | T4 |
| `true`/`false`/`yes`/`no`/0/1 accepted for booleans; anything else is an error | T5 |
| bf16 only on compute capability ≥ 8.0 (pre-Ampere GPUs used to get emulated bf16) | T6 |
| `VAL_HOLDOUT`: seeded hold-out set `<DATASET>_VAL` | T7 |
| per-set protocols under `auto`; cross-dataset sets, SET14_ALL and `l1_ssim` get no paper verdict; SET14_ALL warning | T8, T10 |
| `EVAL_EVERY <= 0`, numeric `RUN_NAME`, Python-style `EVAL_SETS` lists | T9 |
| `~` in `DATA_DIR`/`RUNS_DIR`; dot-files ignored like the downloader does; a clearer error for missing data | T11 |
| calibrate mode: batched (less memory), does not retry ICDAR2003 downloads, adds the SET14 subset search | T12 |
| parameter defaults looked up when a function runs; the training cell continues from the scheduler's epoch | T13 |
| inference weights saved as CPU tensors | T14 |
| GPU-generation hints: Blackwell needs driver ≥ 580 and cu130 | Step 0/1 (read the notebook's section-2 output) |
