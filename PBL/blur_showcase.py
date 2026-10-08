"""x2 super-resolution on realistically blurred images: writes results/x2_blur_showcase.md and its figures.

Three methods are compared on test images degraded like real photos (soft lens, defocus, camera shake, other resize
filters):
* bicubic;
* UnetSR+ as in the paper (runs/BSD300_x2_mixge);
* the same model fine-tuned on random blur (runs/BSD300_x2_mixge_rand_ft_lr0.0001).

The data functions, the model and the metrics come from the notebook itself, so there is only one copy of that code.
Run from PBL/ once those two runs exist (README section 7.4):  uv run python blur_showcase.py
"""
import contextlib
import io
import json
from pathlib import Path

# ---- the notebook's own definitions: parameters, device, settings, data functions, model, metrics, bicubic -----------
NOTEBOOK = json.loads(Path("SimplifiedUNetSR.ipynb").read_text(encoding="utf-8"))
NEEDED = ("def pick_device", "def as_bool", "def make_hr", "class UNetSR", "def psnr", "def bicubic_upscale")
for cell in NOTEBOOK["cells"]:
    src = "".join(cell["source"])
    if "parameters" in cell["metadata"].get("tags", []):
        exec(src)
        SCALE, MODE, DOWNLOAD = 2, "calibrate", False    # x2; "calibrate" skips the notebook's training-only code
    elif cell["cell_type"] == "code" and any(m in src for m in NEEDED):
        with contextlib.redirect_stdout(io.StringIO()):
            exec(src)

OUT = Path("results/x2_blur_showcase.md")
FIGS = Path("results/x2_blur_showcase")
NETS = {"UnetSR+ (paper training)": "BSD300_x2_mixge",
        "UnetSR+ fine-tuned on random blur": "BSD300_x2_mixge_rand_ft_lr0.0001"}
SAMPLES = ["data/BSD300/test/189080.jpg", "data/BSD300/test/108082.jpg", "data/BSD300/test/102061.jpg",
           "data/SET14/test/monarch.png", "data/SET14/train/barbara.png"]
TEST_SETS = {"BSD300 test split (100 images)": sorted(Path("data/BSD300/test").glob("*.jpg")),
             "Set14 (all 14 images)": sorted(Path("data/SET14").rglob("*.png"))}


def kernel_blur(hr, k):
    """A PIL image convolved with the 2-D kernel k (sums to 1), reflect padding, rounded back to 8 bit."""
    k = torch.tensor(k, dtype=torch.float32)
    ry, rx = k.shape[0] // 2, k.shape[1] // 2
    x = F.pad(to_uint8_tensor(hr).float()[None], (rx, rx, ry, ry), mode="reflect")
    x = F.conv2d(x, k.expand(3, 1, *k.shape).contiguous(), groups=3)
    return Image.fromarray(x[0].round().clamp(0, 255).byte().permute(1, 2, 0).numpy())


yy, xx = np.mgrid[-2:3, -2:3]
DISK = (xx ** 2 + yy ** 2 <= 4).astype(np.float32)          # out-of-focus blur: a disk of radius 2 HR px
DISK /= DISK.sum()
MOTION = np.eye(7, dtype=np.float32)[::-1] / 7               # camera shake: a 7 HR px diagonal stroke

# (key, title, what it imitates, HR PIL image -> LR PIL image). Blur is applied to the full-size image, then it is
# down-scaled x2, as in the classical model y = (x * k) down-scaled.
CONDITIONS = [
    ("bilinear", "Training condition: bilinear down-scaling",
     "Exactly how the training inputs were made (the paper's protocol). The reference point.",
     lambda hr: make_lr(hr, 2, "bilinear")),
    ("bicubic", "Bicubic down-scaling",
     "The standard benchmark degradation (\"BI\"). Bicubic is sharper than bilinear, so the input carries a little more "
     "detail than in training. The paper model, which learned to undo bilinear blur, over-sharpens it: see the halos on "
     "the tiger's whiskers and the butterfly's edges.", lambda hr: make_lr(hr, 2, "bicubic")),
    ("lens_soft", "Soft lens: Gaussian blur σ = 1 HR px",
     "A slightly soft lens or mild over-sharpening loss. At ×2 this is σ = 0.5 LR px, the edge of the fine-tuning range.",
     lambda hr: make_lr(blur_hr(hr, 1.0), 2, "bilinear")),
    ("lens_strong", "Strong lens blur: Gaussian σ = 2 HR px",
     "A clearly soft photo. σ = 1 LR px, twice the fine-tuning range.",
     lambda hr: make_lr(blur_hr(hr, 2.0), 2, "bilinear")),
    ("defocus", "Out of focus: disk blur, radius 2 HR px",
     "A camera focused slightly wrong. A disk kernel, a shape that neither model saw during training.",
     lambda hr: make_lr(kernel_blur(hr, DISK), 2, "bilinear")),
    ("motion", "Camera shake: diagonal motion blur, 7 HR px",
     "A hand-held shot with a slow shutter. Direction-dependent blur, which neither model saw during training.",
     lambda hr: make_lr(kernel_blur(hr, MOTION), 2, "bilinear")),
]


def load_net(run):
    net = UNetSR(2).to(DEVICE).eval()
    net.load_state_dict(torch.load(f"runs/{run}/unetsr_x2_mixge.pt", weights_only=True))
    return net


def tensors(images):
    return torch.stack([to_uint8_tensor(im) for im in images]).float().div(255)


@torch.no_grad()
def upscale(method, lr):
    """Upscale a [B,3,h,w] batch with bicubic or a network, 8 images at a time; returns a CPU tensor (not clamped)."""
    out = [bicubic_upscale(b, b.shape[-1] * 2) if method == "bicubic" else method(b.to(DEVICE)).float().cpu()
           for b in lr.split(8)]
    return torch.cat(out)


METHODS = {"bicubic": "bicubic", **{name: load_net(run) for name, run in NETS.items()}}

# ---- scores: mean PSNR / SSIM (RGB, as in the paper) per test set, condition and method ------------------------------
scores = {}
for set_name, files in TEST_SETS.items():
    hrs = [make_hr(Image.open(f).convert("RGB")) for f in files]
    hr_t = tensors(hrs)
    for key, *_, degrade in CONDITIONS:
        lr_t = tensors([degrade(h) for h in hrs])
        for name, method in METHODS.items():
            sr = upscale(method, lr_t)
            scores[set_name, key, name] = (psnr(sr, hr_t).mean().item(), ssim(sr, hr_t, per_image=True).mean().item())
print("scores done:", len(scores))

# ---- figures ------------------------------------------------------------------------------------------------------
FIGS.mkdir(parents=True, exist_ok=True)
C0, C1 = 64, 192                                             # the centre 128x128 of each 256x256 image, shown enlarged


def condition_figure(key, title, degrade):
    hrs = [make_hr(Image.open(f).convert("RGB")) for f in SAMPLES]
    hr_t, lr_t = tensors(hrs), tensors([degrade(h) for h in hrs])
    cols = [("input (128×128, pixels enlarged)", F.interpolate(lr_t, scale_factor=2, mode="nearest"))]
    cols += [(name if name == "bicubic" else name.replace("UnetSR+ ", "UnetSR+\n"), upscale(m, lr_t))
             for name, m in METHODS.items()]
    cols.append(("ground truth", hr_t))
    fig, axes = plt.subplots(len(hrs), len(cols), figsize=(2.6 * len(cols), 2.75 * len(hrs)), squeeze=False)
    for r, path in enumerate(SAMPLES):
        for c, (name, imgs) in enumerate(cols):
            img = imgs[r:r + 1]
            label = name if c == len(cols) - 1 else f"{name}\n{psnr(img, hr_t[r:r + 1]).item():.2f} dB"
            axes[r, c].imshow(img[0, :, C0:C1, C0:C1].clamp(0, 1).permute(1, 2, 0).numpy())
            axes[r, c].set_title(label if r == 0 or c == 0 else label.split("\n")[-1], fontsize=8)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
        axes[r, 0].set_ylabel(Path(path).stem, fontsize=9)
    fig.suptitle(f"×2: {title}  (centre crop; PSNR of the whole image)", fontsize=10)
    plt.tight_layout()
    fig.savefig(FIGS / f"{key}.jpg", dpi=75, pil_kwargs={"quality": 90})
    plt.close(fig)


for key, title, _, degrade in CONDITIONS:
    condition_figure(key, title, degrade)

bsd = next(iter(TEST_SETS))
fig, ax = plt.subplots(figsize=(10, 3.6))                    # a dot plot: the PSNR axis does not start at zero
for i, (name, marker) in enumerate(zip(METHODS, "s^o")):
    vals = [scores[bsd, key, name][0] for key, *_ in CONDITIONS]
    ax.plot(np.arange(len(CONDITIONS)) + (i - 1) * 0.12, vals, marker, markersize=8, label=name)
ax.set_xticks(np.arange(len(CONDITIONS)), [t.split(":")[0] for _, t, *_ in CONDITIONS], fontsize=8)
ax.set(ylabel="PSNR [dB]", title=f"×2, {bsd}: mean PSNR by input degradation")
ax.grid(axis="y", alpha=0.3)
ax.legend(fontsize=8)
plt.tight_layout()
fig.savefig(FIGS / "summary.png", dpi=90)
plt.close(fig)


# ---- the Markdown ----------------------------------------------------------------------------------------------------
def pair(set_name, key, name):
    p, s = scores[set_name, key, name]
    return f"{p:.2f} ({s:.3f})"


def gain(set_name, key, a, b):
    return round(scores[set_name, key, a][0] - scores[set_name, key, b][0], 2) + 0.0   # + 0.0: no "-0.00"


paper, tuned = list(NETS)
lines = [
    "# ×2 super-resolution on realistically blurred images",
    "",
    "> Generated by [`blur_showcase.py`](../blur_showcase.py) from the trained ×2 models; re-run it to refresh. "
    "**No image here was used for training**: the models learned from the 200 BSD300 training images only.",
    "",
    "The models in the main results were trained and tested on one kind of input: a photo shrunk with a bilinear filter. "
    "Real photos are blurred in other ways too: by the lens, by a missed focus, by a shaking hand, or by another resize "
    "filter. This page shows what the ×2 model does with such inputs.",
    "",
    "**What is compared** (each image: 256×256 ground truth → 128×128 input → 256×256 output):",
    "",
    "* **bicubic**: plain bicubic enlargement, the baseline;",
    f"* **{paper}**: the 300-epoch UnetSR+ model trained as in the paper (`runs/{NETS[paper]}`);",
    f"* **{tuned}**: the same model after 100 more epochs with a random Gaussian blur (σ ≤ 0.5 LR px = 1 HR px) and a "
    f"random bilinear / bicubic / box down-scaling (`runs/{NETS[tuned]}`).",
    "",
    "Scores are mean PSNR in dB (higher is better) with SSIM in brackets, on RGB, as in the paper (README §5).",
    "",
    "## Summary",
    "",
    "![mean PSNR by degradation](x2_blur_showcase/summary.png)",
    "",
]
for set_name in TEST_SETS:
    lines += [f"**{set_name}**", "",
              f"| input degradation | bicubic | {paper} | {tuned} | fine-tuning gain |", "|---|---|---|---|---|"]
    lines += [f"| {title} | {pair(set_name, key, 'bicubic')} | {pair(set_name, key, paper)} | "
              f"{pair(set_name, key, tuned)} | {gain(set_name, key, tuned, paper):+.2f} dB |"
              for key, title, *_ in CONDITIONS]
    lines.append("")

lead = {k: gain(bsd, k, paper, "bicubic") for k, *_ in CONDITIONS}            # paper model over bicubic, BSD300
ft = {k: gain(bsd, k, tuned, paper) for k, *_ in CONDITIONS}                  # fine-tuned over paper model, BSD300
lines += [
    "**What this shows** (BSD300 test split; Set14 agrees):",
    "",
    f"* **The paper model is tuned to one kind of input.** On its training condition it beats bicubic by "
    f"{lead['bilinear']:.2f} dB. On any other input that lead shrinks:",
    f"  * {lead['lens_soft']:.2f} dB with a soft lens and {lead['defocus']:.2f} dB out of focus;",
    f"  * {lead['lens_strong']:.2f} dB with strong lens blur and {lead['motion']:.2f} dB with camera shake;",
    f"  * only {lead['bicubic']:.2f} dB on a bicubic-shrunk input. That input is *sharper* than the training input, "
    "so the network, which learned to undo bilinear blur, over-sharpens it (§2: the tiger's whiskers, the "
    "butterfly's edges).",
    f"* **Fine-tuning on random blur removes the resize mismatch and helps with round blur.**",
    f"  * {ft['bicubic']:+.2f} dB on bicubic down-scaling (bicubic was one of its random filters);",
    f"  * {ft['lens_soft']:+.2f} / {ft['lens_strong']:+.2f} dB on soft / strong Gaussian blur;",
    f"  * {ft['defocus']:+.2f} dB on the disk blur, which it never saw.",
    f"  * It costs {-ft['bilinear']:.2f} dB on the training condition.",
    f"* **Camera shake is not helped** ({ft['motion']:+.2f} dB). The fine-tuning only saw round (isotropic) blur, while "
    "motion blur has a direction. Training with directional kernels, as SRMD and BSRGAN do, is the next step.",
    f"* **Strong blur stays strong.** With strong lens blur or camera shake, even the better model is only "
    f"{max(gain(bsd, 'lens_strong', tuned, 'bicubic'), gain(bsd, 'motion', tuned, 'bicubic')):.2f} dB or less above "
    "bicubic. The blur has removed detail that ×2 super-resolution cannot invent.",
    "",
    "Each section below shows five test images: three from BSD300 (a portrait, a tiger, a château) and two from Set14 "
    "(a butterfly, a patterned scarf). The panels show the centre of each image enlarged, so the detail is visible; "
    "the dB values are for the whole image.",
    "",
]
for i, (key, title, what, _) in enumerate(CONDITIONS, start=1):
    lines += [f"## {i} {title}", "", what, "",
              f"BSD300 mean: bicubic {scores[bsd, key, 'bicubic'][0]:.2f} dB, paper model "
              f"{scores[bsd, key, paper][0]:.2f} dB, fine-tuned {scores[bsd, key, tuned][0]:.2f} dB.", "",
              f"![{title}](x2_blur_showcase/{key}.jpg)", ""]
lines += ["## Reproduce", "", "From `PBL/`, after the ×2 runs of README §7.4 exist (about 30 s on a GPU):", "",
          "```bash", "uv run python blur_showcase.py", "```", ""]
OUT.write_text("\n".join(lines), encoding="utf-8", newline="\n")
print(f"wrote {OUT} and {len(CONDITIONS) + 1} figures in {FIGS}/")
