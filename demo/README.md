# Web demo

Upload an image and get it back ×2, ×4 or ×8 larger from the PBL UNetSR models.

| piece | where it runs |
|---|---|
| [`site/index.html`](site/index.html): the page, one static file | Vercel (root directory `demo/site`, no build) |
| [`server.py`](server.py): Flask, `GET /health` and `POST /upscale` | your machine, or a Colab GPU behind ngrok |
| [`colab.ipynb`](colab.ipynb): starts `server.py` on Colab and opens the tunnel | [Open in Colab](https://colab.research.google.com/github/Rah-Rah-Mitra/UNet-Training-Blended-PBL-/blob/web-demo/demo/colab.ipynb) |

The models are the ×2 and ×4 random-blur fine-tunes and the 300-epoch ×8 MixGE run, all on BSD300 (see
[`PBL/README.md`](../PBL/README.md)). They load into the authors' `Unet/Umodel.py` classes. The weights are in the GitHub
release [`demo-weights-v1`](https://github.com/Rah-Rah-Mitra/UNet-Training-Blended-PBL-/releases/tag/demo-weights-v1).

## Run it locally

Put `unetsr_x{2,4,8}.pt` in `demo/weights/`, either downloaded from the release or copied from `PBL/runs/`. Then:

```bash
python demo/server.py        # http://localhost:8765 (CPU is fine for small images)
python demo/test_server.py   # smoke check
```

Needs torch, Pillow, numpy, flask and matplotlib (the authors' model file imports it).

## Run the public site

1. Open the notebook in Colab with **Open in Colab** above. It is preset to a T4 GPU. Add the Colab secret `NGROK_AUTHTOKEN`, then choose **Run all**.
2. The last cell keeps the server running at `https://evolved-oarfish-guiding.ngrok-free.app`. That is the account's fixed ngrok dev domain, and the page has it hardcoded.
3. Open the Vercel site. The badge shows *GPU online* while the notebook runs and *Backend offline* when it doesn't.

Limits: the longest side of the input may be at most 4096 / scale px. Requests run one at a time.
