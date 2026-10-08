# Web demo

Upload an image and get it back ×2, ×4 or ×8 larger from the PBL UNetSR models.

| piece | where it runs |
|---|---|
| [`site/index.html`](site/index.html): the page, one static file; [`site/samples/`](site/samples/): six low-res Set14/BSD300 test images to try | Vercel project `unetsr-upscaler` (<https://unetsr-upscaler.vercel.app>), deployed from `demo/site` with `vercel deploy --prod` |
| [`server.py`](server.py): Flask, `GET /health` and `POST /upscale` | your machine, or a Colab GPU behind ngrok |
| [`colab.ipynb`](colab.ipynb): starts `server.py` on Colab and opens the tunnel | [Open in Colab](https://colab.research.google.com/github/Rah-Rah-Mitra/UNet-Training-Blended-PBL-/blob/master/demo/colab.ipynb) |

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
2. Cell 2 opens the tunnel and prints the link to open. The last cell keeps the server running.
3. The badge on the page shows *GPU online* while the notebook runs and *Backend offline* when it doesn't.

**With the repo owner's token**, the tunnel uses the account's fixed dev domain, `evolved-oarfish-guiding.ngrok-free.app`.
The page has that domain hardcoded, so the link is just https://unetsr-upscaler.vercel.app.

**With anyone else's token**, that domain belongs to another account, so the notebook falls back to the URL ngrok gives
that token. It then prints `https://unetsr-upscaler.vercel.app/?api=<that URL>`, which points the page at that backend.
The page accepts `?api=` only for an `https://` origin.

Limits: the longest side of the input may be at most 4096 / scale px. Requests run one at a time.
