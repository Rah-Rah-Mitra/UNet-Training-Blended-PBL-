"""Flask backend for the UNetSR web demo: POST an image, get it back upscaled x2/x4/x8.

Run from anywhere: python demo/server.py  (then open http://localhost:8765)
Weights: demo/weights/unetsr_x{2,4,8}.pt, the weights-only state dicts the PBL notebook saves.
"""
import io
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request, send_file
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from Unet.Umodel import UNet2, UNet4, UNet8  # noqa: E402  the authors' classes; same weights as the notebook's UNetSR

MAX_SIDE = 4096  # ponytail: rejects outputs over 4096 px a side instead of tiling; tile if bigger inputs are needed
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = {}
for scale, cls in {2: UNet2, 4: UNet4, 8: UNet8}.items():
    path = HERE / "weights" / f"unetsr_x{scale}.pt"
    if path.exists():
        net = cls(3, 3)
        net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        MODELS[scale] = net.eval().to(DEVICE)

app = Flask(__name__, static_folder=HERE / "site", static_url_path="")  # serves the page + samples locally


@app.after_request
def cors(resp):
    # The Vercel page calls this server through ngrok from another origin.
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "ngrok-skip-browser-warning"
    return resp


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.get("/health")
def health():
    return {"device": DEVICE, "scales": sorted(MODELS)}


@app.post("/upscale")
def upscale():
    scale = request.form.get("scale", type=int)
    if scale not in MODELS:
        return jsonify(error=f"scale must be one of {sorted(MODELS)}"), 400
    try:
        img = Image.open(request.files["image"].stream).convert("RGB")
    except Exception:
        return jsonify(error="could not read that file as an image"), 400
    w, h = img.size
    if max(w, h) * scale > MAX_SIDE:
        return jsonify(error=f"image too large: at x{scale} the longest side must be at most {MAX_SIDE // scale} px"), 413

    # Same recipe as super_resolve() in PBL/SimplifiedUNetSR.ipynb, section 11.
    lr = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().div(255)[None].to(DEVICE)
    lr = F.pad(lr, (0, (-w) % 16, 0, (-h) % 16), mode="replicate")
    with torch.inference_mode():
        sr = MODELS[scale](lr).float()[..., : h * scale, : w * scale].clamp(0, 1)
    out = Image.fromarray(sr[0].mul(255).round().byte().permute(1, 2, 0).cpu().numpy())

    buf = io.BytesIO()
    out.save(buf, "PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


if __name__ == "__main__":
    print(f"device={DEVICE} scales={sorted(MODELS)}")
    # ponytail: one request at a time so parallel jobs can't OOM the GPU; add a queue if traffic grows
    app.run(host="0.0.0.0", port=8765, threaded=False)
