"""Smoke check for demo/server.py (needs demo/weights/unetsr_x2.pt). Run: python demo/test_server.py"""
import io

from PIL import Image

from server import app


def png(w, h):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (120, 60, 30)).save(buf, "PNG")
    buf.seek(0)
    return buf


c = app.test_client()

r = c.get("/health")
assert 2 in r.json["scales"], r.json
assert r.headers["Access-Control-Allow-Origin"] == "*"

r = c.post("/upscale", data={"scale": "2", "image": (png(20, 13), "a.png")})
assert r.status_code == 200, r.json
assert Image.open(io.BytesIO(r.data)).size == (40, 26)

r = c.post("/upscale", data={"scale": "2", "image": (png(2049, 8), "big.png")})
assert r.status_code == 413, r.status_code

r = c.post("/upscale", data={"scale": "3", "image": (png(8, 8), "a.png")})
assert r.status_code == 400

print("ok")
