"""Write SimplifiedUNetSR.md: a copy of the notebook with its saved outputs that GitHub can display.

GitHub does not render this notebook (it is too large). The Markdown copy keeps:
* the markdown cells as they are;
* the code;
* the printed output;
* the tables, as HTML;
* the figures, as PNG files in SimplifiedUNetSR_files/.

Progress bars and other stderr output are left out.

Run it from PBL/ whenever the notebook's outputs change:  uv run python notebook_to_markdown.py
Standard library only.
"""
import base64
import json
import re
import sys
from pathlib import Path

WIDGET = "application/vnd.jupyter.widget-view+json"     # tqdm progress bars: skipped


def convert(src):
    src = Path(src)
    out, img_dir = src.with_suffix(".md"), src.with_name(f"{src.stem}_files")
    img_dir.mkdir(exist_ok=True)
    for old in img_dir.glob("*.png"):
        old.unlink()
    nb = json.loads(src.read_text(encoding="utf-8"))
    parts = [f"> **Generated from [`{src.name}`]({src.name}) by `notebook_to_markdown.py`**, because GitHub cannot render "
             "the notebook itself. It shows the code and the saved outputs; edit the notebook, not this file."]
    images = []
    for cell in nb["cells"]:
        text = "".join(cell["source"])
        if cell["cell_type"] != "code":
            parts.append(text)
            continue
        parts.append(f"```python\n{text}\n```")
        for o in cell.get("outputs", []):
            data = o.get("data", {})
            if o["output_type"] == "stream" and o["name"] == "stdout":
                parts.append("```text\n" + "".join(o["text"]).rstrip() + "\n```")
            elif o["output_type"] == "error":
                parts.append(f"```text\n{o['ename']}: {o['evalue']}\n```")
            elif "image/png" in data:
                images.append(img_dir / f"output_{len(images) + 1}.png")
                images[-1].write_bytes(base64.b64decode(data["image/png"]))
                parts.append(f"![output {len(images)}]({img_dir.name}/{images[-1].name})")
            elif "<table" in "".join(data.get("text/html", [])):           # pandas tables; GitHub drops <style>
                parts.append(re.sub(r"<style.*?</style>", "", "".join(data["text/html"]), flags=re.S).strip())
            elif "text/plain" in data and WIDGET not in data:
                parts.append("```text\n" + "".join(data["text/plain"]).rstrip() + "\n```")
    out.write_text("\n\n".join(parts) + "\n", encoding="utf-8", newline="\n")
    md = out.read_text(encoding="utf-8")                   # check: every figure is linked and exists
    assert all(f"{img_dir.name}/{p.name})" in md and p.stat().st_size for p in images)
    print(f"wrote {out} ({len(nb['cells'])} cells, {len(images)} figures in {img_dir}/)")


if __name__ == "__main__":
    convert(sys.argv[1] if len(sys.argv) > 1 else "SimplifiedUNetSR.ipynb")
