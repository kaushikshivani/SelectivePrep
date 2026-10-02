from pathlib import Path
import base64, io, json, re
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEST_DIRS = ["practice-test-3", "2023-hybrid"]

def vertical_trim_keep_width(im, pad=18):
    """Trim only top/bottom whitespace; never crop left/right.

    Also removes isolated PDF page numbers/footers when separated from real content
    by a large blank vertical gap.
    """
    im = im.convert("RGB")
    gray = np.array(im.convert("L"))
    dark = gray < 245
    row_ink = dark.sum(axis=1)
    active = np.where(row_ink >= 3)[0]
    if len(active) == 0:
        return im

    # Build row groups, allowing normal line spacing inside a content block.
    groups = []
    start = prev = int(active[0])
    for y in active[1:]:
        y = int(y)
        if y - prev > 18:
            groups.append([start, prev])
            start = y
        prev = y
    groups.append([start, prev])

    # Remove tiny isolated footer/page-number blocks after a large blank gap.
    while len(groups) >= 2:
        last = groups[-1]
        prevg = groups[-2]
        gap = last[0] - prevg[1]
        h = last[1] - last[0] + 1
        ink = int(row_ink[last[0]:last[1]+1].sum())
        # Conservative: only discard very small isolated blocks after a clearly large gap.
        if gap >= 80 and h <= 55 and ink <= max(2500, im.width * 7):
            groups.pop()
        else:
            break

    top = max(0, groups[0][0] - pad)
    bottom = min(im.height, groups[-1][1] + pad + 1)
    return im.crop((0, top, im.width, bottom))

def trim_data_uri(uri):
    if not uri.startswith("data:image/"):
        return uri
    header, b64 = uri.split(",", 1)
    raw = base64.b64decode(b64)
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    trimmed = vertical_trim_keep_width(im)
    out = io.BytesIO()
    # Keep JPEG output compact. All test images are document scans/renders.
    trimmed.save(out, "JPEG", quality=80, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode("ascii")

def load_original_html(test_dir):
    parts = sorted(test_dir.glob("part-*.txt"))
    if not parts:
        raise RuntimeError(f"No source parts found in {test_dir}")
    return "".join(p.read_text(encoding="utf-8") for p in parts)

def extract_assets(html):
    marker = "const ASSETS="
    start = html.find(marker)
    if start < 0:
        raise RuntimeError("const ASSETS not found")
    json_start = start + len(marker)

    # Locate the next DATA declaration rather than trying to regex-match nested JSON.
    data_pos = html.find("const DATA=", json_start)
    if data_pos < 0:
        raise RuntimeError("const DATA not found")
    between = html[json_start:data_pos]
    semi = between.rfind(";")
    if semi < 0:
        raise RuntimeError("ASSETS terminator not found")
    assets_text = between[:semi].strip()
    assets = json.loads(assets_text)

    replace_end = json_start + semi + 1
    return assets, start, replace_end

def split_assets(assets, target_chars=480000):
    chunks = []
    current = {}
    size = 0
    for key, value in assets.items():
        item_size = len(key) + len(value) + 10
        if current and size + item_size > target_chars:
            chunks.append(current)
            current = {}
            size = 0
        current[key] = value
        size += item_size
    if current:
        chunks.append(current)
    return chunks

def process(test_name):
    test_dir = ROOT / test_name
    html = load_original_html(test_dir)
    assets, replace_start, replace_end = extract_assets(html)

    fixed = {}
    for i, (key, value) in enumerate(assets.items(), 1):
        fixed[key] = trim_data_uri(value)
        if i % 20 == 0:
            print(f"{test_name}: processed {i}/{len(assets)} images")

    chunks = split_assets(fixed)

    # Remove any old generated asset JS files.
    for p in test_dir.glob("assets-*.js"):
        p.unlink()

    tags = []
    for i, chunk in enumerate(chunks, 1):
        name = f"assets-{i:02d}.js"
        content = "window.ASSETS=window.ASSETS||{};Object.assign(window.ASSETS," + json.dumps(chunk, separators=(",", ":")) + ");\n"
        (test_dir / name).write_text(content, encoding="utf-8")
        tags.append(f'<script src="{name}"></script>')

    # Replace huge embedded ASSETS with a small reference.
    direct_html = html[:replace_start] + "const ASSETS=window.ASSETS||{};" + html[replace_end:]

    # Load all image chunks directly, like the existing 2021 tests.
    insert = "\n".join(tags) + "\n"
    head_end = direct_html.find("</head>")
    if head_end < 0:
        raise RuntimeError("No </head> in assembled HTML")
    direct_html = direct_html[:head_end] + insert + direct_html[head_end:]

    # Add a cache-busting version comment for easier verification.
    direct_html = direct_html.replace("<head>", "<head>\n<!-- direct-assets-v2: vertical-only whitespace trim -->", 1)
    (test_dir / "index.html").write_text(direct_html, encoding="utf-8")
    print(f"{test_name}: wrote direct index + {len(chunks)} parallel asset files")

for name in TEST_DIRS:
    process(name)
