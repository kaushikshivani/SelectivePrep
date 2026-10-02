import base64, io, json, os, urllib.request
from pathlib import Path
import fitz
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "2021-sample"
OUT.mkdir(exist_ok=True)

URLS = {
    "reading": "https://education.nsw.gov.au/content/dam/main-education/schooling/parents-and-carers/choosing-a-school-setting/selective-high-schools-and-opportunity-classes-parents/documents/shs-practice-tests-2026-entry/PT1_SHS_reading_questions.pdf",
    "maths": "https://education.nsw.gov.au/content/dam/main-education/schooling/parents-and-carers/choosing-a-school-setting/selective-high-schools-and-opportunity-classes-parents/documents/shs-practice-tests-2026-entry/PT1_SHS_maths_questions_Final.pdf",
    "thinking": "https://education.nsw.gov.au/content/dam/main-education/schooling/parents-and-carers/choosing-a-school-setting/selective-high-schools-and-opportunity-classes-parents/documents/shs-practice-tests-2026-entry/PT1_SHS_thinking_skills_questions.pdf",
}

tmp = ROOT / ".tmp_sample_pdfs"
tmp.mkdir(exist_ok=True)

def download(name):
    p = tmp / f"{name}.pdf"
    urllib.request.urlretrieve(URLS[name], p)
    return fitz.open(p)

def render(page, rect, zoom=1.65):
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rect, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

def to_uri(im, quality=76):
    b = io.BytesIO()
    im.convert("RGB").save(b, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode("ascii")

def vertical_content(page, zoom=1.5, pad=10):
    # Keep the entire page width. Only remove unused space at top/bottom and the PDF footer.
    ys = []
    for b in page.get_text("blocks"):
        x0, y0, x1, y1, text, *_ = b
        if text.strip() and y0 < page.rect.height - 45:
            ys.append((y0, y1))
    if not ys:
        top, bottom = 0, page.rect.height - 50
    else:
        top = max(0, min(y[0] for y in ys) - pad)
        bottom = min(page.rect.height - 45, max(y[1] for y in ys) + pad)
    return render(page, fitz.Rect(0, top, page.rect.width, bottom), zoom)

def sequential_markers(doc, start_q, end_q, x_limit=115):
    expected = start_q
    found = []
    for pno in range(doc.page_count):
        candidates = []
        for w in doc[pno].get_text("words"):
            x0, y0, x1, y1, text, *_ = w
            if text.isdigit() and x0 < x_limit and y0 < 760:
                q = int(text)
                if start_q <= q <= end_q:
                    candidates.append((y0, x0, q, pno, x0, y0, x1, y1))
        candidates.sort()
        for _, _, q, pno, x0, y0, x1, y1 in candidates:
            if q == expected:
                found.append((q, pno, x0, y0, x1, y1))
                expected += 1
                if expected > end_q:
                    return found
    raise RuntimeError(f"Could not find all questions {start_q}-{end_q}; stopped at {expected}")

def question_images(doc, start_q, end_q, x_limit=115, zoom=1.7):
    markers = sequential_markers(doc, start_q, end_q, x_limit)
    out = {}
    for i, marker in enumerate(markers):
        q, pno, _, y0, _, _ = marker
        nxt = markers[i + 1] if i + 1 < len(markers) else None
        end_page = doc.page_count - 1 if nxt is None else nxt[1]
        end_y = None if nxt is None else nxt[3] - 6
        parts = []
        for cur in range(pno, end_page + 1):
            page = doc[cur]
            top = max(0, y0 - 14) if cur == pno else 32
            bottom = end_y if cur == end_page and end_y is not None else min(775, page.rect.height - 55)
            if bottom <= top + 12:
                bottom = min(775, top + 260)
            # Critical rule: never crop left/right. Preserve full PDF width.
            parts.append(render(page, fitz.Rect(0, top, page.rect.width, bottom), zoom))
        if len(parts) == 1:
            out[q] = parts[0]
        else:
            width = max(p.width for p in parts)
            height = sum(p.height for p in parts)
            joined = Image.new("RGB", (width, height), "white")
            y = 0
            for part in parts:
                joined.paste(part, (0, y))
                y += part.height
            out[q] = joined
    return out

reading = download("reading")
maths = download("maths")
thinking = download("thinking")

assets = {}

# Reading source sections. Full width is preserved throughout.
assets["r_ctx_1"] = to_uri(vertical_content(reading[2], 1.45), 74)

# Page 4 contains the end of the source plus Q1-Q4. Context stops safely before Q1.
q1_marker = sequential_markers(reading, 1, 14, 85)[0]
q1_y = q1_marker[3]
assets["r_ctx_2"] = to_uri(render(reading[3], fitz.Rect(0, 28, reading[3].rect.width, q1_y - 18), 1.45), 74)

assets["r_ctx_3"] = to_uri(vertical_content(reading[5], 1.45), 74)
assets["r_ctx_4"] = to_uri(vertical_content(reading[7], 1.45), 74)
assets["r_ctx_5"] = to_uri(vertical_content(reading[8], 1.45), 74)
assets["r_ctx_6"] = to_uri(vertical_content(reading[9], 1.45), 74)
assets["r_ctx_7"] = to_uri(vertical_content(reading[10], 1.45), 74)

for q, im in question_images(reading, 1, 14, 85, 1.65).items():
    assets[f"r_q_{q}"] = to_uri(im, 76)

# These two are the common original-paper prompt/option areas for their sections.
assets["r_q_15_20_prompt"] = to_uri(
    render(reading[8], fitz.Rect(0, 500, reading[8].rect.width, min(775, reading[8].rect.height - 50)), 1.55), 74
)
assets["r_q_21_30_prompt"] = to_uri(
    render(reading[9], fitz.Rect(0, 65, reading[9].rect.width, 535), 1.55), 74
)

for q, im in question_images(maths, 1, 35, 85, 1.7).items():
    assets[f"m_q_{q}"] = to_uri(im, 76)

for q, im in question_images(thinking, 1, 40, 115, 1.7).items():
    assets[f"t_q_{q}"] = to_uri(im, 76)

expected = 7 + 14 + 2 + 35 + 40
if len(assets) != expected:
    raise RuntimeError(f"Expected {expected} assets, got {len(assets)}")

# Keep exactly 11 asset files because index.html already loads assets-01.js ... assets-11.js.
items = list(assets.items())
weights = [len(k) + len(v) + 8 for k, v in items]
total = sum(weights)
chunks = []
start = 0
remaining_weight = total
remaining_chunks = 11
for chunk_no in range(11):
    target = remaining_weight / remaining_chunks
    chunk = {}
    size = 0
    while start < len(items):
        k, v = items[start]
        w = weights[start]
        items_left_after = len(items) - (start + 1)
        chunks_left_after = remaining_chunks - 1
        must_leave = items_left_after >= chunks_left_after
        if chunk and size + w > target and must_leave:
            break
        chunk[k] = v
        size += w
        start += 1
    chunks.append(chunk)
    remaining_weight -= size
    remaining_chunks -= 1

if start != len(items) or len(chunks) != 11 or any(not c for c in chunks):
    raise RuntimeError("Asset chunking failed")

for i, chunk in enumerate(chunks, 1):
    content = "window.ASSETS=window.ASSETS||{};Object.assign(window.ASSETS," + json.dumps(chunk, separators=(",", ":")) + ");\n"
    (OUT / f"assets-{i:02d}.js").write_text(content, encoding="utf-8")

print("Rebuilt 2021 Sample assets with full horizontal width and vertical-only cropping.")
