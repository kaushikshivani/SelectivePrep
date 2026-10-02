import base64, io, json, re, urllib.request
from pathlib import Path

import fitz
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "2021-sample"
OUT.mkdir(exist_ok=True)

URLS = {
    "reading": "https://bettereducation.com.au/resources/download/nsw/SelectiveHighSchoolPlacement/sample%20tests/2021/reading-sample-test-questions.pdf",
    "maths": "https://bettereducation.com.au/resources/download/nsw/SelectiveHighSchoolPlacement/sample%20tests/2021/maths-sample-test-questions.pdf",
    "thinking": "https://bettereducation.com.au/resources/download/nsw/SelectiveHighSchoolPlacement/sample%20tests/2021/thinking-sample-test-questions.pdf",
}

TMP = ROOT / ".tmp_sample_pdfs"
TMP.mkdir(exist_ok=True)

def download(name):
    path = TMP / f"{name}.pdf"
    urllib.request.urlretrieve(URLS[name], path)
    return fitz.open(path)

def render(page, rect, zoom=1.7):
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rect, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

def to_uri(image, quality=78):
    buf = io.BytesIO()
    image.convert("RGB").save(buf, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

def vertical_trim(image, pad=12):
    """Trim only top/bottom whitespace. Never crop left/right."""
    grey = np.array(image.convert("L"))
    ink_per_row = (grey < 245).sum(axis=1)
    rows = np.where(ink_per_row >= 3)[0]
    if len(rows) == 0:
        return image
    top = max(0, int(rows[0]) - pad)
    bottom = min(image.height, int(rows[-1]) + pad + 1)
    return image.crop((0, top, image.width, bottom))

def sequential_markers(doc, start_q, end_q, x_limit=65):
    """Find real question numbers in sequence, ignoring numbers inside options/diagrams."""
    expected = start_q
    markers = []
    for pno in range(doc.page_count):
        candidates = []
        for word in doc[pno].get_text("words"):
            x0, y0, x1, y1, text, *_ = word
            if text.isdigit() and x0 < x_limit and y0 < 760:
                q = int(text)
                if start_q <= q <= end_q:
                    candidates.append((y0, x0, q, pno))
        candidates.sort()
        for y0, x0, q, pno in candidates:
            if q == expected:
                markers.append((q, pno, y0))
                expected += 1
                if expected > end_q:
                    return markers
    raise RuntimeError(f"Could not find all questions {start_q}-{end_q}; stopped at {expected}")

def individual_question_images(doc, start_q, end_q, x_limit=65, zoom=1.7):
    """
    One image per question. A crop NEVER runs onto a later page.
    If the next question starts on the same page, stop just above it.
    Otherwise stop before the footer, then remove only vertical whitespace.
    """
    markers = sequential_markers(doc, start_q, end_q, x_limit)
    result = {}
    for i, (q, pno, y0) in enumerate(markers):
        page = doc[pno]
        if i + 1 < len(markers) and markers[i + 1][1] == pno:
            bottom = markers[i + 1][2] - 8
        else:
            bottom = min(760, page.rect.height - 45)
        top = max(0, y0 - 14)
        image = render(page, fitz.Rect(0, top, page.rect.width, bottom), zoom)
        result[q] = vertical_trim(image, 14)
    return result

def crop_vertical(page, top, bottom, zoom=1.55, pad=10):
    return vertical_trim(render(page, fitz.Rect(0, top, page.rect.width, bottom), zoom), pad)

reading = download("reading")
maths = download("maths")
thinking = download("thinking")

assets = {}

# ---------- READING 1-8: two travel extracts, no questions inside the source ----------
q1_y = sequential_markers(reading, 1, 14, 65)[0][2]
assets["r_ctx_1"] = to_uri(crop_vertical(reading[2], 45, reading[2].rect.height - 45, 1.45, 10), 76)
assets["r_ctx_2"] = to_uri(crop_vertical(reading[3], 30, q1_y - 18, 1.45, 10), 76)

# ---------- READING 9-14: poem only ----------
assets["r_ctx_3"] = to_uri(crop_vertical(reading[5], 45, reading[5].rect.height - 45, 1.45, 10), 76)

# ---------- READING 15-20: source text, continuation, and A-G candidate sentences ----------
assets["r_ctx_4"] = to_uri(crop_vertical(reading[7], 55, reading[7].rect.height - 45, 1.45, 10), 76)
assets["r_ctx_5"] = to_uri(crop_vertical(reading[8], 45, 190, 1.45, 10), 76)

letters = []
for word in reading[8].get_text("words"):
    x0, y0, x1, y1, text, *_ = word
    if text in list("ABCDEFG") and 120 < x0 < 150:
        letters.append((text, y0))
if len(letters) != 7:
    raise RuntimeError(f"Expected A-G option rows, found {letters}")
options_top = min(y for _, y in letters) - 14
options_bottom = max(y for _, y in letters) + 38
assets["r_options_15_20"] = to_uri(crop_vertical(reading[8], options_top, options_bottom, 1.65, 8), 78)

# Individual gap line for each of 15-20, not a page containing several questions.
for q, pno in [(15, 7), (16, 7), (17, 7), (18, 7), (19, 7), (20, 8)]:
    matches = []
    for block in reading[pno].get_text("blocks"):
        x0, y0, x1, y1, text, *_ = block
        if re.search(rf"\b{q}\s", text):
            matches.append((y0, y1, text))
    if not matches:
        raise RuntimeError(f"Could not locate Reading Q{q}")
    y0, y1, _ = matches[-1]
    assets[f"r_q_{q}"] = to_uri(crop_vertical(reading[pno], y0 - 5, y1 + 5, 1.7, 4), 80)

# ---------- READING 21-30: extracts only + one prompt row per question ----------
# Extract A begins after the 21-30 question list on page 10.
extract_a_y = min(
    b[1] for b in reading[9].get_text("blocks") if "Extract A" in b[4]
)
assets["r_ctx_6"] = to_uri(crop_vertical(reading[9], extract_a_y - 8, reading[9].rect.height - 45, 1.5, 10), 76)

extract_b_y = min(
    b[1] for b in reading[10].get_text("blocks") if "Extract B" in b[4]
)
assets["r_ctx_7"] = to_uri(crop_vertical(reading[10], extract_b_y - 8, reading[10].rect.height - 45, 1.5, 10), 76)

question_rows = []
for word in reading[9].get_text("words"):
    x0, y0, x1, y1, text, *_ = word
    if text.isdigit() and 21 <= int(text) <= 30 and 430 < x0 < 480:
        question_rows.append((int(text), y0))
question_rows = sorted(set(question_rows))
if [q for q, _ in question_rows] != list(range(21, 31)):
    raise RuntimeError(f"Reading 21-30 row detection failed: {question_rows}")
for i, (q, y0) in enumerate(question_rows):
    next_y = question_rows[i + 1][1] if i + 1 < len(question_rows) else y0 + 38
    assets[f"r_q_{q}"] = to_uri(crop_vertical(reading[9], y0 - 10, next_y - 4, 1.7, 6), 80)

# Reading 1-14 exact question images.
for q, image in individual_question_images(reading, 1, 14, 65, 1.7).items():
    assets[f"r_q_{q}"] = to_uri(image, 80)

# ---------- MATHS: reviewed question-by-question, one question per image ----------
for q, image in individual_question_images(maths, 1, 35, 65, 1.7).items():
    assets[f"m_q_{q}"] = to_uri(image, 80)

# ---------- THINKING: reviewed question-by-question, one question per image ----------
for q, image in individual_question_images(thinking, 1, 40, 65, 1.7).items():
    assets[f"t_q_{q}"] = to_uri(image, 80)

expected_keys = (
    [f"r_ctx_{i}" for i in range(1, 8)]
    + ["r_options_15_20"]
    + [f"r_q_{i}" for i in range(1, 31)]
    + [f"m_q_{i}" for i in range(1, 36)]
    + [f"t_q_{i}" for i in range(1, 41)]
)
missing = [k for k in expected_keys if k not in assets]
extra = [k for k in assets if k not in expected_keys]
if missing or extra:
    raise RuntimeError(f"Asset validation failed. Missing={missing}, extra={extra}")

# Keep 11 JS asset files because index.html loads assets-01.js ... assets-11.js.
items = [(k, assets[k]) for k in expected_keys]
weights = [len(k) + len(v) + 8 for k, v in items]
remaining_weight = sum(weights)
remaining_chunks = 11
position = 0
chunks = []

for _ in range(11):
    target = remaining_weight / remaining_chunks
    chunk = {}
    size = 0
    while position < len(items):
        key, value = items[position]
        weight = weights[position]
        remaining_items_after = len(items) - position - 1
        chunks_after = remaining_chunks - 1
        must_leave_one_each = remaining_items_after >= chunks_after
        if chunk and size + weight > target and must_leave_one_each:
            break
        chunk[key] = value
        size += weight
        position += 1
    chunks.append(chunk)
    remaining_weight -= size
    remaining_chunks -= 1

if position != len(items) or len(chunks) != 11 or any(not chunk for chunk in chunks):
    raise RuntimeError("Asset chunking failed")

for i, chunk in enumerate(chunks, 1):
    content = "window.ASSETS=window.ASSETS||{};Object.assign(window.ASSETS," + json.dumps(chunk, separators=(",", ":")) + ");\n"
    (OUT / f"assets-{i:02d}.js").write_text(content, encoding="utf-8")

print("Rebuilt and individually validated all 105 questions for 2021 Sample.")
