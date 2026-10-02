import base64, io, json, re, urllib.request
from pathlib import Path

import fitz
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "2021-sample"
OUT.mkdir(exist_ok=True)
TMP = ROOT / ".tmp_sample_pdfs"
TMP.mkdir(exist_ok=True)

# These official Practice Test 1 PDFs contain the same 2021 Sample maths/thinking
# questions. The Reading PDF contains the same material plus one extra cloze
# section, so below we explicitly map the matching question groups.
URLS = {
    "reading": "https://education.nsw.gov.au/content/dam/main-education/schooling/parents-and-carers/choosing-a-school-setting/selective-high-schools-and-opportunity-classes-parents/documents/shs-practice-tests-2026-entry/PT1_SHS_reading_questions.pdf",
    "maths": "https://education.nsw.gov.au/content/dam/main-education/schooling/parents-and-carers/choosing-a-school-setting/selective-high-schools-and-opportunity-classes-parents/documents/shs-practice-tests-2026-entry/PT1_SHS_maths_questions_Final.pdf",
    "thinking": "https://education.nsw.gov.au/content/dam/main-education/schooling/parents-and-carers/choosing-a-school-setting/selective-high-schools-and-opportunity-classes-parents/documents/shs-practice-tests-2026-entry/PT1_SHS_thinking_skills_questions.pdf",
}

def download(name):
    p = TMP / f"{name}.pdf"
    req = urllib.request.Request(URLS[name], headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response, open(p, "wb") as out:
        out.write(response.read())
    return fitz.open(p)

def render(page, rect, zoom=1.75):
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rect, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

def to_uri(im, quality=80):
    b = io.BytesIO()
    im.convert("RGB").save(b, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode("ascii")

def vertical_trim(im, pad=14):
    """Only top/bottom whitespace may be removed. Full horizontal width is preserved."""
    a = np.array(im.convert("L"))
    ink = (a < 246).sum(axis=1)
    rows = np.where(ink >= 3)[0]
    if not len(rows):
        return im
    top = max(0, int(rows[0]) - pad)
    bottom = min(im.height, int(rows[-1]) + pad + 1)
    return im.crop((0, top, im.width, bottom))

def crop(page, top, bottom, zoom=1.75, pad=12):
    top = max(0, top)
    bottom = min(page.rect.height, bottom)
    return vertical_trim(render(page, fitz.Rect(0, top, page.rect.width, bottom), zoom), pad)

def markers(doc, start_q, end_q, x_limit=70):
    """Question-number markers in strict sequence, avoiding numbers inside question text."""
    expected = start_q
    result = []
    for pno in range(doc.page_count):
        candidates = []
        for w in doc[pno].get_text("words"):
            x0, y0, x1, y1, text, *_ = w
            if text.isdigit() and x0 < x_limit and y0 < 770:
                q = int(text)
                if start_q <= q <= end_q:
                    candidates.append((y0, x0, q, pno, x0, y0, x1, y1))
        candidates.sort()
        for _, _, q, pno, x0, y0, x1, y1 in candidates:
            if q == expected:
                result.append((q, pno, x0, y0, x1, y1))
                expected += 1
                if expected > end_q:
                    return result
    raise RuntimeError(f"Stopped at Q{expected}; expected through Q{end_q}")

def one_question_each(doc, start_q, end_q, x_limit=70, zoom=1.75):
    """Exactly one question per image. Never append a following page/question."""
    ms = markers(doc, start_q, end_q, x_limit)
    out = {}
    for i, m in enumerate(ms):
        q, pno, x0, y0, x1, y1 = m
        page = doc[pno]
        if i + 1 < len(ms) and ms[i + 1][1] == pno:
            bottom = ms[i + 1][3] - 8
        else:
            bottom = min(page.rect.height - 48, 775)
        out[q] = crop(page, y0 - 15, bottom, zoom, 14)
    return out

def find_page(doc, phrase):
    for pno in range(doc.page_count):
        if phrase in doc[pno].get_text("text"):
            return pno
    raise RuntimeError(f"Could not find page containing: {phrase}")

def find_word_number(page, number, x_min=None, x_max=None):
    hits = []
    for w in page.get_text("words"):
        x0, y0, x1, y1, text, *_ = w
        if text == str(number):
            if x_min is not None and x0 < x_min:
                continue
            if x_max is not None and x0 > x_max:
                continue
            hits.append((x0, y0, x1, y1))
    if not hits:
        raise RuntimeError(f"Number {number} not found on page")
    return hits

def mask_rect_on_crop(im, page_rect, crop_top, zoom=1.75, extra=4):
    """White out an obsolete printed question number; UI supplies the real number."""
    x0, y0, x1, y1 = page_rect
    d = ImageDraw.Draw(im)
    px0 = max(0, int(x0 * zoom) - extra)
    py0 = max(0, int((y0 - crop_top) * zoom) - extra)
    px1 = min(im.width, int(x1 * zoom) + extra)
    py1 = min(im.height, int((y1 - crop_top) * zoom) + extra)
    d.rectangle((px0, py0, px1, py1), fill="white")
    return im


def crop_and_mask(page, top, bottom, box, zoom=1.8, pad=10):
    """Render full width, mask only a stale printed number, then trim vertically."""
    top = max(0, top)
    bottom = min(page.rect.height, bottom)
    im = render(page, fitz.Rect(0, top, page.rect.width, bottom), zoom)
    x0, y0, x1, y1 = box
    d = ImageDraw.Draw(im)
    d.rectangle((
        max(0, int(x0 * zoom) - 6),
        max(0, int((y0 - top) * zoom) - 6),
        min(im.width, int(x1 * zoom) + 6),
        min(im.height, int((y1 - top) * zoom) + 6),
    ), fill="white")
    return vertical_trim(im, pad)

def exact_number_box(page, number, prefer_x_min=None):
    hits = []
    for w in page.get_text("words"):
        x0, y0, x1, y1, text, *_ = w
        if text == str(number) and y0 < 770:
            if prefer_x_min is not None and x0 < prefer_x_min:
                continue
            hits.append((x0, y0, x1, y1))
    if hits:
        return hits[0]
    # Some gap numbers are not emitted as standalone words by the PDF text extractor.
    # search_for still returns their exact printed rectangle.
    rects = page.search_for(str(number))
    rects = [r for r in rects if r.y0 < 770 and r.width < 30]
    if rects:
        r = rects[0]
        return (r.x0, r.y0, r.x1, r.y1)
    raise RuntimeError(f"Could not find printed number {number}")

reading = download("reading")
maths = download("maths")
thinking = download("thinking")
assets = {}

# ---------- Reading Q1-8: same numbering and same original extracts ----------
r_basic = markers(reading, 1, 8, 70)
r_poem = markers(reading, 17, 22, 70)
r_by_q = {m[0]: m for m in (r_basic + r_poem)}
q1 = r_by_q[1]
assets["r_ctx_1"] = to_uri(crop(reading[2], 35, reading[2].rect.height - 48, 1.5, 10), 78)
assets["r_ctx_2"] = to_uri(crop(reading[3], 30, q1[3] - 18, 1.5, 10), 78)
r_1_8 = one_question_each(reading, 1, 8, 70, 1.75)
for q in range(1, 9):
    assets[f"r_q_{q}"] = to_uri(r_1_8[q], 82)

# ---------- Reading old Q9-14 = current PT1 Q17-22 (same poem/questions) ----------
poem_page = find_page(reading, "The Fish")
assets["r_ctx_3"] = to_uri(crop(reading[poem_page], 45, reading[poem_page].rect.height - 48, 1.5, 10), 78)

poem_markers = r_poem
for i, (old_q, current_q) in enumerate(zip(range(9, 15), range(17, 23))):
    m = r_by_q[current_q]
    page = reading[m[1]]
    if i + 1 < len(poem_markers) and poem_markers[i + 1][1] == m[1]:
        bottom = poem_markers[i + 1][3] - 8
    else:
        bottom = min(page.rect.height - 48, 775)
    im = crop_and_mask(page, m[3] - 15, bottom, (m[2], m[3], m[4], m[5]), 1.75, 14)
    assets[f"r_q_{old_q}"] = to_uri(im, 82)

# ---------- Reading old Q15-20 = current PT1 Octopus gaps Q23-28 ----------
oct_start = find_page(reading, "Octopus farming")
# The article spans this page and the following page. Keep the article and A-G choices,
# but use focused per-gap images as the current question so there is no ambiguity.
oct_pages = [oct_start, oct_start + 1]
for idx, pno in enumerate(oct_pages, 1):
    page = reading[pno]
    # Keep content area, remove footer only.
    assets[f"r_ctx_{3+idx}"] = to_uri(crop(page, 40, page.rect.height - 48, 1.5, 10), 78)

# A-G options are on second Octopus page; include them as one common source panel.
p = reading[oct_start + 1]
option_blocks = []
for b in p.get_text("blocks"):
    text = b[4]
    if any(text.strip().startswith(letter) for letter in "ABCDEFG"):
        option_blocks.append(b)
if option_blocks:
    options_top = min(b[1] for b in option_blocks) - 8
    options_bottom = max(b[3] for b in option_blocks) + 8
else:
    options_top, options_bottom = 150, p.rect.height - 55
assets["r_options_15_20"] = to_uri(crop(p, options_top, options_bottom, 1.65, 8), 80)

octopus_patterns = {
    23: "highly developed and curious creatures",
    24: "shown to use tools",
    25: "reported to be in decline",
    26: "feeding them difficult and expensive",
    27: "basic stock for breeding",
    28: "eggs in captivity",
}
for old_q, current_q in zip(range(15, 21), range(23, 29)):
    # Locate the exact article paragraph by its unique surrounding text.
    # This avoids mistaking the section heading "Questions 23-28" for a gap.
    found_page = None
    gap_block = None
    pattern = octopus_patterns[current_q].lower()
    for pg in [reading[oct_start], reading[oct_start + 1]]:
        for b in pg.get_text("blocks"):
            txt = b[4].lower()
            if pattern in txt:
                found_page = pg
                gap_block = b
                break
        if gap_block is not None:
            break
    if gap_block is None:
        raise RuntimeError(f"Octopus paragraph for {current_q} not found")
    bx0, by0, bx1, by1 = gap_block[:4]
    top = by0 - 7
    bottom = by1 + 7

    # Mask the renumbered PT1 gap number only if it can be identified inside
    # this paragraph. Content must never be discarded just because the printed
    # number is embedded in the text stream.
    number_rects = [
        r for r in found_page.search_for(str(current_q))
        if r.y0 >= by0 - 3 and r.y1 <= by1 + 3 and r.width < 30
    ]
    if number_rects:
        nr = number_rects[0]
        im = crop_and_mask(found_page, top, bottom, (nr.x0, nr.y0, nr.x1, nr.y1), 1.8, 7)
    else:
        im = crop(found_page, top, bottom, 1.8, 7)
    assets[f"r_q_{old_q}"] = to_uri(im, 82)

# ---------- Reading old Q21-30 = current PT1 Dreams Q29-38 ----------
dream_page = find_page(reading, "Read the four extracts below on the theme of dreams")
page = reading[dream_page]
# Source A starts after the prompt list.
extract_a_blocks = [b for b in page.get_text("blocks") if "Extract A" in b[4]]
if not extract_a_blocks:
    raise RuntimeError("Extract A not found")
extract_a_top = min(b[1] for b in extract_a_blocks) - 8
assets["r_ctx_6"] = to_uri(crop(page, extract_a_top, page.rect.height - 48, 1.5, 10), 78)

page2 = reading[dream_page + 1]
assets["r_ctx_7"] = to_uri(crop(page2, 40, page2.rect.height - 48, 1.5, 10), 78)

# Prompt rows use the right-side printed numbers 29-38.
prompt_hits = {}
for w in page.get_text("words"):
    x0, y0, x1, y1, text, *_ = w
    if text.isdigit() and 29 <= int(text) <= 38 and x0 > 400:
        prompt_hits[int(text)] = (x0, y0, x1, y1)
if sorted(prompt_hits) != list(range(29, 39)):
    raise RuntimeError(f"Dream prompt markers incorrect: {sorted(prompt_hits)}")

ordered = [(q, prompt_hits[q]) for q in range(29, 39)]
for i, (current_q, box) in enumerate(ordered):
    old_q = 21 + i
    y0 = box[1]
    next_y = ordered[i + 1][1][1] if i + 1 < len(ordered) else y0 + 42
    top = y0 - 11
    bottom = next_y - 4
    im = crop_and_mask(page, top, bottom, box, 1.8, 7)
    assets[f"r_q_{old_q}"] = to_uri(im, 82)

# ---------- Maths: 35 individually isolated questions ----------
m_imgs = one_question_each(maths, 1, 35, 70, 1.75)
for q in range(1, 36):
    assets[f"m_q_{q}"] = to_uri(m_imgs[q], 82)

# ---------- Thinking: 40 individually isolated questions ----------
t_imgs = one_question_each(thinking, 1, 40, 70, 1.75)
for q in range(1, 41):
    assets[f"t_q_{q}"] = to_uri(t_imgs[q], 82)

# Strict completeness check: 105 individual questions plus Reading contexts/options.
expected = (
    [f"r_ctx_{i}" for i in range(1, 8)]
    + ["r_options_15_20"]
    + [f"r_q_{i}" for i in range(1, 31)]
    + [f"m_q_{i}" for i in range(1, 36)]
    + [f"t_q_{i}" for i in range(1, 41)]
)
missing = [k for k in expected if k not in assets]
extra = [k for k in assets if k not in expected]
if missing or extra:
    raise RuntimeError(f"Asset check failed. Missing={missing}, extra={extra}")

# Keep 11 asset JS files because index.html already loads these exact filenames.
items = [(k, assets[k]) for k in expected]
weights = [len(k) + len(v) + 8 for k, v in items]
chunks = []
pos = 0
remaining_weight = sum(weights)
remaining_chunks = 11
for _ in range(11):
    target = remaining_weight / remaining_chunks
    chunk = {}
    size = 0
    while pos < len(items):
        k, v = items[pos]
        w = weights[pos]
        items_left = len(items) - pos - 1
        chunks_left = remaining_chunks - 1
        if chunk and size + w > target and items_left >= chunks_left:
            break
        chunk[k] = v
        size += w
        pos += 1
    chunks.append(chunk)
    remaining_weight -= size
    remaining_chunks -= 1

if pos != len(items) or any(not c for c in chunks):
    raise RuntimeError("Chunking failed")

for i, chunk in enumerate(chunks, 1):
    content = "window.ASSETS=window.ASSETS||{};Object.assign(window.ASSETS," + json.dumps(chunk, separators=(",", ":")) + ");\n"
    (OUT / f"assets-{i:02d}.js").write_text(content, encoding="utf-8")

print("Validated and rebuilt 105 individual question images: 30 Reading, 35 Maths, 40 Thinking.")
