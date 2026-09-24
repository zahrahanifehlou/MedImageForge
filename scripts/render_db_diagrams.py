"""Render README diagrams of the manifest database (Step-18 docs).

Generates docs/images/db-schema.png (entity diagram) and
docs/images/db-sample.png (live sample rows) using only Pillow.
Run from the repo root:  python scripts/render_db_diagrams.py
"""
from pathlib import Path
import sqlite3
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "artifacts" / "manifest.db"
OUT = ROOT / "docs" / "images"

F_TITLE = ImageFont.load_default(size=20)
F_COL = ImageFont.load_default(size=14)
F_SMALL = ImageFont.load_default(size=12)

INK = (40, 44, 52)
HEADER = (52, 73, 94)
BOX = (250, 250, 252)
EDGE = (120, 130, 140)
FK = (200, 90, 60)
PK = (60, 130, 180)
DIM = (120, 120, 128)

TABLES = {
    "files": {
        "cols": [
            ("id", "PK", PK), ("rel_path", "UNIQUE", None), ("kind", "slice|mask|metadata", None),
            ("patient_id", "-> patients", FK), ("window", "brain|bone", None),
            ("slice_no", "", None), ("sha256", "content identity", None),
            ("size_bytes", "", None), ("status", "raw|curated|...", None),
        ],
        "note": "every file in the raw zone - Step 4 ingest",
    },
    "curation": {
        "cols": [
            ("rel_path", "PK -> files", FK), ("decision", "accepted|rejected", None),
            ("reasons", "", None), ("curated_path", "", None),
            ("source_sha256", "raw bytes", None), ("curated_sha256", "normalized bytes", None),
        ],
        "note": "validation decisions - Step 5",
    },
    "patients": {
        "cols": [
            ("patient_id", "PK - REAL ID", PK), ("pseudonym", "UNIQUE PAT-xxxx", None),
            ("created_at", "", None),
        ],
        "note": "real<->pseudonym map - Step 6; never leaves the controlled zone",
    },
    "slice_annotations": {
        "cols": [
            ("id", "PK", PK), ("patient_id", "-> patients", FK), ("slice_no", "", None),
            ("mask_rel_path", "", None), ("source_file", "", None),
            ("source_sha256", "which CSV version", None), ("annotator", "", None),
        ],
        "note": "one row per labeled slice - Step 7",
    },
    "slice_labels": {
        "cols": [
            ("annotation_id", "PK,FK -> slice_annotations", FK),
            ("label_code", "PK,FK -> label_taxonomy", FK), ("value", "0|1", None),
        ],
        "note": "multi-label: 6 codes per annotation",
    },
    "label_taxonomy": {
        "cols": [
            ("code", "PK", PK), ("display_name", "", None),
            ("category", "hemorrhage_type|other_finding", None), ("source_column", "CSV column", None),
        ],
        "note": "the label dictionary - names are data, not schema",
    },
}

POS = {  # (x, y) of each box top-left
    "files": (40, 60), "curation": (40, 460),
    "patients": (520, 60), "slice_annotations": (520, 330),
    "slice_labels": (1000, 330), "label_taxonomy": (1000, 60),
}


def draw_table(d, name, spec, x, y, counts):
    w, row_h, pad = 400, 22, 8
    title_h = 46
    h = title_h + len(spec["cols"]) * row_h + 34
    d.rectangle([x, y, x + w, y + h], fill=BOX, outline=EDGE, width=2)
    d.rectangle([x, y, x + w, y + title_h], fill=HEADER)
    d.text((x + pad, y + 6), name, font=F_TITLE, fill="white")
    d.text((x + pad, y + 28), f"{counts.get(name, 0):,} rows", font=F_SMALL, fill=(200, 210, 220))
    cy = y + title_h + 4
    for col, typ, color in spec["cols"]:
        d.text((x + pad, cy), col, font=F_COL, fill=color or INK)
        if typ:
            d.text((x + 180, cy), typ, font=F_SMALL, fill=color or DIM)
        cy += row_h
    d.text((x + pad, cy + 2), spec["note"], font=F_SMALL, fill=DIM)
    return (x, y, x + w, y + h)


def arrow(d, p1, p2, label="", dashed=False):
    if dashed:
        steps = 24
        for i in range(steps):
            if i % 2 == 0:
                a = (p1[0] + (p2[0] - p1[0]) * i / steps, p1[1] + (p2[1] - p1[1]) * i / steps)
                b = (p1[0] + (p2[0] - p1[0]) * (i + 1) / steps, p1[1] + (p2[1] - p1[1]) * (i + 1) / steps)
                d.line([a, b], fill=DIM, width=2)
    else:
        d.line([p1, p2], fill=FK, width=3)
    # arrowhead
    import math
    ang = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    for da in (2.6, -2.6):
        d.line([p2, (p2[0] + 12 * math.cos(ang + da), p2[1] + 12 * math.sin(ang + da))],
               fill=FK if not dashed else DIM, width=3)
    if label:
        mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
        d.text((mx + 6, my - 8), label, font=F_SMALL, fill=FK)


def render_schema(counts):
    img = Image.new("RGB", (1440, 800), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((40, 18), "manifest.db - schema & relationships", font=F_TITLE, fill=INK)
    boxes = {}
    for name, spec in TABLES.items():
        boxes[name] = draw_table(d, name, spec, *POS[name], counts)
    f, c, p, sa, sl, lt = (boxes[k] for k in ("files", "curation", "patients", "slice_annotations", "slice_labels", "label_taxonomy"))
    arrow(d, (c[2], c[1] + 40), (f[0], f[1] + 240))                      # curation → files
    arrow(d, (sl[0], sl[1] + 60), (sa[2], sa[1] + 60), "FK")             # slice_labels → annotations
    arrow(d, (sl[0] + 120, sl[1]), (lt[0] + 120, lt[3]), "FK")           # slice_labels → taxonomy
    arrow(d, (sa[1] + 120, sa[1]), (p[1] + 160, p[3]), "patient_id", dashed=True)
    arrow(d, (f[2] - 60, f[1]), (p[0], p[1] + 40), "patient_id", dashed=True)
    d.text((40, 750), "solid = declared foreign key · dashed = logical join (patient_id)",
           font=F_SMALL, fill=DIM)
    img.save(OUT / "db-schema.png")


def render_sample():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    files = [dict(r) for r in con.execute(
        "select rel_path, kind, patient_id, window, slice_no, substr(sha256,1,12)||'…' sha256, status "
        "from files where kind='slice' limit 3")]
    ann = [dict(r) for r in con.execute(
        "select id, patient_id, slice_no, annotator, substr(source_sha256,1,12)||'…' src_sha "
        "from slice_annotations limit 3")]
    lab = [dict(r) for r in con.execute(
        "select annotation_id, label_code, value from slice_labels where value=1 limit 4")]
    pat = [dict(r) for r in con.execute("select patient_id, pseudonym from patients limit 3")]

    img = Image.new("RGB", (1200, 620), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((30, 16), "manifest.db - live rows", font=F_TITLE, fill=INK)

    def table(y, title, rows, widths):
        d.text((30, y), title, font=F_COL, fill=HEADER)
        y += 26
        cols = list(rows[0].keys())
        x = 30
        for c, w in zip(cols, widths):
            d.text((x, y), c, font=F_SMALL, fill=DIM)
            x += w
        y += 18
        for r in rows:
            x = 30
            for c, w in zip(cols, widths):
                v = str(r[c]) if r[c] is not None else "NULL"
                d.text((x, y), v[:38], font=F_COL, fill=INK)
                x += w
            y += 24
        return y + 22

    y = table(60, "files", files, [280, 70, 90, 70, 70, 130, 80])
    y = table(y, "slice_annotations", ann, [50, 90, 70, 200, 150])
    y = table(y, "slice_labels (only value=1 shown)", lab, [120, 160, 60])
    table(y, "patients - the ONLY table holding real<->pseudonym mapping", pat, [90, 160])
    img.save(OUT / "db-sample.png")


if __name__ == "__main__":
    counts = {}
    con = sqlite3.connect(DB)
    for (n,) in con.execute("select name from sqlite_master where type='table'"):
        counts[n] = con.execute(f"select count(*) from {n}").fetchone()[0]
    render_schema(counts)
    render_sample()
    print("wrote docs/images/db-schema.png and db-sample.png")
