"""Set the static TOC / LOF / LOT page numbers from the rendered PDF, repeating until they are stable.

Usage: python fix_pages.py report.docx   (also leaves report.pdf next to it)
"""
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from docxlib import Doc, W

DOCX = Path(sys.argv[1]).resolve()
WORK = DOCX.parent / "pages_work"


def render():
    pdf = DOCX.with_suffix(".pdf")
    pdf.unlink(missing_ok=True)
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(DOCX.parent), str(DOCX)],
                   check=True, capture_output=True)
    info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
    pages = []
    for i in range(1, int(re.search(r"Pages:\s+(\d+)", info).group(1)) + 1):
        t = subprocess.run(["pdftotext", "-layout", "-f", str(i), "-l", str(i), str(pdf), "-"],
                           capture_output=True, text=True).stdout
        lines = [re.sub(r"\s+", " ", l).strip() for l in t.splitlines() if l.strip()]
        label = lines[-1] if lines and re.fullmatch(r"[0-9]+|[ivxlc]+", lines[-1]) else None
        pages.append((label, lines))
    return pages


def norm(s):
    return re.sub(r"\s+", " ", s.replace("​", "")).strip()


def entries(doc):
    out = []
    for p in doc.body.iter(W + "p"):
        ts = list(p.iter(W + "t"))
        if len(ts) >= 2 and list(p.iter(W + "tab")) and re.fullmatch(r"[0-9]+|[ivxlc]+", (ts[-1].text or "").strip()):
            out.append((p, ts[-1], norm(" ".join(t.text or "" for t in ts[:-1]))))
    return out


def locate(title, pages, start):
    for label, lines in pages[start:]:
        for l in lines:
            if l == title or l.startswith(title) or (len(title) > 40 and l.startswith(title[:40])):
                return label
    return None


for it in range(5):
    shutil.rmtree(WORK, ignore_errors=True)
    with zipfile.ZipFile(DOCX) as z:
        z.extractall(WORK)
    doc = Doc(WORK / "word/document.xml")
    pages = render()
    first_body = next(i for i, (lab, _) in enumerate(pages) if lab == "1")
    changed, missing = 0, []
    for p, tnode, title in entries(doc):
        t = re.sub(r"^((?:Fig\.|Table) \d+\.\d+)\s*", r"\1 ", title)
        front = t.split()[0] in ("ACKNOWLEDGMENT", "ABSTRACT", "LIST")
        lab = locate(t, pages, 0 if front else first_body)
        if lab is None:
            missing.append(title)
            continue
        if (tnode.text or "").strip() != lab:
            tnode.text = lab
            changed += 1
    doc.save()
    DOCX.unlink()
    with zipfile.ZipFile(DOCX, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(WORK.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(WORK).as_posix())
    print(f"pass {it}: {changed} page numbers changed; not found: {missing}")
    if changed == 0:
        break
render()
shutil.rmtree(WORK, ignore_errors=True)
