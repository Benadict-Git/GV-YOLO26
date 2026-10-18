"""lxml helpers for editing the report's word/document.xml in place, preserving direct formatting."""
import copy
from pathlib import Path
import re

from lxml import etree

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"


class Doc:
    def __init__(self, path):
        self.path = path
        self.tree = etree.parse(str(path))
        self.body = self.tree.getroot().find(W + "body")

    def save(self, path=None):
        self.tree.write(str(path or self.path), xml_declaration=True, encoding="UTF-8", standalone=True)

    @staticmethod
    def text(e):
        return "".join(t.text or "" for t in e.iter(W + "t"))

    def find(self, prefix, kind="p", nth=0, contains=False):
        hits = [e for e in self.body if e.tag == W + kind and
                ((prefix in self.text(e)) if contains else self.text(e).startswith(prefix))]
        assert len(hits) > nth, f"not found: {prefix!r} ({len(hits)} hits)"
        return hits[nth]

    def table_after(self, prefix):
        e = self.find(prefix)
        while e is not None and e.tag != W + "tbl":
            e = e.getnext()
        assert e is not None, prefix
        return e

    # ---------- runs
    @staticmethod
    def _base_rpr(p, bold):
        for r in p.findall(W + "r"):
            rpr = r.find(W + "rPr")
            if r.find(W + "t") is None:
                continue
            is_b = rpr is not None and rpr.find(W + "b") is not None and rpr.find(W + "b").get(W + "val") != "false"
            if is_b == bold:
                return copy.deepcopy(rpr) if rpr is not None else etree.Element(W + "rPr")
        for r in p.findall(W + "r"):
            if r.find(W + "rPr") is not None:
                return copy.deepcopy(r.find(W + "rPr"))
        return etree.Element(W + "rPr")

    @staticmethod
    def _set_fmt(rpr, fmt):
        for tag in ("b", "bCs", "i", "iCs", "highlight"):
            for x in rpr.findall(W + tag):
                rpr.remove(x)
        add = [t for t, on in (("b", "b" in fmt), ("bCs", "b" in fmt), ("i", "i" in fmt), ("iCs", "i" in fmt)) if on]
        anchor = rpr.find(W + "rFonts")
        idx = list(rpr).index(anchor) + 1 if anchor is not None else 0
        for k, tag in enumerate(add):
            rpr.insert(idx + k, etree.Element(W + tag))
        return rpr

    def set_runs(self, p, segments):
        """Replace the text runs of paragraph p. segments: str or [(text, fmt)] with fmt in '', 'b', 'i', 'bi'."""
        if isinstance(segments, str):
            segments = [(segments, "")]
        base = self._base_rpr(p, bold=False)
        keep = [r for r in p.findall(W + "r") if r.find(W + "br") is not None and r.find(W + "t") is None]
        for r in p.findall(W + "r") + p.findall(W + "hyperlink"):
            p.remove(r)
        for txt, fmt in segments:
            r = etree.SubElement(p, W + "r")
            r.append(self._set_fmt(copy.deepcopy(base), fmt))
            t = etree.SubElement(r, W + "t")
            t.text = txt
            t.set(XML_SPACE, "preserve")
        for r in keep:
            p.append(r)
        return p

    def set_par(self, prefix, segments, **kw):
        return self.set_runs(self.find(prefix, **kw), segments)

    def insert_after(self, anchor, segments, template=None):
        new = copy.deepcopy(template if template is not None else anchor)
        for r in new.findall(W + "r"):
            if r.find(W + "br") is not None and r.find(W + "t") is None:
                new.remove(r)
        anchor.addnext(new)
        return self.set_runs(new, segments)

    def remove(self, e):
        e.getparent().remove(e)

    def caption(self, prefix, label, title):
        return self.set_runs(self.find(prefix), [(label + " ", "b"), (title, "")])

    # ---------- tables
    @staticmethod
    def rows(tbl):
        return tbl.findall(W + "tr")

    def set_cell(self, tbl, r, c, value, fmt=None):
        tc = self.rows(tbl)[r].findall(W + "tc")[c]
        ps = tc.findall(W + "p")
        for extra in ps[1:]:
            tc.remove(extra)
        p = ps[0]
        if p.find(W + "r") is None:
            src = self.rows(tbl)[r].findall(W + "tc")[0].find(W + "p").find(W + "r")
            if src is not None:
                p.append(copy.deepcopy(src))
        if fmt is None:
            rr = p.find(W + "r")
            rpr = rr.find(W + "rPr") if rr is not None else None
            b = rpr.find(W + "b") if rpr is not None else None
            fmt = "b" if b is not None and b.get(W + "val") != "false" else ""
        self.set_runs(p, [(str(value), fmt)])

    def set_row(self, tbl, r, values, fmt=None):
        for c, v in enumerate(values):
            if v is not None:
                self.set_cell(tbl, r, c, v, fmt)

    def add_row(self, tbl, values, clone=-1, after=None, fmt=None):
        rows = self.rows(tbl)
        new = copy.deepcopy(rows[clone])
        (rows[after] if after is not None else rows[-1]).addnext(new)
        self.set_row(tbl, self.rows(tbl).index(new), values, fmt)

    def table_text(self, tbl):
        return [[self.text(tc) for tc in tr.findall(W + "tc")] for tr in self.rows(tbl)]


def replace_everywhere(doc, old, new):
    n = 0
    for t in doc.tree.getroot().iter(W + "t"):
        if t.text and old in t.text:
            t.text = t.text.replace(old, new)
            n += 1
    return n


def set_code(tbl, lines):
    """Replace the lines of a one-cell code-listing table."""
    tc = tbl.find(f"{W}tr/{W}tc")
    ps = tc.findall(W + "p")
    tmpl = copy.deepcopy(ps[0])
    for p in ps:
        tc.remove(p)
    for line in lines:
        p = copy.deepcopy(tmpl)
        ts = list(p.iter(W + "t"))
        ts[0].text = line
        ts[0].set(XML_SPACE, "preserve")
        for t in ts[1:]:
            t.getparent().remove(t)
        tc.append(p)


def set_tab_texts(p, parts):
    """Set the w:t nodes of a tab-separated list entry (TOC / LOF / LOT) in order."""
    ts = list(p.iter(W + "t"))
    assert len(ts) == len(parts), (len(ts), parts)
    for t, v in zip(ts, parts):
        t.text = v
        t.set(XML_SPACE, "preserve")
    return p


def clone_after(anchor, parts=None):
    new = copy.deepcopy(anchor)
    anchor.addnext(new)
    if parts is not None:
        set_tab_texts(new, parts)
    return new


def add_column(doc, tbl, header, values):
    """Append a column by cloning the last cell of every row; rescale the grid so the table width is unchanged."""
    grid = tbl.find(W + "tblGrid")
    cols = grid.findall(W + "gridCol")
    total = sum(int(c.get(W + "w")) for c in cols)
    n = len(cols) + 1
    widths = [total // n] * n
    for c, w in zip(cols, widths):
        c.set(W + "w", str(w))
    nc = copy.deepcopy(cols[-1])
    nc.set(W + "w", str(widths[-1]))
    grid.append(nc)
    for r, (tr, v) in enumerate(zip(doc.rows(tbl), [header] + list(values))):
        tcs = tr.findall(W + "tc")
        tcs[-1].addnext(copy.deepcopy(tcs[-1]))
        for i, tc in enumerate(tr.findall(W + "tc")):
            tcw = tc.find(f"{W}tcPr/{W}tcW")
            if tcw is not None:
                tcw.set(W + "w", str(widths[i]))
        doc.set_cell(tbl, r, len(tcs), v)


def renumber_citations(doc, mapping):
    pat = re.compile(r"\[(\d+)\]")
    for t in doc.tree.getroot().iter(W + "t"):
        if t.text and "[" in t.text:
            t.text = pat.sub(lambda m: f"[{mapping.get(int(m.group(1)), int(m.group(1)))}]", t.text)


def set_equation(p, expr, number):
    ts = list(p.iter(W + "t"))
    ts[0].text, ts[-1].text = expr, number


def merge_runs(work_dir):
    """Merge consecutive text runs with identical formatting across all xml files in word/."""
    work_path = Path(work_dir)
    for xml_file in (work_path / "word").glob("*.xml"):
        tree = etree.parse(str(xml_file))
        root = tree.getroot()
        changed = False
        for p in root.iter(W + "p"):
            children = list(p)
            i = 0
            while i < len(children) - 1:
                r1 = children[i]
                r2 = children[i + 1]
                if r1.tag == W + "r" and r2.tag == W + "r":
                    non_text1 = [c for c in r1 if c.tag not in (W + "rPr", W + "t")]
                    non_text2 = [c for c in r2 if c.tag not in (W + "rPr", W + "t")]
                    if not non_text1 and not non_text2:
                        rpr1 = r1.find(W + "rPr")
                        rpr2 = r2.find(W + "rPr")
                        rpr1_str = etree.tostring(rpr1) if rpr1 is not None else b""
                        rpr2_str = etree.tostring(rpr2) if rpr2 is not None else b""
                        if rpr1_str == rpr2_str:
                            t1 = r1.find(W + "t")
                            t2 = r2.find(W + "t")
                            if t1 is not None and t2 is not None:
                                t1.text = (t1.text or "") + (t2.text or "")
                                t1.set(XML_SPACE, "preserve")
                            elif t1 is None and t2 is not None:
                                r1.append(t2)
                            p.remove(r2)
                            children.pop(i + 1)
                            changed = True
                            continue
                i += 1
        if changed:
            tree.write(str(xml_file), xml_declaration=True, encoding="UTF-8", standalone=True)
