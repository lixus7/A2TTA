#!/usr/bin/env python3
"""Render the markdown result tables to clean, zoomable **vector PDF** (one page
per sub-table) + a PNG of each page. No LaTeX needed. Bold (**x**) cells are
drawn bold, underline (_x_) italic; the leading '#'/'##' headings become page
titles.

Usage:
  python render_result_tables.py \
      ../tables/main_table.md ../tables/ablation.md ../tables/tsas_new_sensors.md
Outputs (next to each .md): <stem>.pdf (multipage) + <stem>__p1.png (first page).
"""
import os.path as osp
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

CELL_FS_BASE = 6.5          # shrinks automatically with column count
TITLE_FS = 11


def _strip(cell):
    """Return (text, bold, italic) from a markdown cell."""
    c = cell.strip()
    bold = c.startswith("**") and c.endswith("**")
    if bold:
        c = c[2:-2].strip()
    ital = (c.startswith("_") and c.endswith("_")) or (c.startswith("*") and c.endswith("*"))
    if ital:
        c = c.strip("_*").strip()
    return c, bold, ital


def parse_tables(md_path):
    """-> list of (title, header:list, rows:list[list]) for each md table."""
    lines = open(md_path).read().split("\n")
    tables, title = [], osp.basename(md_path)
    i = 0
    while i < len(lines):
        ln = lines[i]
        h = re.match(r"^#{1,6}\s+(.*)", ln)
        if h:
            title = h.group(1).strip()
        if ln.strip().startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i]); i += 1
            if len(block) >= 2:
                def cells(row):
                    parts = row.split("|")[1:-1]
                    return [p for p in parts]
                header = [c.strip() for c in cells(block[0])]
                rows = [cells(r) for r in block[2:]]          # skip separator
                tables.append((title, header, rows))
            continue
        i += 1
    return tables


def render(md_path):
    tables = parse_tables(md_path)
    if not tables:
        print(f"[tables] no tables in {md_path}"); return
    stem = md_path[:-3] if md_path.endswith(".md") else md_path
    pdf_path = stem + ".pdf"
    plt.rcParams.update({"savefig.facecolor": "white", "figure.facecolor": "white"})
    with PdfPages(pdf_path) as pdf:
        for pi, (title, header, rows) in enumerate(tables):
            ncol = len(header)
            nrow = len(rows) + 1
            fs = max(4.0, CELL_FS_BASE * min(1.0, 30.0 / max(ncol, 1)))
            # landscape, width scales with columns
            fig_w = max(8.0, 0.42 * ncol)
            fig_h = max(2.5, 0.30 * nrow + 0.6)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            ax.axis("off")
            ax.set_title(title, fontsize=TITLE_FS, fontweight="bold", loc="left", pad=10)

            cell_text, cell_bold, cell_ital = [], [], []
            hdr_text = [_strip(h)[0] for h in header]
            for r in rows:
                tr, br, ir = [], [], []
                # pad/truncate to header length
                r = (r + [""] * ncol)[:ncol]
                for c in r:
                    t, b, it = _strip(c)
                    tr.append(t); br.append(b); ir.append(it)
                cell_text.append(tr); cell_bold.append(br); cell_ital.append(ir)

            tab = ax.table(cellText=cell_text, colLabels=hdr_text,
                           loc="center", cellLoc="center")
            tab.auto_set_font_size(False)
            tab.set_fontsize(fs)
            tab.scale(1, 1.25)
            # style: header bold + shaded; bold/italic data cells
            for (r, c), cell in tab.get_celld().items():
                cell.set_linewidth(0.3)
                if r == 0:
                    cell.set_text_props(fontweight="bold")
                    cell.set_facecolor("#efefef")
                else:
                    if cell_bold[r - 1][c]:
                        cell.set_text_props(fontweight="bold")
                        cell.set_facecolor("#eaf3ff")
                    elif cell_ital[r - 1][c]:
                        cell.set_text_props(fontstyle="italic")
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            if pi == 0:
                fig.savefig(stem + "__p1.png", dpi=200, bbox_inches="tight")
            plt.close(fig)
    print(f"[tables] {osp.basename(md_path)} -> {osp.basename(pdf_path)} "
          f"({len(tables)} pages)")


def main():
    paths = sys.argv[1:] or [
        "../tables/main_table.md", "../tables/ablation.md",
        "../tables/tsas_new_sensors.md"]
    here = osp.dirname(osp.abspath(__file__))
    for p in paths:
        render(p if osp.isabs(p) else osp.join(here, p))


if __name__ == "__main__":
    main()
