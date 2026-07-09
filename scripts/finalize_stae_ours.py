#!/usr/bin/env python3
"""Finalize STAE-Ours integration:
 1. main_table.md  -> recompute best/second over all non-FM columns (incl the
    already-inserted STAE-Ours col) and rewrite **bold**/_underline_.
 2. main_table_part{1,2,3}.tex -> insert a STAE-Ours column right after STAE
    (colspec, group/sub headers, \\multicolumn dataset titles, every data row)
    with \\best{}/\\second{} recomputed from the SAME md numbers.

Highlightable = all method columns EXCEPT the 8 foundation-model cols
(Chronos2-U/M, TimesFM ZS/FT, MoE ZS/FT, Moirai2 ZS/FT) = md/tex indices 18..25.
"""
import os, re

MD = "../tables/main_table.md"
TEX = ["../tables/main_table_part1.tex",
       "../tables/main_table_part2.tex",
       "../tables/main_table_part3.tex"]
METS = ["MAE", "RMSE", "MAPE"]
HZ = ["3", "6", "12", "Avg"]
FM_IDX = set(range(18, 26))   # 8 foundation-model columns (0-based among methods)
NMETH = 29                    # after STAE-Ours inserted: 0..28


def parse_mean(cell):
    c = cell.replace("**", "").replace("_", "").strip()
    if c in ("", "--", "—"):
        return None
    m = re.match(r"[-+]?\d+\.?\d*", c)
    return float(m.group()) if m else None


def best_second(means):
    cand = [(i, means[i]) for i in range(NMETH) if i not in FM_IDX and means[i] is not None]
    cand.sort(key=lambda t: t[1])
    b = cand[0][0] if cand else None
    s = cand[1][0] if len(cand) > 1 else None
    return b, s


def md_to_tex_val(raw):
    """'11.74 ± 0.08' -> '11.74$_{\\pm0.08}$' ; '12.67' -> '12.67'."""
    raw = raw.strip()
    if "±" in raw:
        mu, sd = [x.strip() for x in raw.split("±")]
        return f"{mu}$_{{\\pm{sd}}}$"
    return raw


def do_md():
    lines = open(MD).read().split("\n")
    out = []
    ds = met = None
    lookup = {}   # (ds,met,hz) -> (best_idx, second_idx)
    ours = {}     # (ds,met,hz) -> raw STAE-Ours value (md text)
    for ln in lines:
        mh = re.match(r"^##\s+(\S+)", ln)
        if mh:
            ds = mh.group(1); met = None
        if ln.startswith("|") and "±" in ln and ds and re.match(r"^(PEMS|TFNSW)", ds):
            cells = ln.split("|")
            if len(cells) >= 3 + NMETH + 1:           # leading '' + Metric + Hz + 29 + trailing ''
                c0 = cells[1].strip().strip("*").strip()
                c0 = c0.split()[0] if c0 else c0
                if c0 in METS:
                    met = c0
                hz = cells[2].strip()
                if met and hz in HZ:
                    meth = cells[3:3 + NMETH]
                    means = [parse_mean(x) for x in meth]
                    b, s = best_second(means)
                    lookup[(ds, met, hz)] = (b, s)
                    ours[(ds, met, hz)] = meth[27].replace("**", "").replace("_", "").strip()
                    new = cells[:]
                    for i in range(NMETH):
                        raw = meth[i].replace("**", "").replace("_", "").strip()
                        if i in FM_IDX or raw in ("", "--", "—"):
                            new[3 + i] = f" {raw} " if raw else meth[i]
                        elif i == b:
                            new[3 + i] = f" **{raw}** "
                        elif i == s:
                            new[3 + i] = f" _{raw}_ "
                        else:
                            new[3 + i] = f" {raw} "
                    out.append("|".join(new))
                    continue
        out.append(ln)
    open(MD, "w").write("\n".join(out))
    return lookup, ours


def do_tex(path, lookup, ours):
    txt = open(path).read()
    # 1) colspec 30 -> 31 c
    txt = re.sub(r"(\\begin\{tabular\}\{)(c{30})(\})", lambda m: m.group(1) + "c" * 31 + m.group(3), txt)
    # 2) group header: Ours spans 2
    txt = txt.replace("& \\textbf{STAEFormer} & \\textbf{Ours} \\\\",
                      "& \\textbf{STAEFormer} & \\multicolumn{2}{c}{\\textbf{Ours}} \\\\")
    # 3) sub-header
    txt = txt.replace("& \\textbf{STAE} & \\textbf{A2TTA} \\\\",
                      "& \\textbf{STAE} & \\textbf{STAE-Ours} & \\textbf{A2TTA} \\\\")
    # 4) dataset title multicolumn 30 -> 31
    txt = txt.replace("\\multicolumn{30}{c}{", "\\multicolumn{31}{c}{")

    lines = txt.split("\n")
    out = []
    ds = met = None
    # brace-aware: inner may contain one nested {...} group (e.g. $_{\pm0.02}$)
    strip_hl = lambda c: re.sub(r"\\(?:best|second)\{((?:[^{}]|\{[^{}]*\})*)\}", r"\1", c)
    for ln in lines:
        md = re.search(r"\\multicolumn\{\d+\}\{c\}\{\\textbf\{(PEMS\d+|TFNSW)\}\}", ln)
        if md:
            ds = md.group(1); met = None
        m = re.match(r"^(.*?)(\\\\\s*)$", ln)
        is_data = False
        if m and ds and "&" in ln:
            body, tail = m.group(1), m.group(2)
            parts = body.split("&")
            if len(parts) == 2 + 28:                       # metric + Len + 28 value cells
                hz = parts[1].strip()
                mm = re.search(r"\\multirow\{4\}\{\*\}\{(MAE|RMSE|MAPE)\}", parts[0])
                if mm:
                    met = mm.group(1)
                if hz in HZ and met and (ds, met, hz) in lookup:
                    is_data = True
                    # insert STAE-Ours after STAE (parts[28]); A2TTA is parts[29]
                    so_raw = ours.get((ds, met, hz), "")
                    so_cell = f" {md_to_tex_val(so_raw)} "
                    parts = parts[:29] + [so_cell] + parts[29:]   # now 2+29
                    b, s = lookup[(ds, met, hz)]
                    for i in range(NMETH):
                        idx = 2 + i
                        inner = strip_hl(parts[idx]).strip()
                        if i in FM_IDX:
                            parts[idx] = f" {inner} "
                        elif i == b:
                            parts[idx] = f" \\best{{{inner}}} "
                        elif i == s:
                            parts[idx] = f" \\second{{{inner}}} "
                        else:
                            parts[idx] = f" {inner} "
                    out.append("&".join(parts) + tail)
        if not is_data:
            out.append(ln)
    open(path, "w").write("\n".join(out))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.dirname(here))   # eac/
    import shutil, glob
    # backups
    for f in [MD] + TEX:
        shutil.copy(f, f + ".bak_finalstae")
    lookup, ours = do_md()
    for t in TEX:
        do_tex(t, lookup, ours)
    print("MD rows highlighted:", len(lookup))
    print("done. backups *.bak_finalstae")


if __name__ == "__main__":
    main()
