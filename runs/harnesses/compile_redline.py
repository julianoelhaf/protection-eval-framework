#!/usr/bin/env python3
"""Compile the redline manuscript (marked-up + clean variants) with tectonic and summarize.

The cluster `texlive` conda env is an incomplete install (missing latex.ltx); the self-contained
`tectonic` engine (env `tex`) is used instead. tectonic cannot run inkscape, so the 5 cosmetic
\\includesvg icons are shimmed to placeholder rules (TEST-ONLY; the real figure renders on Overleaf).
Builds in a fresh scratch copy of paper/ (no repo pollution); parses each .log for TeX errors,
undefined references/citations, and multiply-defined labels. Stdlib only.
"""
import argparse, os, re, shutil, subprocess

SVG_SHIM = (r'\usepackage{svg}' + '\n'
            r'\RenewDocumentCommand{\includesvg}{om}{\rule{5mm}{5mm}}'
            r'  % TEST-ONLY shim: tectonic has no inkscape; real icons render on Overleaf')

def run_tectonic(build_dir, texname, cache):
    env = dict(os.environ, TECTONIC_CACHE_DIR=cache)
    cmd = ["tectonic", "--keep-logs", "--chatter", "minimal", texname]
    return subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True, env=env)

def parse_log(logpath):
    if not os.path.exists(logpath):
        return {"log_exists": False}
    txt = open(logpath, errors="replace").read()
    errors = re.findall(r'(?m)^! (.+)$', txt)
    undef_ref = sorted(set(re.findall(r"Reference `([^']+)' on page [^ ]+ undefined", txt))
                       | set(re.findall(r"Reference `([^']+)' undefined", txt)))
    undef_cite = sorted(set(re.findall(r"Citation `([^']+)' on page [^ ]+ undefined", txt)))
    dup_labels = sorted(set(re.findall(r"Label `([^']+)' multiply defined", txt)))
    overfull = len(re.findall(r'Overfull \\hbox', txt))
    missing_char = len(re.findall(r'Missing character', txt))
    m = re.search(r'Output written on \S+ \((\d+) pages?', txt)
    pages = int(m.group(1)) if m else None
    return {"log_exists": True, "errors": errors, "undef_ref": undef_ref, "undef_cite": undef_cite,
            "dup_labels": dup_labels, "overfull": overfull, "missing_char": missing_char,
            "pages": pages, "raw": txt}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--tex", required=True)
    ap.add_argument("--outroot", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--cache", required=True)
    a = ap.parse_args()
    os.makedirs(a.outroot, exist_ok=True)
    lines = []
    def log(s): print(s, flush=True); lines.append(s)
    log(f"tectonic compile — cache={a.cache}")

    base = a.tex[:-4]
    variants = [("markup", None),
                ("clean", (r"\markuptrue   % <<<", r"\markupfalse  % <<<"))]
    for name, repl in variants:
        bdir = os.path.join(a.outroot, name)
        if os.path.exists(bdir): shutil.rmtree(bdir)
        shutil.copytree(a.src_dir, bdir)
        texpath = os.path.join(bdir, a.tex)
        t = open(texpath).read()
        if SVG_SHIM.splitlines()[1] not in t:            # idempotent shim injection
            t = t.replace(r'\usepackage{svg}', SVG_SHIM, 1)
        if repl:
            t2 = t.replace(repl[0], repl[1], 1)
            if t2 == t:
                log(f"[{name}] WARNING: markup toggle not found; variant == markup")
            t = t2
        open(texpath, "w").write(t)

        p = run_tectonic(bdir, a.tex, a.cache)
        info = parse_log(os.path.join(bdir, base + ".log"))
        pdf = os.path.join(bdir, base + ".pdf")
        ok = os.path.exists(pdf)
        log(f"\n===== VARIANT: {name} =====")
        log(f"tectonic returncode : {p.returncode}")
        log(f"PDF produced        : {ok}   pages: {info.get('pages')}")
        log(f"TeX errors (!)      : {len(info.get('errors', []))}")
        for e in info.get('errors', [])[:25]:
            log(f"      ! {e}")
        ur, uc, dl = info.get('undef_ref', []), info.get('undef_cite', []), info.get('dup_labels', [])
        log(f"undefined refs      : {len(ur)}  {ur[:30]}")
        log(f"undefined citations : {len(uc)}  {uc[:30]}")
        log(f"multiply-def labels : {len(dl)}  {dl}")
        log(f"overfull hboxes     : {info.get('overfull')}   missing-char warns: {info.get('missing_char')}")
        if ok:
            dest = os.path.join(a.outroot, f"{base}_{name}.pdf")
            shutil.copy(pdf, dest)
            log(f"PDF -> {dest}")
        else:
            log("---- tectonic stderr tail ----")
            for ln in (p.stderr or "").splitlines()[-30:]:
                log("      " + ln)
            if info.get("log_exists"):
                log("---- .log tail ----")
                for ln in info["raw"].splitlines()[-25:]:
                    log("      " + ln)

    with open(a.results, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSUMMARY -> {a.results}", flush=True)

if __name__ == "__main__":
    main()
