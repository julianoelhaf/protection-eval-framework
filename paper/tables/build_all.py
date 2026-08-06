"""Regenerate every paper table from committed evidence.

Runs every ``table_*.py`` generator in this directory (each writes one
``table__<name>.tex``) and prints a provenance summary, including a WARNINGS
section listing exactly which cells are NOT committed-backed (legacy_values).

    python paper/tables/build_all.py

Each generator is also runnable on its own, e.g.:

    python paper/tables/table_reference_20ms.py
"""
import importlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sources as S  # noqa: E402


def main():
    mods = sorted(p.stem for p in HERE.glob("table_*.py"))
    print("Regenerating %d paper tables from committed evidence in reports/ ...\n" % len(mods))
    S.WARNINGS.clear()
    ok = 0
    for name in mods:
        mod = importlib.import_module(name)
        if not hasattr(mod, "build"):
            continue
        try:
            mod.build()
            ok += 1
        except Exception as exc:  # noqa: BLE001
            print("  FAILED %s: %s" % (name, exc))
    print("\n%d/%d tables written to %s/" % (ok, len(mods), HERE.name))

    if S.WARNINGS:
        print("\nWARNINGS -- cells NOT backed by committed evidence (see legacy_values.py):")
        for w in S.WARNINGS:
            print("  * " + w)
    else:
        print("\nAll cells committed-backed.")


if __name__ == "__main__":
    main()
