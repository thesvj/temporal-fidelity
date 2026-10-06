"""Repair `order_rev` labels written before the fix in exp_g3.py.

"Did N2 move AFTER N1?" is true exactly when "did N1 move BEFORE N2?" is true, so order_rev carries the
same label as order_real. The first version of exp_g3.py stored 1 - gt, which makes a perfectly consistent
model look below chance (Molmo2: .243 instead of .757). Only the `gt` column is wrong; the responses and
margins are untouched, so the fix is a relabel, not a re-run.

Idempotent: it rewrites gt from the matching order_real row rather than flipping, so running it twice is
harmless. Files written after the exp_g3.py fix are left unchanged.

Usage: python repair_rev_gt.py results_v3/*_expg3.csv
"""
import csv
import shutil
import sys
from pathlib import Path


def repair(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return f"{path}: empty"
    fields = list(rows[0].keys())
    truth = {r["path"]: r["gt"] for r in rows if r["item"] == "order_real"}
    changed = 0
    for r in rows:
        if r["item"] == "order_rev" and r["path"] in truth and r["gt"] != truth[r["path"]]:
            r["gt"] = truth[r["path"]]
            changed += 1
    if not changed:
        return f"{Path(path).name}: already correct"
    shutil.copy(path, str(path) + ".prerepair")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return f"{Path(path).name}: relabelled {changed} order_rev rows (backup .prerepair)"


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(repair(p))
