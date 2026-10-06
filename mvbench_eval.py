"""Harness validation on MVBench (plan + amendment v2: reviews/mainpush_1004/plan_mvbench.md).

Paths (each run once, greedy):
  vcf-official  VideoChat-Flash model.chat with the official media_dict (bounds; 'img' for TVQA frame folders),
                max_num_frames=512, max_new_tokens=16, the official MVBench prompt; all 19 tasks.
  vcf-adapter   our adapter's exact call: max_num_frames=8, no media_dict (so no bounds); unbounded tasks only.
  vl2-official  VideoLLaMA2 processor with bounds (8 frames) + mm_infer, the official MVBench prompt; on unbounded
                items this is exactly our adapter's call.
Parser: the official VideoLLaMA2 rule (first option-letter pattern, else option text), without its "pred = C"
fallback; strict-letter accuracy is reported too. Decoded frame counts are logged per item for VideoChat-Flash.
Resumable (appends per item). Items from mvbench_prep.py (19 tasks; NTU pose is not distributed).

Usage: python mvbench_eval.py --path vcf-official --out results_mvbench/vcf_official.csv
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from models import load_model

VCF_PROMPT = "Question: {q}\nOption:\n{opts}\nOnly give the best option."
VL2_PROMPT = "Question: {q}\nOptions:\n{opts}Answer with the option's letter from the given choices directly and only give the best option."


def options(cands):
    return "".join(f"({'ABCDE'[i]}) {c}\n" for i, c in enumerate(cands))


def parse(output, cands):
    """Official VideoLLaMA2 MVBench rule without its fixed fallback; returns (letter or None, strict letter or None)."""
    letters = "ABCDE"[:len(cands)]
    out = output.replace("answer", "").replace("Answer", "")
    found = re.findall(f"[\\(,\\ ]*[{letters[0]}-{letters[-1]}][\\),\\ ]*", out)
    strict = found[0].strip().strip("()") if found else None
    if strict in letters:
        return strict, strict
    for i, c in enumerate(cands):
        if c.lower() in output.lower():
            return letters[i], None
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, choices=["vcf-official", "vcf-adapter", "vl2-official"])
    ap.add_argument("--items", default="data/mvbench/items.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="items per task (smoke test)")
    a = ap.parse_args()
    df = pd.read_csv(a.items, keep_default_na=False)
    if a.path == "vcf-adapter":
        df = df[df.start.astype(str) == ""]
    if a.limit:
        df = df.groupby("task", group_keys=False).head(a.limit)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists() and out.stat().st_size > 0:
        done = {(r["task"], r["idx"]) for r in csv.DictReader(open(out)) if not r["response"].startswith("ERROR")}

    if a.path.startswith("vcf"):
        model = load_model("videochat-flash")
        m, tok = model._model, model._tok
        mod = sys.modules[type(m).__module__]
        orig, nfr = mod.load_video, []

        def counting(*args, **kw):                      # record how many frames the loader decoded
            frames, msg = orig(*args, **kw)
            nfr.append(len(frames))
            return frames, msg
        mod.load_video = counting
    else:
        model = load_model("video-llama2")
        from videollama2 import mm_infer

    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["task", "idx", "answer", "pred", "strict", "correct", "correct_strict",
                                           "n_frames", "response"])
        if not out.exists() or out.stat().st_size == 0:
            w.writeheader()
        for r in tqdm(list(df.itertuples()), mininterval=30):
            if (r.task, str(r.idx)) in done:
                continue
            cands = json.loads(r.candidates)
            bound = str(r.start) != ""
            nf = ""
            try:
                if a.path == "vcf-official":
                    md = {"video_read_type": "img" if r.type == "frame" else "decord"}
                    if bound:
                        md.update(start=float(r.start), end=float(r.end))
                    nfr.clear()
                    resp = m.chat(r.video, tok, VCF_PROMPT.format(q=r.question, opts=options(cands).rstrip("\n") + "\n"),
                                  return_history=False, max_num_frames=512, media_dict=md,
                                  generation_config=dict(max_new_tokens=16, do_sample=False))
                    nf = nfr[-1] if nfr else ""
                elif a.path == "vcf-adapter":
                    nfr.clear()
                    resp = m.chat(r.video, tok, VCF_PROMPT.format(q=r.question, opts=options(cands).rstrip("\n") + "\n"),
                                  return_history=False, max_num_frames=model.n,
                                  generation_config=dict(max_new_tokens=64, do_sample=False))
                    nf = nfr[-1] if nfr else ""
                else:
                    s, e = (float(r.start), float(r.end)) if bound else (None, None)
                    vt = model._proc["video"](r.video, s=s, e=e)
                    resp = mm_infer(vt, VL2_PROMPT.format(q=r.question, opts=options(cands)), model=model._model,
                                    tokenizer=model._tok, modal="video", do_sample=False)
            except Exception as ex:  # noqa: BLE001 — record; a rerun retries ERROR rows
                resp = f"ERROR {type(ex).__name__}: {ex}"[:300]
            p, st = (None, None) if resp.startswith("ERROR") else parse(resp, cands)
            w.writerow(dict(task=r.task, idx=r.idx, answer=r.answer, pred=p or "", strict=st or "",
                            correct=int(p == r.answer), correct_strict=int(st == r.answer), n_frames=nf,
                            response=resp.replace("\n", " ")[:200]))
            fh.flush()

    res = pd.read_csv(out, keep_default_na=False)
    res = res[~res.response.astype(str).str.startswith("ERROR")].drop_duplicates(["task", "idx"], keep="last")
    per = res.groupby("task")[["correct", "correct_strict"]].mean()
    print((100 * per).round(1).to_string())
    print(f"MVB_RESULT path={a.path} tasks={len(per)} items={len(res)} acc={100 * per.correct.mean():.2f} "
          f"strict={100 * per.correct_strict.mean():.2f} parsed={(res.pred != '').mean():.3f}")


if __name__ == "__main__":
    main()
