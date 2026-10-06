"""Judge fallback for TempCompass multi-choice, as on the TempCompass leaderboard.

The leaderboard scores an answer by rule first and sends unmatched answers to an LLM judge (gpt-3.5-turbo-1106) that replies
Correct/Incorrect. tc_official.py scores unmatched answers 0. Here unmatched answers go to a local judge instead
(Qwen2.5-7B-Instruct, text only, greedy; instruction paraphrased from the TempCompass judge, not copied), so a model that answers in sentences is scored as the leaderboard scores it.

    .venv-qwen2.5-vl/bin/python tc_judge.py results_tc_official/llava-next-video_f32.csv [...]
writes <csv>.judged.csv (adds `judge`, `correct_judged`) and prints match rate, strict, matched-only and judged accuracy.
"""
import csv
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

JUDGE = "Qwen/Qwen2.5-7B-Instruct"
PROMPT = ("You will receive a multi-choice question, the ground-truth answer and the prediction from a question answering (QA) "
          "model. Your task is to determine whether QA model prediction is correct, based on the question and ground-truth "
          "answer. If the prediction is correct, respond \"Correct\". If the prediction is incorrect, respond \"Incorrect\".\n\n"
          "Multi-Choice Question:\n{q}\nGround-Truth Answer: {a}\nModel Prediction: {p}")


def main():
    tok = AutoTokenizer.from_pretrained(JUDGE)
    lm = AutoModelForCausalLM.from_pretrained(JUDGE, torch_dtype=torch.bfloat16, device_map="auto")
    for path in sys.argv[1:]:
        rows = list(csv.DictReader(open(path)))
        for r in rows:
            if r["matched"] == "1":
                r["judge"], r["correct_judged"] = "", r["correct"]
                continue
            msg = [{"role": "user", "content": PROMPT.format(q=r["question"], a=r["answer"], p=r["response"])}]
            ids = tok.apply_chat_template(msg, add_generation_prompt=True, return_tensors="pt").to(lm.device)
            with torch.no_grad():
                out = lm.generate(ids, max_new_tokens=4, do_sample=False)
            v = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
            r["judge"] = v
            r["correct_judged"] = "1" if v.lower().startswith("correct") else "0"
        with open(path.replace(".csv", ".judged.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        n = len(rows); m = [r for r in rows if r["matched"] == "1"]
        print(f"{path}: n={n} match={len(m)/n:.3f} strict={100*sum(r['correct']=='1' for r in rows)/n:.2f} "
              f"matched_acc={100*sum(r['correct']=='1' for r in m)/max(1,len(m)):.2f} "
              f"judged={100*sum(r['correct_judged']=='1' for r in rows)/n:.2f}", flush=True)


if __name__ == "__main__":
    main()
