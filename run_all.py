"""
Run the full experiment pipeline end-to-end.

Usage:
  python run_all.py                                    # all models, full dataset
  python run_all.py --models llava-next-video molmo2   # specific models
  python run_all.py --quick --models llava-next-video  # smoke test (5 per condition)
  python run_all.py --skip-gen --skip-layers           # skip steps already done
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ALL_MODELS = [
    "llava-next-video", "video-llama2", "qwen2.5-vl",
    "molmo2", "internvl2.5", "videollama3", "videochat-flash",
]
ALL_FPS = [8, 16, 30]


def run(cmd: list[str]):
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run([sys.executable] + cmd)
    if result.returncode != 0:
        print(f"ERROR: {cmd[0]} failed (exit {result.returncode})")
        sys.exit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models",  nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    p.add_argument("--n",       type=int, default=100, help="videos per condition")
    p.add_argument("--limit",   type=int, default=None, help="inference cap per interval")
    p.add_argument("--quick",   action="store_true", help="--n 5 --limit 5")
    p.add_argument("--skip-gen",     action="store_true")
    p.add_argument("--skip-probe",   action="store_true")
    p.add_argument("--skip-layers",  action="store_true")
    p.add_argument("--skip-analyze", action="store_true")
    p.add_argument("--skip-download", action="store_true", help="(ignored, kept for compat)")
    args = p.parse_args()

    if args.quick:
        args.n, args.limit = 5, 5

    limit_flag = ["--limit", str(args.limit)] if args.limit else []

    # 1. Generate stimuli
    if not args.skip_gen:
        if Path("data/videos/metadata.csv").exists() and Path("data/sim_videos/metadata.csv").exists():
            print("\n=== Generate stimuli (SKIPPED — data exists) ===")
        else:
            print("\n=== Generate stimuli ===")
            run(["generate.py", "--out", "data/videos", "--n", str(args.n), "--fps", *[str(f) for f in ALL_FPS]])
            run(["generate.py", "--out", "data/sim_videos", "--n", str(args.n), "--fps", "30", "--simultaneity"])

    # 2. Probe: E001, E002, E003, E004
    if not args.skip_probe:
        print("\n=== Probing tasks ===")
        for model in args.models:
            for task in ("order", "interval"):
                run(["probe.py", "--model", model, "--task", task,
                     "--meta", "data/videos/metadata.csv", "--fps", "30",
                     "--out", f"results/{model}_{task}.csv", *limit_flag])

            run(["probe.py", "--model", model, "--task", "simultaneity",
                 "--meta", "data/sim_videos/metadata.csv", *limit_flag])

            for fps in ALL_FPS:
                fps_out = Path(f"results/{model}_order_fps{fps}.csv")
                if fps == 30:
                    e001_out = Path(f"results/{model}_order.csv")
                    if e001_out.exists() and not fps_out.exists():
                        shutil.copy(e001_out, fps_out)
                        print(f"\n$ cp {e001_out} {fps_out}  (reuse E001 fps30)")
                        continue
                run(["probe.py", "--model", model, "--task", "order",
                     "--meta", "data/videos/metadata.csv", "--fps", str(fps), *limit_flag])

    # 3. Layer probes: E005
    if not args.skip_layers:
        print("\n=== Layer probing (E005) ===")
        for model in args.models:
            run(["layer_probe.py", "--model", model,
                 "--meta", "data/videos/metadata.csv", "--stage", "both", *limit_flag])

    # 4. Analysis + figures
    if not args.skip_analyze:
        print("\n=== Analyze ===")
        run(["analyze.py"])

    print("\nDone.  Figures -> figs/   Tables -> results/")


if __name__ == "__main__":
    main()
