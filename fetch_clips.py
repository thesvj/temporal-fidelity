"""Find and download openly licensed source clips from Wikimedia Commons for the real-clip run.

Commons is used because every file carries a machine-readable licence and author, and the API needs no
key. Only licences that allow redistribution of modified clips are kept (public domain, CC0, CC BY,
CC BY-SA). Nothing is selected here: this lists candidates and downloads them; which clips are used,
and the onset frame of each, is decided by screening and recorded in clips.csv.

  python fetch_clips.py search   --out data/real_src            # writes candidates.csv
  python fetch_clips.py download --out data/real_src [--max 200]  # 480p derivative of every candidate
"""
import argparse
import csv
import json
import re
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://commons.wikimedia.org/w/api.php"
UA = "TemporalFidelityResearch/0.1 (academic; set your contact e-mail here)"
OK_LICENCE = re.compile(r"^(cc0|public domain|pd\b|cc by(-sa)? [1-4]\.\d)", re.I)
# activity -> search terms. One person, one clear start of motion is what screening looks for.
QUERIES = {
    "sprint": ["sprint start blocks", "100 metres start", "sprinter running"],
    "long jump": ["long jump", "triple jump"],
    "high jump": ["high jump", "pole vault"],
    "throw": ["shot put", "javelin throw", "discus throw", "hammer throw"],
    "swim": ["swimming start dive", "swimmer dive", "platform diving"],
    "lift": ["weightlifting snatch", "clean and jerk", "deadlift", "kettlebell"],
    "gymnastics": ["gymnastics vault", "pommel horse", "balance beam", "trampoline"],
    "racket": ["tennis serve", "badminton serve", "table tennis serve"],
    "ball": ["penalty kick", "free throw basketball", "baseball pitch", "cricket bowling", "golf swing",
             "bowling throw"],
    "target": ["archery shot", "darts throw"],
    "board": ["skateboard trick", "ski jump", "snowboard jump"],
    "fitness": ["push-up exercise", "squat exercise", "rope jumping", "pull-up", "burpee"],
    "combat": ["boxing punching bag", "fencing lunge", "karate kata", "taekwondo kick"],
    "cycle": ["track cycling start", "bmx jump"],
    "other": ["juggling", "hula hoop", "cartwheel", "backflip", "handstand"],
}

# Second round: wider than sport, still one clear mover in front of a fixed camera.
QUERIES_2 = {
    "yoga": ["yoga pose demonstration", "tai chi", "stretching exercise", "calisthenics"],
    "dance": ["solo dance", "hula hoop", "rope skipping", "ballet"],
    "acrobatics": ["handstand", "cartwheel", "somersault", "parkour jump"],
    "craft": ["blacksmith hammer", "potter wheel", "chopping wood", "sawing wood", "glassblowing", "knitting",
              "weaving loom"],
    "music": ["drummer playing", "guitar playing", "piano playing hands", "violin playing"],
    "kitchen": ["chopping vegetables", "kneading dough", "pouring water glass", "cracking egg"],
    "animal": ["dog jumping", "cat jumping", "bird taking off", "squirrel eating", "frog jumping"],
    "sign": ["sign language", "semaphore"],
    "sport2": ["archery", "golf swing", "tennis serve", "bowling", "fencing", "table tennis"],
}


def api(**params):
    params.update(format="json", action="query")
    req = urllib.request.Request(API + "?" + urllib.parse.urlencode(params), headers={"User-Agent": UA})
    for attempt in range(4):
        try:
            return json.load(urllib.request.urlopen(req, timeout=60))
        except Exception:
            time.sleep(2 + 3 * attempt)
    return {}


def strip(html):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html or "")).strip()


def search(out, queries):
    rows = {}
    for cat, terms in queries.items():
        for term in terms:
            d = api(generator="search", gsrnamespace=6, gsrlimit=50, gsrsearch=f"filetype:video {term}",
                    prop="imageinfo", iiprop="url|size|mime|extmetadata",
                    iiextmetadatafilter="LicenseShortName|Artist|LicenseUrl")
            for p in d.get("query", {}).get("pages", {}).values():
                ii = (p.get("imageinfo") or [{}])[0]
                em = ii.get("extmetadata", {})
                lic = strip(em.get("LicenseShortName", {}).get("value"))
                dur = float(ii.get("duration") or 0)
                if not OK_LICENCE.match(lic) or not 4 <= dur <= 180 or min(ii.get("width", 0), ii.get("height", 0)) < 360:
                    continue
                rows.setdefault(p["title"], dict(
                    title=p["title"], category=cat, query=term, duration=round(dur, 2),
                    width=ii["width"], height=ii["height"], licence=lic,
                    licence_url=strip(em.get("LicenseUrl", {}).get("value")),
                    author=strip(em.get("Artist", {}).get("value"))[:200],
                    url=ii["url"], page=ii.get("descriptionurl", "")))
            time.sleep(0.5)
            print(f"{cat:10} {term:28} total kept {len(rows)}")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "candidates.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(next(iter(rows.values()))))
        w.writeheader()
        w.writerows(rows.values())
    print(f"{len(rows)} candidates -> {out / 'candidates.csv'}")


def download(out, limit):
    """Fetch the original and re-encode to 30 fps, 480 px tall, intra-only H.264 (no audio)."""
    rows = list(csv.DictReader(open(out / "candidates.csv")))[:limit or None]
    (out / "clips").mkdir(exist_ok=True)
    for i, r in enumerate(rows):
        dst = out / "clips" / (re.sub(r"[^A-Za-z0-9]+", "_", r["title"][5:].rsplit(".", 1)[0])[:80] + ".mp4")
        r["file"] = str(dst)
        if dst.exists():
            continue
        tmp = out / "clips" / "_tmp_src"
        try:
            req = urllib.request.Request(r["url"], headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=180) as resp, open(tmp, "wb") as fh:
                while chunk := resp.read(1 << 20):
                    fh.write(chunk)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-an", "-vf", "fps=30,scale=-2:480",
                            "-c:v", "libx264", "-g", "1", "-crf", "14", "-pix_fmt", "yuv420p", str(dst)],
                           check=True, timeout=600)
            print(f"[{i + 1}/{len(rows)}] {dst.name}")
        except Exception as e:
            print(f"[{i + 1}/{len(rows)}] FAILED {r['title']}: {e}")
        finally:
            tmp.unlink(missing_ok=True)
        time.sleep(1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["search", "download"])
    ap.add_argument("--out", default="data/real_src")
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--round", type=int, default=1, choices=[1, 2])
    args = ap.parse_args()
    if args.cmd == "search":
        search(Path(args.out), QUERIES if args.round == 1 else QUERIES_2)
    else:
        download(Path(args.out), args.max)


if __name__ == "__main__":
    main()
