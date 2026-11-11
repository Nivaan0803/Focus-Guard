"""Download everything the study has collected into study/.

Usage:
    python fetch_study_data.py                 # all of it, into study/
    python fetch_study_data.py --participant P01
    python fetch_study_data.py --base https://my-other-deployment.vercel.app

The export token is read from web/.env.local, which `vercel env pull` writes
and .gitignore keeps out of the repo. Nothing is printed that would reveal it.

Files land in study/, which .gitignore excludes from the repo because it holds
human-participants data. Keep it that way.

WHY THIS EXISTS
---------------
The export endpoint needs a bearer token, and on Windows `curl` is an alias for
Invoke-WebRequest, which takes different arguments from real curl - so the
obvious one-liner silently does the wrong thing. One command that cannot be
mistyped is worth more than a README section explaining the difference.
"""
from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_BASE = "https://focus-guard-study.vercel.app"


def repo_root() -> Path:
    """Anchor every path to the repo, not to the current directory.

    Run from inside web/ with relative paths and the output lands in
    web/study/ - which the deployment SERVES. Participant CSVs would then be
    downloadable by anyone with the URL, and nothing in .gitignore would stop
    them being committed. Resolving from the repo root removes the chance.
    """
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / "web" / "consent" / "consent.json").is_file():
            return candidate
    return here


ROOT = repo_root()
ENV_FILE = ROOT / "web" / ".env.local"
OUT_DIR = ROOT / "study"

# (kind, filename, what it feeds)
EXPORTS = [
    ("study",      "results.csv",            "analyze_study.py  (arithmetic sessions)"),
    ("study_sart", "results_sart.csv",       "analyze_study.py  (SART sessions)"),
    ("training",   "training.csv",           "train_model.py    <- THE ONE TO TRAIN ON"),
    ("features",   "all_features.csv",       "raw rows, provenance labels - not for training"),
    ("trials",     "sart_trials.csv",        "analyze_sart.py"),
    ("probes",     "probe_responses.csv",    "analyze_sart.py"),
    ("consent",    "consent_receipts.csv",   "your SRC compliance record"),
]


def read_token(path: Path) -> str:
    if not path.is_file():
        sys.exit(f"No {path}. Run `vercel env pull` inside web/ first.")
    # vercel env pull writes UTF-8 with a BOM; left in, the first key name is
    # unmatchable and the token silently comes back empty.
    text = path.read_text(encoding="utf-8-sig")
    for line in text.splitlines():
        if line.startswith("EXPORT_TOKEN="):
            token = line.split("=", 1)[1].strip().strip('"').strip("'")
            if token:
                return token
    sys.exit(f"No EXPORT_TOKEN in {path}. Set it in the Vercel dashboard and "
             "run `vercel env pull` again.")


def fetch(base: str, kind: str, token: str, participant: str | None) -> str:
    url = f"{base}/api/export?kind={kind}"
    if participant:
        url += f"&participant={participant}"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as err:
        if err.code == 401:
            sys.exit("401 from the export endpoint. The EXPORT_TOKEN in "
                     f"{ENV_FILE} does not match the one deployed. Re-run "
                     "`vercel env pull` inside web/.")
        if err.code == 503:
            sys.exit("503: no database is attached to the deployment.")
        sys.exit(f"{kind}: HTTP {err.code} {err.reason}")
    except urllib.error.URLError as err:
        sys.exit(f"{kind}: could not reach {base} ({err.reason})")


def rows_in(text: str) -> int:
    body = text.strip()
    if not body:
        return 0
    return max(0, len(body.split("\r\n" if "\r\n" in body else "\n")) - 1)


def main(argv) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=DEFAULT_BASE, help=f"deployment URL (default {DEFAULT_BASE})")
    ap.add_argument("--participant", default=None, help="limit to one participant code, e.g. P01")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)

    out = args.out.resolve()
    # Belt and braces: --out could still point somewhere served.
    web = (ROOT / "web").resolve()
    if out == web or web in out.parents:
        sys.exit(f"Refusing to write participant data to {out}.\n"
                 "Everything under web/ is published by the deployment, so "
                 "those files would be downloadable by anyone with the URL.")
    args.out = out

    token = read_token(ENV_FILE)
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"{args.base}\n")
    counts = {}
    for kind, filename, feeds in EXPORTS:
        # participant filtering is not meaningful for the consent registry
        who = None if kind == "consent" else args.participant
        text = fetch(args.base, kind, token, who)
        path = args.out / filename
        path.write_text(text, encoding="utf-8", newline="")
        counts[kind] = rows_in(text)
        print(f"  {filename:<24}{counts[kind]:>7} rows   {feeds}")

    print(f"\nSaved to {args.out}/  (gitignored - it holds participant data)")

    # ---- the checks worth making before you trust any of it ----
    print()
    if counts["training"] == 0:
        print("!! training.csv is EMPTY.")
        print("   kind=training only returns `label_only` sessions, because labels")
        print("   collected while the system was beeping at the participant are")
        print("   confounded. Run a label_only block, or re-export with")
        print("   ?condition=all if you accept that trade-off.")
    else:
        labels = [line.rsplit(",", 1)[-1].strip()
                  for line in (args.out / "training.csv").read_text(encoding="utf-8")
                  .strip().splitlines()[1:]]
        distracted = labels.count("distracted")
        focused = labels.count("focused")
        print(f"training.csv: {focused} focused, {distracted} distracted")
        if distracted == 0 or focused == 0:
            print("!! Only one class. train_model.py will refuse this file.")
            print("   You need more participants, or more probes per session.")
        else:
            minority = min(distracted, focused) / (distracted + focused)
            if minority < 0.15:
                print(f"!! Minority class is {minority:.1%}. Accuracy will look high for")
                print("   the wrong reason - read F1 on the distracted class, and")
                print("   consider --balance when training.")

    if counts["probes"] == 0:
        print("\n!! No probe answers. Without them there are no labels at all.")

    print("\nNext:")
    print(f"  python analyze_sart.py {args.out}/sart_trials.csv {args.out}/probe_responses.csv")
    print("      ^ do this FIRST. If lapses are not elevated before off-task")
    print("        probes, the labels are noise and nothing trained on them means")
    print("        anything.")
    print(f"  python analyze_study.py {args.out}/results_sart.csv")
    print(f"  python train_model.py {args.out}/training.csv")


if __name__ == "__main__":
    main(sys.argv[1:])
