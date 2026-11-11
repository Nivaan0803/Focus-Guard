"""Build a printable consent form from web/consent/consent.json.

Usage:
    python make_consent_pdf.py                      # -> docs/isef-informed-consent-v2-DRAFT.pdf
    python make_consent_pdf.py -o path/to/out.pdf
    python make_consent_pdf.py --final              # drops the DRAFT banner

WHY THIS IS A SCRIPT AND NOT A ONE-OFF FILE
-------------------------------------------
consent.json carries a warning that the on-screen summary is paraphrased FROM
the attached form, so the two must be edited together or they drift. Generating
the form FROM consent.json removes the possibility: there is one source of
words, and the PDF is a rendering of it. Edit consent.json, bump its `version`,
re-run this, and the screen and the paper say the same thing by construction.

That matters here more than usual, because web/lib/pdf.js EMBEDS the attached
form into every participant's signed document. A form describing a task they
did not do would be bound into their signed consent.

WHAT THIS IS NOT
----------------
It is NOT an approved ISEF form, and the default output says so on every page.
An adult sponsor and the SRC approve consent documents; a script cannot. The
intended use is to hand your sponsor something accurate to edit rather than a
blank page, and to re-render it once they have agreed the wording.

Only matplotlib is needed, which is already a dependency for the study charts.
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from datetime import date
from pathlib import Path

CONSENT = Path("web/consent/consent.json")
DEFAULT_OUT = Path("docs/isef-informed-consent-v2-DRAFT.pdf")

# Letter, in inches.
PAGE_W, PAGE_H = 8.5, 11.0
MARGIN = 0.85
TOP = 0.93
BOTTOM = 0.07


class Page:
    """A cursor that walks down a page and starts a new one when it runs out.

    matplotlib has no text flow, so this is the minimum needed to lay out a
    document: a y position in figure coordinates, and a rule for when to break.
    """

    def __init__(self, pdf, plt, banner):
        self.pdf = pdf
        self.plt = plt
        self.banner = banner
        self.fig = None
        self.y = 0.0
        self.pages = 0
        self._new()

    def _new(self):
        if self.fig is not None:
            self.pdf.savefig(self.fig)
            self.plt.close(self.fig)
        self.fig = self.plt.figure(figsize=(PAGE_W, PAGE_H))
        self.pages += 1
        self.y = TOP
        if self.banner:
            self.fig.text(0.5, 0.975, self.banner, ha="center", va="top",
                          fontsize=9, color="#b00020", weight="bold")
            self.y = TOP - 0.025

    def space(self, amount=0.012):
        self.y -= amount

    def _room(self, needed):
        if self.y - needed < BOTTOM:
            self._new()

    def line(self, text, size=10, weight="normal", indent=0.0, color="black",
             wrap=96, leading=None):
        leading = leading or (size * 1.55 / 72.0 / PAGE_H)
        chunks = textwrap.wrap(text, wrap) if text else [""]
        self._room(leading * len(chunks))
        for chunk in chunks:
            self.fig.text(MARGIN / PAGE_W + indent, self.y, chunk,
                          ha="left", va="top", fontsize=size,
                          weight=weight, color=color)
            self.y -= leading

    def heading(self, text, size=12):
        self.space(0.016)
        self._room(0.05)
        self.line(text, size=size, weight="bold")
        self.space(0.006)

    def bullet(self, text, size=10):
        leading = size * 1.55 / 72.0 / PAGE_H
        chunks = textwrap.wrap(text, 88)
        self._room(leading * (len(chunks) + 1))
        self.fig.text(MARGIN / PAGE_W, self.y, "•", ha="left", va="top", fontsize=size)
        for i, chunk in enumerate(chunks):
            self.fig.text(MARGIN / PAGE_W + 0.022, self.y, chunk,
                          ha="left", va="top", fontsize=size)
            self.y -= leading
        self.space(0.004)

    def checkbox(self, text, size=10):
        leading = size * 1.55 / 72.0 / PAGE_H
        chunks = textwrap.wrap(text, 84)
        self._room(leading * (len(chunks) + 1))
        x = MARGIN / PAGE_W
        self.fig.patches.append(self.plt.Rectangle(
            (x, self.y - 0.0105), 0.013, 0.0105, transform=self.fig.transFigure,
            fill=False, edgecolor="black", linewidth=0.9))
        for chunk in chunks:
            self.fig.text(x + 0.026, self.y, chunk, ha="left", va="top", fontsize=size)
            self.y -= leading
        self.space(0.005)

    def rule(self):
        self._room(0.02)
        self.fig.add_artist(self.plt.Line2D(
            [MARGIN / PAGE_W, 1 - MARGIN / PAGE_W], [self.y, self.y],
            transform=self.fig.transFigure, color="#999999", linewidth=0.7))
        self.space(0.016)

    def signature_block(self, who, note=None):
        # Reserve the WHOLE block, not just its first line. Letting it split
        # strands the date field at the top of the next page, which is exactly
        # the sort of thing someone signs without filling in.
        self._room(0.235 if note else 0.215)
        self.space(0.022)
        self.line(who, size=10, weight="bold")
        self.space(0.030)
        for label, width in (("Printed name", 0.46), ("Signature", 0.46), ("Date", 0.20)):
            self._room(0.05)
            x0 = MARGIN / PAGE_W
            self.fig.add_artist(self.plt.Line2D(
                [x0, x0 + width], [self.y, self.y],
                transform=self.fig.transFigure, color="black", linewidth=0.8))
            self.fig.text(x0, self.y - 0.011, label, ha="left", va="top",
                          fontsize=8, color="#555555")
            self.space(0.047)
        if note:
            self.line(note, size=8, color="#555555")

    def finish(self):
        total = self.pages
        self.pdf.savefig(self.fig)
        self.plt.close(self.fig)
        return total


def build(cfg: dict, out: Path, draft: bool) -> int:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        sys.exit("matplotlib is required: pip install matplotlib")

    banner = ("DRAFT - FOR SPONSOR AND SRC REVIEW - NOT FOR USE WITH PARTICIPANTS"
              if draft else "")

    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        p = Page(pdf, plt, banner)

        p.line("INFORMED CONSENT / ASSENT", size=15, weight="bold")
        p.space(0.010)
        p.line(cfg.get("studyTitle", ""), size=10, weight="bold", wrap=88)
        p.space(0.010)
        p.line(f"Student researcher: {cfg.get('researcher', '')}", size=10)
        p.line(f"Adult sponsor: {cfg.get('sponsor', '')}"
               + (f"  ({cfg['sponsorContact']})" if cfg.get("sponsorContact") else ""), size=10)
        p.line(f"Form version: {cfg.get('version', '')}", size=9, color="#555555")
        p.space(0.008)
        p.rule()

        p.heading("What this study involves")
        for item in cfg.get("summary", []):
            p.bullet(item)

        p.heading("What you are agreeing to")
        p.line("Please read each statement. Tick it only if it is true for you.",
               size=9, color="#555555")
        p.space(0.010)
        for statement in cfg.get("statements", []):
            p.checkbox(statement)

        p.heading("If you have questions")
        p.line(
            f"Contact the adult sponsor, {cfg.get('sponsor', '')}"
            + (f", at {cfg['sponsorContact']}" if cfg.get("sponsorContact") else "")
            + ". You may withdraw at any point, before or during the study, "
              "without giving a reason and without any consequence.", size=10)

        p.heading("Signatures")
        p.signature_block("Research participant")
        p.signature_block(
            f"Parent or guardian (required if the participant is under "
            f"{cfg.get('minorAgeThreshold', 18)})")
        p.signature_block("Student researcher")

        if draft:
            p.space(0.020)
            p.rule()
            p.line("This document was generated from web/consent/consent.json by "
                   "make_consent_pdf.py. It has not been reviewed or approved. "
                   "Edit the wording in consent.json rather than in this file, so "
                   "the form and the text shown on screen cannot disagree, then "
                   "re-run the script.", size=8, color="#b00020")

        pages = p.finish()

    return pages


def main(argv) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--consent", type=Path, default=CONSENT)
    ap.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--final", action="store_true",
                    help="drop the DRAFT banner (only after the SRC has approved the wording)")
    args = ap.parse_args(argv)

    if not args.consent.is_file():
        sys.exit(f"No consent config at {args.consent}")
    cfg = json.loads(args.consent.read_text(encoding="utf-8"))

    pages = build(cfg, args.out, draft=not args.final)

    print(f"Wrote {args.out}  ({pages} page(s))")
    print(f"  form version : {cfg.get('version')}")
    print(f"  summary      : {len(cfg.get('summary', []))} points")
    print(f"  statements   : {len(cfg.get('statements', []))}")
    print(f"  generated    : {date.today().isoformat()}")
    if args.final:
        print("\n  --final used: no DRAFT banner. Only do this once your sponsor and")
        print("  SRC have approved this exact wording.")
    else:
        print("\n  Marked DRAFT on every page. Give it to your sponsor to edit.")
        print("  Change wording in consent.json, not in the PDF, then re-run this.")
    print("\nWhen the wording is approved:")
    print("  1. python make_consent_pdf.py --final -o web/consent/isef-informed-consent-v2.pdf")
    print("  2. set \"pdfPresent\": true in web/consent/consent.json")
    print("  3. restore 'and the attached consent form' to the first statement")
    print("  4. git push")


if __name__ == "__main__":
    main(sys.argv[1:])
