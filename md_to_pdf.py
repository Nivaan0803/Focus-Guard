"""Render a Markdown file to PDF.

Usage:
    python md_to_pdf.py docs/ai-use-disclosure.md
    python md_to_pdf.py docs/ai-use-disclosure.md -o out.pdf

Written for the AI-use disclosure, which has to be submitted as a document
rather than read in a repository. It handles the subset of Markdown that file
uses: headings, paragraphs, bullet lists, tables, horizontal rules, bold and
inline code. It is not a general Markdown implementation and does not pretend
to be - anything it does not understand is rendered as plain text rather than
silently dropped, so nothing disappears from a document that may be read by a
reviewer.

Only matplotlib, already a project dependency.
"""
from __future__ import annotations

import argparse
import re
import sys
import textwrap
from pathlib import Path

PAGE_W, PAGE_H = 8.5, 11.0
MARGIN = 0.8
TOP = 0.945
BOTTOM = 0.055
LEFT = MARGIN / PAGE_W
RIGHT = 1 - MARGIN / PAGE_W

# Inline markers are stripped rather than styled: matplotlib has no rich text
# runs, and a half-applied bold is worse than none.
INLINE = re.compile(r"\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`|\[(.+?)\]\(.+?\)")


def plain(text: str) -> str:
    return INLINE.sub(lambda m: next(g for g in m.groups() if g is not None), text)


class Doc:
    def __init__(self, plt, pdf, footer):
        self.plt, self.pdf, self.footer = plt, pdf, footer
        self.fig = None
        self.y = 0.0
        self.page = 0
        self._new()

    def _new(self):
        if self.fig is not None:
            self._stamp()
            self.pdf.savefig(self.fig)
            self.plt.close(self.fig)
        self.fig = self.plt.figure(figsize=(PAGE_W, PAGE_H))
        self.page += 1
        self.y = TOP

    def _stamp(self):
        self.fig.text(RIGHT, 0.028, str(self.page), ha="right", va="center",
                      fontsize=8, color="#888888")
        if self.footer:
            self.fig.text(LEFT, 0.028, self.footer, ha="left", va="center",
                          fontsize=8, color="#888888")

    def room(self, need):
        if self.y - need < BOTTOM:
            self._new()

    def gap(self, amount):
        # Never start a page with blank space where a heading's margin fell.
        if self.y < TOP - 1e-9:
            self.y -= amount

    def text(self, body, size=9.5, weight="normal", indent=0.0, color="black",
             family="sans-serif", wrap=None):
        lead = size * 1.6 / 72.0 / PAGE_H
        width = wrap or int((RIGHT - LEFT - indent) * PAGE_W * 72 / (size * 0.52))
        lines = textwrap.wrap(body, max(20, width)) if body.strip() else [""]
        for chunk in lines:
            self.room(lead)
            self.fig.text(LEFT + indent, self.y, chunk, ha="left", va="top",
                          fontsize=size, weight=weight, color=color, family=family)
            self.y -= lead

    def heading(self, text, level):
        size = {1: 16, 2: 13, 3: 11}.get(level, 10)
        self.gap(0.022 if level > 1 else 0.030)
        self.room(0.055)
        self.text(text, size=size, weight="bold")
        if level == 1:
            self.gap(0.004)
            self.room(0.01)
            self.fig.add_artist(self.plt.Line2D([LEFT, RIGHT], [self.y, self.y],
                                transform=self.fig.transFigure,
                                color="#333333", linewidth=1.0))
        self.gap(0.010)

    def bullet(self, body, size=9.5):
        lead = size * 1.6 / 72.0 / PAGE_H
        lines = textwrap.wrap(body, 104)
        self.room(lead * len(lines))
        self.fig.text(LEFT + 0.008, self.y, "•", ha="left", va="top", fontsize=size)
        for chunk in lines:
            self.room(lead)
            self.fig.text(LEFT + 0.026, self.y, chunk, ha="left", va="top", fontsize=size)
            self.y -= lead
        self.gap(0.003)

    def rule(self):
        self.gap(0.010)
        self.room(0.02)
        self.fig.add_artist(self.plt.Line2D([LEFT, RIGHT], [self.y, self.y],
                            transform=self.fig.transFigure,
                            color="#bbbbbb", linewidth=0.7))
        self.gap(0.014)

    def table(self, rows):
        """Monospaced and column-padded. A proportional font would not line up,
        and a table that does not line up is harder to read than a list."""
        if not rows:
            return
        cols = max(len(r) for r in rows)
        rows = [r + [""] * (cols - len(r)) for r in rows]
        size = 8.0
        lead = size * 1.65 / 72.0 / PAGE_H

        # How many characters actually fit, derived from the page rather than
        # guessed: DejaVu Sans Mono advances about 0.605 em. A hardcoded
        # budget here overflowed the right margin and silently truncated the
        # last column of every wide table - the text was simply gone from the
        # page, which in a document a reviewer reads is the worst failure mode.
        usable = int((RIGHT - LEFT) * PAGE_W * 72 / (size * 0.605))
        budget = usable - 2 * (cols - 1)        # 2 spaces between columns
        widths = [max(len(r[i]) for r in rows) for i in range(cols)]
        while sum(widths) > budget:
            widths[widths.index(max(widths))] -= 1

        for index, row in enumerate(rows):
            wrapped = [textwrap.wrap(cell, max(6, widths[i])) or [""]
                       for i, cell in enumerate(row)]
            height = max(len(w) for w in wrapped)
            self.room(lead * (height + 1))
            for line_no in range(height):
                parts = [(wrapped[i][line_no] if line_no < len(wrapped[i]) else "")
                         .ljust(widths[i]) for i in range(cols)]
                self.fig.text(LEFT, self.y, "  ".join(parts), ha="left", va="top",
                              fontsize=size, family="monospace",
                              weight="bold" if index == 0 else "normal")
                self.y -= lead
            if index == 0:
                self.fig.add_artist(self.plt.Line2D(
                    [LEFT, RIGHT], [self.y + lead * 0.25, self.y + lead * 0.25],
                    transform=self.fig.transFigure, color="#aaaaaa", linewidth=0.6))
                self.y -= lead * 0.15
        self.gap(0.010)

    def finish(self):
        self._stamp()
        self.pdf.savefig(self.fig)
        self.plt.close(self.fig)
        return self.page


def split_row(line: str):
    return [plain(c.strip()) for c in line.strip().strip("|").split("|")]


def render(md: str, out: Path, footer: str) -> int:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        sys.exit("matplotlib is required: pip install matplotlib")

    lines = md.splitlines()
    out.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out) as pdf:
        doc = Doc(plt, pdf, footer)
        i = 0
        paragraph: list[str] = []

        def flush():
            nonlocal paragraph
            if paragraph:
                doc.text(plain(" ".join(paragraph)))
                doc.gap(0.009)
                paragraph = []

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            if not stripped:
                flush(); i += 1; continue

            if stripped.startswith("#"):
                flush()
                level = len(stripped) - len(stripped.lstrip("#"))
                doc.heading(plain(stripped.lstrip("#").strip()), level)
                i += 1; continue

            if set(stripped) <= {"-", "*", "_"} and len(stripped) >= 3:
                flush(); doc.rule(); i += 1; continue

            # a table: a header row followed by a |---|---| separator
            if stripped.startswith("|") and i + 1 < len(lines) \
                    and re.match(r"^\|[\s:|-]+\|$", lines[i + 1].strip()):
                flush()
                rows = [split_row(stripped)]
                i += 2
                while i < len(lines) and lines[i].strip().startswith("|"):
                    rows.append(split_row(lines[i]))
                    i += 1
                doc.table(rows)
                continue

            if re.match(r"^[-*+]\s+", stripped):
                flush()
                doc.bullet(plain(re.sub(r"^[-*+]\s+", "", stripped)))
                i += 1; continue

            if re.match(r"^\d+\.\s+", stripped):
                flush()
                doc.bullet(plain(stripped))
                i += 1; continue

            paragraph.append(stripped)
            i += 1

        flush()
        return doc.finish()


def main(argv) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--footer", default=None, help="footer text (default: the filename)")
    args = ap.parse_args(argv)

    if not args.source.is_file():
        sys.exit(f"No file at {args.source}")
    out = args.out or args.source.with_suffix(".pdf")
    pages = render(args.source.read_text(encoding="utf-8"), out,
                   args.footer if args.footer is not None else args.source.name)
    print(f"Wrote {out}  ({pages} page(s))")


if __name__ == "__main__":
    main(sys.argv[1:])
