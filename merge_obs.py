#!/usr/bin/env python3
"""
Build a clean observation file from several runs, filtering by encoding.

Needed because the structured arm was re-run after a loader fix. The original
file holds valid verbatim rows AND invalid structured rows (SQL sent with no
schema); the new file holds the fixed structured rows. Concatenating them would
average the broken structured cells with the fixed ones, which is worse than
either alone — assemble() takes the mean per cell and cannot know one source is
junk.

So: take only the encodings you trust from each file.

    python merge_obs.py --out _obs_clean.jsonl \\
        _obs_ccc.jsonl:verbatim \\
        _obs_ccc2.jsonl:structured

Each argument is PATH or PATH:ENCODING. With an encoding, only rows of that
encoding are taken. Without, all rows are taken.

Prints what it took from where, and refuses to write a file in which the same
(domain, item, condition, candidate, encoding) cell came from two different
sources — that is the mistake this script exists to prevent.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict

KEY = ("domain", "item_id", "condition", "candidate_type", "encoding",
       "repetition")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="+", metavar="PATH[:ENCODING]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-cross-source-cells", action="store_true",
                    help="permit the same cell from two sources (they will be "
                         "averaged). Only sensible for genuine repeats of the "
                         "SAME code.")
    args = ap.parse_args()

    kept: list[dict] = []
    origin: dict[tuple, set[str]] = defaultdict(set)
    report = []

    for spec in args.sources:
        path, _, enc = spec.partition(":")
        try:
            with open(path, encoding="utf-8") as fh:
                rows = [json.loads(l) for l in fh if l.strip()]
        except OSError as e:
            print(f"cannot read {path}: {e}", file=sys.stderr)
            return 2

        before = len(rows)
        if enc:
            rows = [r for r in rows if r.get("encoding") == enc]
        encs = Counter(r.get("encoding") for r in rows)
        report.append((path, enc or "(all)", before, len(rows), dict(encs)))

        for r in rows:
            kept.append(r)
            origin[tuple(str(r.get(k, "")) for k in KEY)].add(path)

    print("source                          filter        rows in   taken   encodings")
    for path, enc, before, after, encs in report:
        print(f"  {path:<28} {enc:<12} {before:>8} {after:>7}   {encs}")

    clash = {k: v for k, v in origin.items() if len(v) > 1}
    if clash and not args.allow_cross_source_cells:
        print(f"\nREFUSING TO WRITE: {len(clash)} cell(s) appear in more than "
              f"one source.", file=sys.stderr)
        for k, v in list(clash.items())[:5]:
            print(f"  {'|'.join(k)}  from {sorted(v)}", file=sys.stderr)
        print("\nThose cells would be averaged across sources. If the sources "
              "ran DIFFERENT code (e.g. before and after a fix), that average "
              "is meaningless -- narrow the filters so each cell comes from one "
              "source. If they are genuine repeats of the same code, pass "
              "--allow-cross-source-cells.", file=sys.stderr)
        return 1

    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        for r in kept:
            fh.write(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n")

    cells = len(origin)
    print(f"\nwrote {len(kept)} rows ({cells} distinct cells) to {args.out}")
    if clash:
        print(f"  note: {len(clash)} cell(s) came from multiple sources and "
              f"will be averaged (you passed --allow-cross-source-cells)")
    print(f"\nnext: python analyse_ccc.py --obs {args.out} --by-domain")
    return 0


if __name__ == "__main__":
    sys.exit(main())
