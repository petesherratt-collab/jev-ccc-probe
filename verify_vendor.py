#!/usr/bin/env python3
"""
Verify the vendored harness modules against their pins.

WHY THIS EXISTS
---------------
This repository is a separate research line from the Generated-Trace Leak Harness,
but its stimuli are built by the harness's own item banks and prompt builders. Those
modules are **vendored byte-identical and hash-pinned** rather than submoduled,
following the harness's own practice: it vendors and pins its substrate block
(`_SUBSTRATE_PIN`, byte-identical v0.6 through v1.5.1) for exactly this reason. A pin
makes this repo self-contained and its stimuli reproducible years later; a submodule
makes it a moving reference that needs the other repo to still exist, on the same
branch, with the same history.

The pin is a **self-consistency / drift detector, not an authenticity proof** — the
same honest caveat the harness applies to its own file pin. It catches "someone edited
a vendored module and the stimuli silently changed". It does not prove the vendored
copy matches what upstream had, which would need an external anchor.

Run it before any run that produces evidence, and in CI:

    python verify_vendor.py            # exit 0 all pins match, 1 otherwise
    python verify_vendor.py --repin     # recompute pins (deliberate update only)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

PINS = pathlib.Path("VENDOR_PINS.json")
VENDOR = pathlib.Path("experiments")


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repin", action="store_true",
                    help="recompute and rewrite the pins. Only for a deliberate "
                         "vendor update, and say so in the commit message.")
    args = ap.parse_args()

    if not PINS.exists():
        print(f"{PINS} missing", file=sys.stderr)
        return 1
    doc = json.loads(PINS.read_text(encoding="utf-8"))
    pinned: dict[str, str] = doc["modules"]
    # Walk the whole vendor tree, not just *.py. The arith stimuli depend on the
    # frozen text caches under experiments/results/, and an earlier version of this
    # checker globbed only "*.py" -- so those files were pinned but never checked,
    # which is the silent-drift case this script exists to prevent.
    present = {q.relative_to(VENDOR).as_posix(): q
               for q in sorted(VENDOR.rglob("*"))
               if q.is_file() and "__pycache__" not in q.parts}

    if args.repin:
        doc["modules"] = {n: sha(q) for n, q in present.items()}
        PINS.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8", newline="\n")
        print(f"repinned {len(present)} modules")
        return 0

    missing = sorted(set(pinned) - set(present))
    extra = sorted(set(present) - set(pinned))
    changed = [n for n in sorted(set(pinned) & set(present))
               if sha(present[n]) != pinned[n]]
    py = sum(1 for n in present if n.endswith(".py"))
    data = len(present) - py

    print(f"vendored from  : {doc['source_repo']}")
    print(f"source commit  : {doc['source_commit'][:12]} ({doc['source_branch']})")
    print(f"vendored at    : {doc['vendored_at']}")
    print(f"files pinned   : {len(pinned)}  ({py} modules, {data} frozen data files)")

    ok = True
    for label, items in (("MISSING (pinned but absent)", missing),
                         ("UNPINNED (present but not pinned)", extra),
                         ("CHANGED (pin mismatch)", changed)):
        if items:
            ok = False
            print(f"\n{label}:")
            for n in items:
                print(f"  {n}")

    if not ok:
        print("\nVENDOR DRIFT. The stimuli this repo produces are not the stimuli "
              "its results were\nmeasured on. Do not publish a run until this is "
              "resolved: either restore the\nvendored copies, or --repin "
              "deliberately and re-run everything downstream.",
              file=sys.stderr)
        return 1

    print("\nall pins match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
