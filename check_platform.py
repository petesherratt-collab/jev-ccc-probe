#!/usr/bin/env python3
"""
Prove the platform is not touching what the model sees.

The claim being tested is narrow and worth stating precisely: a run on Windows
and a run on Linux should send BYTE-IDENTICAL requests to Jev. It should hold --
the prompts come from Python string literals and UTF-8 JSON, and `ObsWriter`
pins LF -- but "should hold" is not a measurement, and an evidence stream headed
for a DOI'd archive deserves the measurement.

Every row carries `state_sha`: a SHA-256 of the exact request (state plus
questions, canonical JSON, private keys stripped). This tool reduces a run to a
single STIMULUS DIGEST over those hashes, keyed by cell. Two runs on different
platforms that produce the same digest sent the same requests. Full stop.

It also computes a VALUE DIGEST over the returned probabilities. That one is
only meaningful for a deterministic model -- check determinism.py first. If the
model is deterministic and the value digests match too, the whole pipeline is
platform-independent end to end.

MODES
-----
    # fingerprint one run
    python check_platform.py --obs _obs_ccc.jsonl

    # compare two runs from different machines
    python check_platform.py --obs linux_obs.jsonl --against windows_obs.jsonl

Exit codes: 0 all compared digests match, 1 a mismatch, 2 bad input.
Designed to be usable as a CI gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict

import jevlib as J

# The cell key. Anything not in here is run metadata (run_id, timestamps,
# latency, cost) and must NOT enter a digest -- those legitimately differ
# between two runs of the same design.
KEY_FIELDS = ("domain", "item_id", "condition", "candidate_type",
              "encoding", "repetition")

NOUL_DP = 9   # generous: a real platform difference would not hide in the 9th dp


def cell_key(row: dict) -> str:
    return "|".join(str(row.get(f, "")) for f in KEY_FIELDS)


def digest(lines: list[str]) -> str:
    h = hashlib.sha256()
    for line in sorted(lines):
        h.update(line.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def summarise(path: str) -> dict:
    rows = J.read_obs(path)
    ok = [r for r in rows if r.get("ok")]

    stim, vals, dupes = {}, {}, Counter()
    for r in ok:
        k = cell_key(r)
        dupes[k] += 1
        if r.get("state_sha"):
            stim[k] = r["state_sha"]
        if r.get("noul") is not None:
            vals[k] = f"{float(r['noul']):.{NOUL_DP}f}"

    plats = Counter()
    for r in rows:
        plats[(r.get("os") or r.get("platform") or "unrecorded",
               r.get("python") or "?",
               r.get("machine") or "?")] += 1

    return {
        "path": path,
        "rows": len(rows),
        "ok": len(ok),
        "failed": len(rows) - len(ok),
        "cells": len(stim),
        "repeated_cells": sum(1 for v in dupes.values() if v > 1),
        "platforms": plats,
        "models": Counter(r.get("model_served") for r in rows if r.get("model_served")),
        "stim": stim,
        "vals": vals,
        "stim_digest": digest([f"{k}={v}" for k, v in stim.items()]),
        "val_digest": digest([f"{k}={v}" for k, v in vals.items()]) if vals else None,
        "domains": Counter(r.get("domain") for r in ok),
    }


def report_one(s: dict) -> None:
    print(f"\n{s['path']}")
    print(f"  rows                : {s['rows']}  (ok {s['ok']}, failed {s['failed']})")
    print(f"  distinct cells      : {s['cells']}")
    if s["repeated_cells"]:
        print(f"  cells with repeats  : {s['repeated_cells']}  "
              f"(digest uses the last row per cell)")
    print(f"  by domain           : {dict(s['domains'])}")
    for (os_, py, mach), n in s["platforms"].items():
        print(f"  platform            : {os_} / python {py} / {mach}  ({n} rows)")
    for m, n in s["models"].items():
        print(f"  model served        : {m}  ({n} rows)")
    print(f"  STIMULUS DIGEST     : {s['stim_digest']}")
    print(f"  VALUE DIGEST        : {s['val_digest'] or 'n/a (no probabilities)'}")


def compare(a: dict, b: dict) -> int:
    print("\n" + "=" * 74)
    print("CROSS-PLATFORM COMPARISON")
    print("=" * 74)

    pa = {p[0] for p in a["platforms"]}
    pb = {p[0] for p in b["platforms"]}
    print(f"  A: {a['path']}  [{', '.join(sorted(pa))}]")
    print(f"  B: {b['path']}  [{', '.join(sorted(pb))}]")
    if pa == pb and pa != {"unrecorded"}:
        print(f"\n  ! Both runs report the same platform ({', '.join(sorted(pa))}).")
        print("  ! This compares two runs on ONE platform, which tests")
        print("  ! reproducibility but says nothing about cross-platform")
        print("  ! equivalence. Run one of them on the other machine.")

    failed = 0

    # ---- stimulus ------------------------------------------------------- #
    print("\n  STIMULUS EQUIVALENCE (what was sent to the model)")
    only_a = sorted(set(a["stim"]) - set(b["stim"]))
    only_b = sorted(set(b["stim"]) - set(a["stim"]))
    shared = sorted(set(a["stim"]) & set(b["stim"]))
    print(f"    cells in both       : {len(shared)}")

    # A missing cell is a FAILURE, not a footnote. One run covering fewer cells
    # than the other means a domain failed to load, a run was interrupted, or
    # the two grids differ -- and in every one of those cases "the cells we both
    # have are identical" is not a result anyone should act on. Reporting MATCH
    # here would let a half-finished run through a CI gate.
    if only_a or only_b:
        failed = 1
        print(f"    only in A           : {len(only_a)}")
        print(f"    only in B           : {len(only_b)}")
        for k in (only_a[:5] + only_b[:5]):
            print(f"      {k}")
        gone = Counter(k.split("|")[0] for k in (only_a + only_b))
        print(f"    affected domains    : {dict(gone)}")
        print("\n    INCOMPLETE: the two runs do not cover the same cells.")
        print("    A whole domain missing on one side usually means its module")
        print("    failed to import there -- check that run's stderr for a")
        print("    'FAILED to load' line. Fix that before reading anything")
        print("    below, which compares only the overlap.")

    diff = [k for k in shared if a["stim"][k] != b["stim"][k]]
    if not shared:
        print("    nothing to compare.")
        failed = 1
    elif diff:
        failed = 1
        print(f"    MISMATCHED          : {len(diff)} of {len(shared)}")
        print("\n    The two platforms sent DIFFERENT requests. Any comparison")
        print("    of the results is confounded. First differing cells:")
        for k in diff[:10]:
            print(f"      {k}")
            print(f"        A {a['stim'][k]}")
            print(f"        B {b['stim'][k]}")
        if len(diff) > 10:
            print(f"      ... and {len(diff) - 10} more")
        by_dom = Counter(k.split("|")[0] for k in diff)
        by_enc = Counter(k.split("|")[4] for k in diff)
        print(f"\n    by domain   : {dict(by_dom)}")
        print(f"    by encoding : {dict(by_enc)}")
        print("\n    Likeliest causes, in order: a text file read without an")
        print("    explicit encoding; a path or newline leaking into a prompt;")
        print("    a different commit of the harness on the two machines.")
        print("    Check the repo is at the same commit on both before")
        print("    suspecting anything subtler.")
    elif only_a or only_b:
        print(f"    the {len(shared)} shared cells are identical, but see the")
        print("    INCOMPLETE warning above -- this is not a clean match.")
    else:
        print(f"    MATCH               : all {len(shared)} cells identical")
        print("    The platform is not touching what the model sees.")

    # ---- values --------------------------------------------------------- #
    print("\n  VALUE EQUIVALENCE (what the model returned)")
    if not a["vals"] or not b["vals"]:
        print("    one side has no probabilities -- skipped.")
    else:
        vshared = sorted(set(a["vals"]) & set(b["vals"]))
        vdiff = [k for k in vshared if a["vals"][k] != b["vals"][k]]
        if not vshared:
            print("    no shared cells.")
        elif vdiff:
            worst = max(vdiff, key=lambda k: abs(float(a["vals"][k])
                                                 - float(b["vals"][k])))
            gap = abs(float(a["vals"][worst]) - float(b["vals"][worst]))
            print(f"    DIFFER              : {len(vdiff)} of {len(vshared)}")
            print(f"    largest gap         : {gap:.6f}  ({worst})")
            if diff:
                print("    Expected: the requests differed too. Fix that first.")
            else:
                print("    The requests were identical, so this is the MODEL")
                print("    being non-deterministic, not a platform problem.")
                print("    Confirm with determinism.py and size the run for")
                print("    repeats. This is not a reason to distrust either box.")
        else:
            print(f"    MATCH               : all {len(vshared)} cells identical")
            print("    Deterministic and platform-independent end to end.")

    print("\n" + "-" * 74)
    if failed and (only_a or only_b) and not diff:
        print("  RESULT: incomplete comparison. The runs cover different cells,")
        print("  so equivalence is not established either way.")
    elif failed:
        print("  RESULT: stimulus mismatch. Do not pool these runs.")
    else:
        print("  RESULT: the two runs sent identical requests.")
        print("  Safe to state in the methods section that the platform does")
        print("  not affect the stimuli, and to cite these digests.")
    print("-" * 74)
    return failed


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--obs", required=True, help="observation file (run A)")
    ap.add_argument("--against", default=None,
                    help="second observation file (run B), from the other machine")
    ap.add_argument("--json", action="store_true",
                    help="emit the digests as JSON for a manifest")
    args = ap.parse_args()

    try:
        a = summarise(args.obs)
    except (OSError, J.JevError) as e:
        print(f"cannot read {args.obs}: {e}", file=sys.stderr)
        return 2

    if not a["stim"]:
        print(f"{args.obs} has no state_sha values. It was written by an older "
              f"version of the runner; re-run to get comparable rows.",
              file=sys.stderr)
        return 2

    print("=" * 74)
    print("RUN FINGERPRINT")
    print("=" * 74)
    report_one(a)

    rc = 0
    b = None
    if args.against:
        try:
            b = summarise(args.against)
        except (OSError, J.JevError) as e:
            print(f"cannot read {args.against}: {e}", file=sys.stderr)
            return 2
        report_one(b)
        rc = compare(a, b)
    else:
        print("\n  Record the STIMULUS DIGEST above. Run the same command on the")
        print("  other machine and compare, or pass --against <its obs file>.")

    if args.json:
        payload = {"a": {k: a[k] for k in
                         ("path", "rows", "ok", "cells", "stim_digest", "val_digest")}}
        if b:
            payload["b"] = {k: b[k] for k in
                            ("path", "rows", "ok", "cells", "stim_digest", "val_digest")}
            payload["stimulus_match"] = (a["stim_digest"] == b["stim_digest"])
        print("\n" + json.dumps(payload, indent=2))

    return rc


if __name__ == "__main__":
    sys.exit(main())
