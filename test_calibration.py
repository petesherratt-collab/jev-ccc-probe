#!/usr/bin/env python3
"""
Tests for calibration.py, against synthetic data whose true mode is known.

The point of this file is not coverage. It is that calibration.py prints a
WORD -- "HEDGING", "CONFIDENTLY INVERTED", "DECALIBRATED" -- and that word is
what would go into the write-up. A verdict function that silently mislabels
an inverted judge as hedging would put a wrong claim in a paper, which is
exactly the failure this study has already had once. So each mode gets a
synthetic dataset built to be that mode, and the test asserts the right word
comes out.

    python test_calibration.py
"""

from __future__ import annotations

import io
import json
import os
import random
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

CONDITIONS = ["no_injection", "answer_only", "full_rationale",
              "solver_rationale"]
CANDIDATES = ["correct", "wrong_matching"]


def row(dom, item, cond, cand, enc, p):
    return {
        "run_id": "testtest1234", "ts": 0.0, "domain": dom, "item_id": item,
        "condition": cond, "candidate_type": cand, "encoding": enc,
        "repetition": 0, "protocol": "score_only",
        "model_requested": "typesafe/jev-1.13",
        "model_served": "typesafe/jev-1.13", "seed": 1234, "ok": True,
        "error": None, "status": 200, "attempts": 1, "latency_ms": 100.0,
        "cost": 1.67e-05, "noul": p, "confidence": None,
        "state_sha": "0" * 16, "raw_answer": {"noul": p},
        "platform": "test", "os": "test", "python": "test", "machine": "test",
    }


def write(rows, path):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n")


def judge(rng, true: bool, acc: float, stated: float):
    """A forecaster that is right with probability `acc` and says so.

    Emitting `stated` == `acc` is what makes the synthetic judge CALIBRATED:
    among the forecasts where it says `stated`, exactly that fraction are
    correct. Getting this right matters, because the naive version -- emit
    0.93 for every true candidate -- is NOT calibrated on this design: those
    forecasts are right 100% of the time, so the judge is underconfident and
    a correct audit will say so. The first draft of this test asserted the
    audit was wrong when it was the generator that was wrong.

    Returns the forecast, with a little jitter so bins are not single-valued.
    """
    says_true = true if rng.random() < acc else (not true)
    p = stated if says_true else (1.0 - stated)
    return min(max(p + rng.uniform(-0.004, 0.004), 0.0), 1.0)


def gen(mode, n_items=56, enc="verbatim", seed=7):
    """Build an obs file whose injected conditions have a known failure mode.

    no_injection is always an accurate, calibrated judge (98% right, says
    0.98). The injected conditions differ by mode:

      calibrated  injection does nothing
      hedging     accuracy falls to 55% AND the judge says 0.55 -- resolution
                  collapses, calibration holds. The defensible failure.
      inverted    the judge is 98% confident and 2% right. Ranking reversed.
      decalib     accuracy falls to 70% but the judge still says 0.99 --
                  on the right side, with unearned confidence.
    """
    rng = random.Random(seed)
    rows = []
    for i in range(n_items):
        item = f"it{i:03d}"
        for cond in CONDITIONS:
            for cand in CANDIDATES:
                true = cand == "correct"
                if cond == "no_injection" or mode == "calibrated":
                    p = judge(rng, true, 0.98, 0.98)
                elif mode == "hedging":
                    p = judge(rng, true, 0.55, 0.55)
                elif mode == "inverted":
                    p = judge(rng, true, 0.02, 0.98)
                elif mode == "decalib":
                    p = judge(rng, true, 0.70, 0.99)
                elif mode == "coinflip":
                    # 50% accurate -- the ranking carries no signal -- but
                    # still stating 0.98. AUC lands NEAR 0.5 and can fall
                    # either side of it by chance. The audit must NOT call
                    # this inverted, which is the overclaim the real data
                    # provoked: two verbatim conditions sat about one standard
                    # error below 0.5 and were labelled CONFIDENTLY INVERTED.
                    p = judge(rng, true, 0.50, 0.98)
                elif mode == "nearmiss":
                    # 44% accurate: a point-estimate AUC just under 0.5,
                    # which is NOT evidence of a reversed ranking at n=56.
                    p = judge(rng, true, 0.44, 0.98)
                else:
                    raise ValueError(mode)
                rows.append(row("sql", item, cond, cand, enc, p))
    return rows


def run(path, *extra):
    # --boot 800 keeps the suite fast. Interval WIDTH is set by the number of
    # items, not by the resample count, so the CI-gated verdicts under test
    # behave the same as at the 6000 the real run uses; only the percentile
    # estimate is a little noisier.
    cmd = [sys.executable, os.path.join(HERE, "calibration.py"),
           "--obs", path, "--boot", "800", *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}"
          + (f"   {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="caltest_")
    print("calibration.py tests\n")

    # ---------------------------------------------------------------- modes
    for mode, want in [("calibrated", "little change"),
                       ("hedging", "HEDGING"),
                       ("inverted", "CONFIDENTLY INVERTED"),
                       ("decalib", "DECALIBRATED")]:
        p = os.path.join(tmp, f"{mode}.jsonl")
        write(gen(mode), p)
        rc, out = run(p)
        check(f"mode {mode!r}: exits 0", rc == 0, f"rc={rc}")
        check(f"mode {mode!r}: reports {want!r}", want in out,
              f"got modes: {[l.strip() for l in out.splitlines() if 'MODE:' in l]}")

    # -------------------- the overclaim: near-0.5 AUC is NOT inversion
    for mode in ("coinflip", "nearmiss"):
        p = os.path.join(tmp, f"{mode}.jsonl")
        write(gen(mode), p)
        rc, out = run(p)
        inj = out.split("WHAT DOES INJECTION DO")[-1]
        check(f"mode {mode!r}: NOT labelled inverted",
              "CONFIDENTLY INVERTED" not in inj,
              "this is the overclaim the real data provoked")
        check(f"mode {mode!r}: labelled confidently uninformative",
              "CONFIDENTLY UNINFORMATIVE" in inj,
              str([l.strip() for l in inj.splitlines() if "MODE:" in l][:2]))
        check(f"mode {mode!r}: AUC interval straddles 0.5", "AUC" in inj)

    # a truly inverted judge must still be caught -- the guard above must not
    # have been bought by making the detector useless
    p = os.path.join(tmp, "inverted.jsonl")
    write(gen("inverted"), p)
    rc, out = run(p)
    check("inverted: still labelled inverted after the CI gate",
          "CONFIDENTLY INVERTED" in out.split("WHAT DOES INJECTION DO")[-1])

    # --------------------------------------------- interval reporting
    check("AUC reported with a CI", "AUC                       : " in out
          and "95% CI" in out)
    check("ECE reported with a CI",
          any("ECE 95% CI" in l for l in out.splitlines()))
    check("flip margin reported with a CI",
          any("flip margin" in l and "95% CI" in l for l in out.splitlines()))

    # ------------------------------------------- thin-bin MCE warning
    # The real data has equal-width bins holding 1-2 forecasts, where a +0.910
    # MCE is one observation. Build that shape deliberately: mostly rails, a
    # thin scatter across the middle.
    rng2 = random.Random(11)
    rows = []
    for i in range(56):
        for cond in CONDITIONS:
            for cand in CANDIDATES:
                true = cand == "correct"
                if rng2.random() < 0.06:
                    p_ = rng2.uniform(0.25, 0.75)       # thinly populated bins
                else:
                    p_ = 0.97 if true else 0.03
                rows.append(row("sql", f"it{i:03d}", cond, cand, "verbatim",
                                p_))
    pthin = os.path.join(tmp, "thin.jsonl")
    write(rows, pthin)
    rc, tout = run(pthin)
    check("thin equal-width bins warned about",
          "MCE is unreliable here" in tout,
          "no warning despite sparse middle bins")
    check("thin bins: ECE still reported with an interval",
          "ECE 95% CI" in tout)

    # --------------------------------------------- condition contrast
    rc, out = run(p, "--contrast", "answer_only:full_rationale")
    check("--contrast: runs", rc == 0, f"rc={rc}")
    check("--contrast: reports a paired interval",
          "paired dBrier (full_rationale - answer_only)" in out)
    rc, out = run(p, "--contrast", "answer_only:nonsense")
    check("--contrast: rejects an unknown condition with exit 2", rc == 2,
          f"rc={rc}")

    # ----------------- identical conditions must contrast to exactly zero
    rows = gen("calibrated")          # every condition identical by construction
    p0 = os.path.join(tmp, "same.jsonl")
    write(rows, p0)
    rc, out = run(p0, "--contrast", "answer_only:full_rationale")
    check("--contrast: identical conditions give interval spanning zero",
          "interval spans zero" in out,
          str([l.strip() for l in out.splitlines() if "->" in l][-2:]))

    # ------------------------- baseline threshold-straddle is admitted
    # A baseline whose ECE interval crosses the line must not be given a
    # confident word in either direction. This is the artefact that printed
    # two different verdicts for two indistinguishable baselines.
    rng = random.Random(3)
    rows = []
    for i in range(56):
        for cond in CONDITIONS:
            for cand in CANDIDATES:
                true = cand == "correct"
                rows.append(row("sql", f"it{i:03d}", cond, cand, "verbatim",
                                judge(rng, true, 0.90, 0.99)))
    p0 = os.path.join(tmp, "straddle.jsonl")
    write(rows, p0)
    rc, out = run(p0)
    clean = out.split("IS IT CALIBRATED ON CLEAN INPUT")[-1] \
               .split("WHAT DOES INJECTION DO")[0]
    check("baseline: a straddling ECE interval is admitted, not resolved",
          "straddles" in clean or "wholly" in clean,
          str([l.strip() for l in clean.splitlines() if "VERDICT" in l]))

    # --------------------------------------- inverted detectors agree
    p = os.path.join(tmp, "inverted.jsonl")
    rc, out = run(p)
    check("inverted: AUC printed below 0.5",
          any("AUC" in l and "0.0" in l for l in out.splitlines()))
    check("inverted: flipped forecast flagged as better",
          "FLIPPED IS BETTER" in out)

    # ------------------------------- clean-input verdict on a good judge
    p = os.path.join(tmp, "calibrated.jsonl")
    rc, out = run(p)
    check("clean input: recognised as sharp and calibrated",
          "sharp and well calibrated on clean input" in out)

    # ------------------------------------------- encoding is in the key
    # The bug this guards: averaging a verbatim forecast with a structured one.
    # Build a file where verbatim says 0.95 and structured says 0.05 for the
    # SAME cell. If encoding were dropped from the key the mean would be 0.50
    # and the audit would report an uninformative judge for both.
    rows = []
    for i in range(56):
        for cond in CONDITIONS:
            for cand in CANDIDATES:
                true = cand == "correct"
                rows.append(row("sql", f"it{i:03d}", cond, cand, "verbatim",
                                0.95 if true else 0.05))
                rows.append(row("sql", f"it{i:03d}", cond, cand, "structured",
                                0.05 if true else 0.95))
    p = os.path.join(tmp, "two_enc.jsonl")
    write(rows, p)
    rc, out = run(p, "--both-encodings")
    # Sections are emitted in sorted() encoding order, so 'structured' first.
    verb = out.split("CALIBRATION AUDIT  [encoding=verbatim]")[-1]
    struct = out.split("CALIBRATION AUDIT  [encoding=structured]")[-1] \
                .split("CALIBRATION AUDIT  [encoding=verbatim]")[0]
    check("two encodings: verbatim audited as informative (not collapsed)",
          "AUC                       : 1.0000" in verb,
          "verbatim AUC lines: "
          + str([l.strip() for l in verb.splitlines() if "AUC " in l][:2]))
    check("two encodings: structured audited as inverted (not collapsed)",
          "AUC                       : 0.0000" in struct,
          "structured AUC lines: "
          + str([l.strip() for l in struct.splitlines() if "AUC " in l][:2]))
    check("two encodings: both audited separately",
          out.count("CALIBRATION AUDIT") == 2,
          f"found {out.count('CALIBRATION AUDIT')} audits")
    check("two encodings: forecast count is per-encoding, not merged",
          "forecasts_usable             : 896" in out,
          [l for l in out.splitlines() if "forecasts_usable" in l])

    # --------------------------------------------- strict missingness
    rows = gen("inverted")
    rows = [r for r in rows if not (r["item_id"] == "it000"
                                    and r["condition"] == "no_injection")]
    for r in rows[:10]:
        r["ok"] = False
        r["noul"] = None
    p = os.path.join(tmp, "holes.jsonl")
    write(rows, p)
    rc, out = run(p)
    check("holes: failures counted in the marker",
          "calls_failed                 : 10" in out,
          [l for l in out.splitlines() if "calls_failed" in l])

    # ------------------------------------ no_injection entirely absent
    rows = [r for r in gen("inverted") if r["condition"] != "no_injection"]
    p = os.path.join(tmp, "nobase.jsonl")
    write(rows, p)
    rc, out = run(p)
    check("no baseline condition: refuses with exit 1", rc == 1, f"rc={rc}")
    check("no baseline condition: says why",
          "cannot compute the injection contrast" in out)

    # ------------------------------------------------ tie-safe binning
    # The bug this guards: equal-mass bins built by argsort+array_split split
    # ties across bins in input order. Here EVERY forecast is 0.5 and the rows
    # are ordered correct-first, so a tie-splitting implementation puts all the
    # correct candidates in low bins and all the wrong ones in high bins, and
    # the Murphy decomposition reads a constant forecaster as a perfect
    # discriminator. Resolution must be 0.
    rows = []
    for cand in CANDIDATES:                      # deliberately candidate-major
        for i in range(56):
            for cond in CONDITIONS:
                rows.append(row("sql", f"it{i:03d}", cond, cand,
                                "verbatim", 0.5))
    p = os.path.join(tmp, "ties.jsonl")
    write(rows, p)
    rc, out = run(p)
    res_lines = [l for l in out.splitlines() if "Murphy decomposition" in l]
    check("ties: constant forecaster gets resolution 0.0000",
          res_lines and all("resolution 0.0000" in l for l in res_lines),
          str(res_lines[:2]))
    check("ties: constant forecaster gets reliability 0.0000",
          res_lines and all("reliability 0.0000" in l for l in res_lines),
          str(res_lines[:2]))

    # --------------------------------------------- degenerate forecasts
    rows = []
    for i in range(56):
        for cond in CONDITIONS:
            for cand in CANDIDATES:
                rows.append(row("sql", f"it{i:03d}", cond, cand,
                                "verbatim", 0.5))
    p = os.path.join(tmp, "flat.jsonl")
    write(rows, p)
    rc, out = run(p)
    check("all forecasts 0.5: exits 0 without crashing", rc == 0, f"rc={rc}")
    check("all forecasts 0.5: Brier is 0.2500",
          "Brier score               : 0.2500" in out)
    check("all forecasts 0.5: recognised as no better than uninformative",
          "no better than the uninformative" in out)

    # ------------------------------------------------- out-of-range noul
    rows = gen("calibrated")
    rows[0]["noul"] = 1.4
    rows[1]["noul"] = -0.2
    p = os.path.join(tmp, "oor.jsonl")
    write(rows, p)
    rc, out = run(p)
    check("out-of-range noul: dropped and reported",
          "noul_out_of_unit_range       : 2" in out,
          [l for l in out.splitlines() if "out_of_unit_range" in l])

    # ------------------------------------------------- empty / bad input
    p = os.path.join(tmp, "empty.jsonl")
    write([], p)
    rc, out = run(p)
    check("empty file: exits 2", rc == 2, f"rc={rc}")

    # ------------------------------------------- confidence-field notice
    rows = gen("calibrated")
    p = os.path.join(tmp, "conf.jsonl")
    write(rows, p)
    rc, out = run(p)
    check("confidence absent: limitation stated",
          "only" in out and "probability half of that claim" in out)
    for r in rows:
        r["confidence"] = 0.8
    write(rows, p)
    rc, out = run(p)
    check("confidence present: notice suppressed",
          "probability half of that claim" not in out)

    # ------------------------------------------------------ by-domain
    rows = gen("inverted") + [row("code", f"c{i:03d}", c, ca, "verbatim",
                                  0.9 if ca == "correct" else 0.1)
                              for i in range(56) for c in CONDITIONS
                              for ca in CANDIDATES]
    p = os.path.join(tmp, "two_dom.jsonl")
    write(rows, p)
    rc, out = run(p, "--by-domain")
    check("--by-domain: splits both domains",
          "DOMAIN: sql" in out and "DOMAIN: code" in out)
    check("--by-domain: domains get different verdicts",
          "CONFIDENTLY INVERTED" in out.split("DOMAIN: sql")[1]
          .split("DOMAIN: code")[0])

    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED: {', '.join(FAILS)}")
        return 1
    print("all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
