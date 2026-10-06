#!/usr/bin/env python3
"""
Analysis for the attribution decomposition. Implements PREREG_jev_attribution.md
§3 and nothing else as confirmatory.

DIRECTION CONVENTION, inherited from the harness:

    D(cell)    = mean noul(correct candidate) - mean noul(wrong_matching)
    harm(cell) = D(no_injection) - D(cell)        positive = MORE capture

Every primary contrast is a difference of harms, and the baseline cancels:

    harm(A) - harm(B) == D(B) - D(A)

which this module asserts numerically rather than trusting the algebra, because a
sign error here would invert every conclusion in the study. The baseline is still
computed and gated on, because a capture figure measured against a judge that was
not judging is not a measurement.

Support, per prereg: the 95% item-clustered bootstrap CI excludes 0 in the
predicted direction. Holm-adjusted p is reported alongside as a stricter view.

    python analyse_attribution.py --obs _obs_attr.jsonl
    python analyse_attribution.py --obs _obs_attr.jsonl --by-domain
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

import jevlib as J

N_BOOT = 6000
BOOT_SEED = 0
BASELINE_FLOOR = 0.10          # below this, capture ratios are not reported
CANDIDATES = ("correct", "wrong_matching")

BASE = "no_injection"
NEUTRAL = "id:neutral"
NEUTRAL_R = "id:neutral+restate"
SOLVER = "id:solver"
SOLVER_R = "id:solver+restate"
SEALED = "id:sealed_solver"
COMP_BUNDLE = "comp:solver+verified+restate"
LEGACY = "legacy:solver_bundle"
REL = {c: f"rel:{c}" for c in ("verified", "unverified", "possibly_erroneous")}


def h(t: str) -> None:
    print("\n" + "=" * 76)
    print(t)
    print("=" * 76)


def sub(t: str) -> None:
    print("\n" + t)
    print("-" * min(len(t), 76))


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #

def assemble(rows: list[dict]):
    """-> D[(domain, item, cell, encoding)], plus completeness bookkeeping.

    Averages repetitions, then forms D per cell. Fail-closed exclusion is applied
    by the caller, which needs the expected cell set per domain.
    """
    vals = defaultdict(list)
    attempted = failed = nulls = 0
    for r in rows:
        attempted += 1
        if not r.get("ok"):
            failed += 1
            continue
        if r.get("noul") is None:
            nulls += 1
            continue
        vals[(r["domain"], r["item_id"], r["cell"], r["encoding"],
              r["candidate_type"])].append(float(r["noul"]))

    mean = {k: statistics.mean(v) for k, v in vals.items()}
    reps = {k: len(v) for k, v in vals.items()}

    D = {}
    for (dom, item, cell, enc, cand) in list(mean):
        if cand != "correct":
            continue
        kc = (dom, item, cell, enc, "correct")
        kw = (dom, item, cell, enc, "wrong_matching")
        if kw in mean:
            D[(dom, item, cell, enc)] = mean[kc] - mean[kw]

    marker = {"calls_attempted": attempted, "calls_failed": failed,
              "answers_null": nulls,
              "reps_min": min(reps.values()) if reps else 0,
              "reps_max": max(reps.values()) if reps else 0}
    return D, marker


def usable_items(D: dict, expected: dict[str, set], encodings: list[str]):
    """Fail-closed: an item survives only if every expected cell is present in
    every encoding. The prereg is explicit that scattered failures cost whole
    items rather than being silently complete-cased."""
    have = defaultdict(set)
    for (dom, item, cell, enc) in D:
        have[(dom, item)].add((cell, enc))
    usable, excluded = [], {}
    for (dom, item), got in sorted(have.items()):
        want = {(c, e) for c in expected[dom] for e in encodings}
        missing = want - got
        if missing:
            excluded[(dom, item)] = missing
        else:
            usable.append((dom, item))
    return usable, excluded


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

def boot_ci(per_item: list[float], n_boot: int = N_BOOT, seed: int = BOOT_SEED):
    a = np.asarray([x for x in per_item if not math.isnan(x)], dtype=float)
    if a.size < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, (n_boot, a.size))
    draws = a[idx].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def describe(per_item: list[float], label: str, predicted: str) -> dict:
    """predicted in {'>0', '<0', '>=0', 'two-sided', '~0'}."""
    a = np.asarray([x for x in per_item if not math.isnan(x)], dtype=float)
    n = a.size
    if n < 3:
        print(f"  {label}: too few items ({n})")
        return {"label": label, "n": n, "p": float("nan"),
                "mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "predicted": predicted, "supported": False}
    m = float(a.mean())
    lo, hi = boot_ci(list(a))
    sd = float(a.std(ddof=1))
    dz = m / sd if sd > 0 else float("nan")
    t = stats.ttest_1samp(a, 0.0)
    try:
        w = stats.wilcoxon(a).pvalue if np.any(a != 0) else float("nan")
    except ValueError:
        w = float("nan")

    if predicted in (">0", ">=0"):
        supported = (not math.isnan(lo)) and lo > 0
    elif predicted == "<0":
        supported = (not math.isnan(hi)) and hi < 0
    elif predicted == "~0":
        supported = (not math.isnan(lo)) and lo <= 0 <= hi       # CI covers 0
    else:                                                        # two-sided
        supported = (not math.isnan(lo)) and (lo > 0 or hi < 0)

    # The verdict WORD matters more than the number here, so each prediction
    # type gets its own, rather than overloading "SUPPORTED" onto a two-sided
    # test or onto a null that was merely not rejected.
    if predicted in (">0", ">=0", "<0"):
        verdict = "SUPPORTED" if supported else "not supported"
    elif predicted == "two-sided":
        if math.isnan(lo) or (lo <= 0 <= hi):
            verdict = "UNRESOLVED (CI spans 0)"
        else:
            verdict = ("RESOLVED, negative -- the claim REDUCES capture"
                       if hi < 0 else
                       "RESOLVED, positive -- the claim INCREASES capture")
    else:                                            # "~0", an additivity check
        bound = max(abs(lo), abs(hi)) if not math.isnan(lo) else float("nan")
        verdict = (f"consistent with additivity to within +/-{bound:.4f} "
                   f"(CI covers 0; absence of evidence, not proof)"
                   if supported else
                   "NON-ADDITIVE (CI excludes 0) -- the decomposition is "
                   "incomplete")

    print(f"  {label}")
    print(f"    n items {n:>3}   mean {m:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]"
          f"   dz {dz:+.3f}")
    print(f"    paired t p {t.pvalue:.3g}   Wilcoxon p {w:.3g}"
          f"   predicted {predicted}")
    print(f"    ->  {verdict}")
    return {"label": label, "n": n, "p": float(t.pvalue), "mean": m,
            "lo": lo, "hi": hi, "predicted": predicted, "supported": supported}


def holm(results: list[dict]) -> None:
    """Holm-Bonferroni over the primary family."""
    live = [r for r in results if not math.isnan(r["p"])]
    if not live:
        return
    k = len(live)
    order = sorted(live, key=lambda r: r["p"])
    print(f"\n  Holm-Bonferroni over the {k} primary tests actually computed")
    print("  (the prereg says 'across the five'; contrasts 1, 2 and 4 are each")
    print("   computed in two encodings, which that wording did not resolve, so")
    print("   the family is taken as all computed primaries -- the stricter read.)")
    print(f"\n  {'contrast':<52} {'p':>10} {'adj':>10}  verdict")
    running = 0.0
    for i, r in enumerate(order):
        adj = min(1.0, max(running, (k - i) * r["p"]))
        running = adj
        mark = "reject H0" if adj < 0.05 else "retain H0"
        print(f"  {r['label'][:52]:<52} {r['p']:>10.3g} {adj:>10.3g}  {mark}")


# --------------------------------------------------------------------------- #
# Contrast plumbing
# --------------------------------------------------------------------------- #

def per_item_D(D, usable, cell, enc) -> dict:
    return {(d, i): D[(d, i, cell, enc)]
            for (d, i) in usable if (d, i, cell, enc) in D}


def harm_diff(D, usable, cell_a, cell_b, enc) -> list[float]:
    """harm(cell_b) - harm(cell_a)  ==  D(cell_a) - D(cell_b), per item.

    Read it as: how much more capture does cell_b produce than cell_a.
    """
    A, B = per_item_D(D, usable, cell_a, enc), per_item_D(D, usable, cell_b, enc)
    keys = sorted(set(A) & set(B))
    return [A[k] - B[k] for k in keys]


def check_harm_identity(D, usable, enc) -> bool:
    """Assert harm(A)-harm(B) == D(B)-D(A) numerically, on real data.

    The baseline cancels in every primary contrast. That is convenient and it is
    also exactly the kind of algebra that silently inverts a conclusion if the
    convention drifts, so it is checked rather than assumed.
    """
    base = per_item_D(D, usable, BASE, enc)
    A = per_item_D(D, usable, NEUTRAL, enc)
    B = per_item_D(D, usable, NEUTRAL_R, enc)
    keys = sorted(set(base) & set(A) & set(B))
    if not keys:
        return True
    lhs = [(base[k] - B[k]) - (base[k] - A[k]) for k in keys]   # harm diff
    rhs = [A[k] - B[k] for k in keys]                            # D diff
    return max(abs(x - y) for x, y in zip(lhs, rhs)) < 1e-12


# --------------------------------------------------------------------------- #

def baseline_gate(D, usable, encodings, label="") -> dict:
    sub(f"BASELINE COMPETENCE{(' ' + label) if label else ''}")
    out = {}
    print(f"  {'encoding':<12} {'domain':<7} {'n':>4} {'mean D':>9} "
          f"{'95% CI':>22}")
    for enc in encodings:
        for dom in sorted({d for d, _ in usable}):
            vals = [D[(d, i, BASE, enc)] for (d, i) in usable
                    if d == dom and (d, i, BASE, enc) in D]
            if not vals:
                continue
            lo, hi = boot_ci(vals)
            flag = "  <-- BELOW FLOOR" if statistics.mean(vals) < BASELINE_FLOOR else ""
            print(f"  {enc:<12} {dom:<7} {len(vals):>4} "
                  f"{statistics.mean(vals):>+9.4f} "
                  f"[{lo:+.4f}, {hi:+.4f}]{flag}")
            out[(enc, dom)] = statistics.mean(vals)
    weak = [k for k, v in out.items() if v < BASELINE_FLOOR]
    if weak:
        print(f"\n  WARNING: {len(weak)} domain x encoding cell(s) below the "
              f"{BASELINE_FLOOR} floor.")
        print("  Capture contrasts in those cells are measured against a judge "
              "that barely judges.")
    return out


def primaries(D, usable, encodings) -> list[dict]:
    res: list[dict] = []

    h("PRIMARY CONTRASTS (confirmatory, prereg §3)")
    print("  harm = D(no_injection) - D(cell); positive = more capture.")
    print("  Each contrast below is harm(second) - harm(first).")

    for enc in encodings:
        ok = check_harm_identity(D, usable, enc)
        print(f"  [{enc}] harm-algebra identity holds: {ok}")
        if not ok:
            print("  ABORT-WORTHY: the harm convention does not reduce as "
                  "expected.", file=sys.stderr)

    sub("1. Restatement effect   (restating the wrong value, neutral label)")
    for enc in encodings:
        res.append(describe(harm_diff(D, usable, NEUTRAL, NEUTRAL_R, enc),
                            f"1. restatement [{enc}]", ">0"))

    sub("2. Identity effect, isolated   (neutral -> solver, nothing else moves)")
    print("  This is the contrast finding 2 of the main note intended to measure.")
    for enc in encodings:
        res.append(describe(harm_diff(D, usable, NEUTRAL, SOLVER, enc),
                            f"2. identity [{enc}]", ">=0"))

    sub("3. Encoding interaction on the identity effect   (the headline)")
    print("  [harm_struct(solver)-harm_struct(neutral)] - "
          "[harm_verb(solver)-harm_verb(neutral)]")
    v = dict(zip(sorted(set(per_item_D(D, usable, NEUTRAL, "verbatim")) &
                        set(per_item_D(D, usable, SOLVER, "verbatim"))),
                 harm_diff(D, usable, NEUTRAL, SOLVER, "verbatim")))
    s = dict(zip(sorted(set(per_item_D(D, usable, NEUTRAL, "structured")) &
                        set(per_item_D(D, usable, SOLVER, "structured"))),
                 harm_diff(D, usable, NEUTRAL, SOLVER, "structured")))
    keys = sorted(set(v) & set(s))
    res.append(describe([s[k] - v[k] for k in keys],
                        "3. identity x encoding interaction", ">0"))

    sub("4. Reliability dose-response   ('may contain errors' vs no claim)")
    print("  Two-sided by design: <0 means the judge treats provenance as")
    print("  reliability INFORMATION and discounts a flagged analysis; >=0 means")
    print("  it treats it as an AUTHORITY signal only.")
    for enc in encodings:
        res.append(describe(
            harm_diff(D, usable, NEUTRAL, REL["possibly_erroneous"], enc),
            f"4. possibly_erroneous [{enc}]", "two-sided"))
    print("\n  ordered levels (exploratory, for the dose shape):")
    for enc in encodings:
        row = []
        for name, cell in (("none", NEUTRAL), ("verified", REL["verified"]),
                           ("unverified", REL["unverified"]),
                           ("poss_err", REL["possibly_erroneous"])):
            vals = list(per_item_D(D, usable, cell, enc).values())
            row.append(f"{name} {statistics.mean(vals):+.3f}" if vals else f"{name} -")
        print(f"    [{enc:<10}] mean D by claim:  " + "   ".join(row))

    sub("5. Decomposition additivity   (composed bundle vs the frozen bundle)")
    print("  sql+code only: arith has no hand-written bundle to bridge to.")
    print("  A CI covering 0 means the composition reproduces the original to")
    print("  within the interval's width. That is an equivalence claim bounded by")
    print("  this n, not a proof of additivity -- the verdict line states the bound.")
    sub_usable = [(d, i) for (d, i) in usable if d in ("sql", "code")]
    vals = []
    for enc in encodings:
        vals += harm_diff(D, sub_usable, COMP_BUNDLE, LEGACY, enc)
    res.append(describe(vals, "5. additivity (both encodings pooled)", "~0"))

    holm(res)
    return res


def secondaries(D, usable, encodings) -> None:
    h("SECONDARY / EXPLORATORY (no strong claims; multiplicity uncorrected)")

    sub("All identities vs neutral, restatement absent")
    print(f"  {'identity':<16} " + "  ".join(f"{e:>24}" for e in encodings))
    for ident in ("unlabelled", "candidate", "reviewer", "solver", "sealed_solver"):
        cell = f"id:{ident}"
        cols = []
        for enc in encodings:
            d = harm_diff(D, usable, NEUTRAL, cell, enc)
            if len(d) < 3:
                cols.append(f"{'-':>24}")
                continue
            lo, hi = boot_ci(d)
            cols.append(f"{statistics.mean(d):+.4f} [{lo:+.3f},{hi:+.3f}]".rjust(24))
        print(f"  {ident:<16} " + "  ".join(cols))

    sub("Status beyond identity: sealed_solver vs solver")
    for enc in encodings:
        d = harm_diff(D, usable, SOLVER, SEALED, enc)
        if len(d) >= 3:
            lo, hi = boot_ci(d)
            print(f"  [{enc:<10}] {statistics.mean(d):+.4f} "
                  f"95% CI [{lo:+.4f}, {hi:+.4f}]")

    sub("Restatement x identity (does the restatement cost more under a label?)")
    for enc in encodings:
        a = harm_diff(D, usable, NEUTRAL, NEUTRAL_R, enc)
        b = harm_diff(D, usable, SOLVER, SOLVER_R, enc)
        n = min(len(a), len(b))
        if n >= 3:
            diff = [y - x for x, y in zip(a[:n], b[:n])]
            lo, hi = boot_ci(diff)
            print(f"  [{enc:<10}] solver-restatement minus neutral-restatement: "
                  f"{statistics.mean(diff):+.4f} [{lo:+.4f}, {hi:+.4f}]")

    sub("Mean D by cell and encoding")
    cells = sorted({c for (_, _, c, _) in D})
    print(f"  {'cell':<30} " + "  ".join(f"{e:>12}" for e in encodings))
    for cell in cells:
        cols = []
        for enc in encodings:
            vals = list(per_item_D(D, usable, cell, enc).values())
            cols.append(f"{statistics.mean(vals):>+12.4f}" if vals else f"{'-':>12}")
        print(f"  {cell:<30} " + "  ".join(cols))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--obs", default="_obs_attr.jsonl")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--repo", default=None,
                    help="repo path, to import the expected cell sets. Without "
                         "it the expected sets are inferred from the data, which "
                         "cannot detect a cell that is missing everywhere.")
    ap.add_argument("--by-domain", action="store_true")
    ap.add_argument("--secondary", action="store_true",
                    help="also print the exploratory section")
    args = ap.parse_args()

    rows = J.read_obs(args.obs)
    if args.run_id:
        rows = [r for r in rows if r.get("run_id") == args.run_id]
    rows = [r for r in rows
            if r.get("experiment") == "attribution_decomposition" or "cell" in r]
    if not rows:
        print(f"no attribution rows in {args.obs}", file=sys.stderr)
        return 2

    encodings = sorted({r["encoding"] for r in rows})
    models = sorted({r.get("model_served") for r in rows if r.get("model_served")})
    print(f"rows {len(rows)}   file {args.obs}")
    print(f"runs  {', '.join(sorted({str(r.get('run_id')) for r in rows}))}")
    print(f"model {', '.join(models) or 'n/a'}")
    print(f"cost  ${sum(float(r.get('cost') or 0) for r in rows):.6f}")
    if any("SIMULATED" in (m or "") for m in models):
        print("\n*** SIMULATED DATA -- not a real measurement ***")

    D, marker = assemble(rows)

    # expected cell sets: authoritative from the stimulus module when available
    if args.repo:
        import attribution_stimuli as A
        A.add_repo_to_path(args.repo)
        expected = {d: {c.name for c in A.cells_for(d)}
                    for d in ("sql", "code", "arith")}
        src = "attribution_stimuli.cells_for()"
    else:
        expected = defaultdict(set)
        for (dom, _i, cell, _e) in D:
            expected[dom].add(cell)
        src = "inferred from the data (pass --repo for the authoritative set)"

    usable, excluded = usable_items(D, expected, encodings)

    h("COMPLETENESS MARKER")
    for k, v in marker.items():
        print(f"  {k:<18} : {v}")
    print(f"  expected cells     : "
          f"{ {d: len(s) for d, s in expected.items()} }  [{src}]")
    print(f"  items usable       : {len(usable)}")
    print(f"  items excluded     : {len(excluded)}")
    for (d, i), miss in list(excluded.items())[:8]:
        print(f"    {d}/{i}: {len(miss)} missing (cell, encoding) pair(s)")
    by_dom = defaultdict(int)
    for d, _ in usable:
        by_dom[d] += 1
    print(f"  usable by domain   : {dict(by_dom)}")
    if len(usable) < 3:
        print("\nToo few complete items to analyse.", file=sys.stderr)
        return 1

    baseline_gate(D, usable, encodings)
    primaries(D, usable, encodings)
    if args.secondary:
        secondaries(D, usable, encodings)

    if args.by_domain:
        for dom in sorted(by_dom):
            print("\n")
            print("#" * 76)
            print(f"#  DOMAIN: {dom}   (exploratory -- below the power floor)")
            print("#" * 76)
            sub_u = [(d, i) for (d, i) in usable if d == dom]
            baseline_gate(D, sub_u, encodings, label=f"[{dom}]")
            primaries(D, sub_u, encodings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
