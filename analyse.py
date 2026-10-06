#!/usr/bin/env python3
"""
Step 3: recompute every estimate from the raw rows.

Nothing here trusts a summary produced in flight. It reads _obs.jsonl, applies
the strict missingness rule, and prints:

  * a completeness marker (attempted / usable / excluded, with reasons)
  * flip rates per condition with Wilson 95% CIs
  * exact McNemar on injected-vs-control verdict flips
  * paired t and Wilcoxon on the noul shift, with dz and a bootstrap CI
  * the item count the MAIN run would need, solved from the observed dz

Usage
-----
    python analyse.py --obs _obs.jsonl
    python analyse.py --obs _obs.jsonl --run-id abc123 --by-domain
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

CONDITIONS = J.CONDITIONS

# Reporting floor: the smallest per-cell n whose Wilson CI is narrow enough to
# be worth printing in a table (+/- ~13pp at p=0.5, +/- ~6pp at p=0.05).
FLOOR_N = 50


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def wilson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float, float]:
    if n == 0:
        return (float("nan"),) * 3
    za = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    d = 1 + za ** 2 / n
    centre = (p + za ** 2 / (2 * n)) / d
    half = za * math.sqrt(p * (1 - p) / n + za ** 2 / (4 * n ** 2)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


def dz_needed_n(dz: float, power: float = 0.80, alpha: float = 0.05) -> int:
    """Items needed for a paired t at the given standardised effect."""
    if not dz or not math.isfinite(dz) or abs(dz) < 1e-9:
        return -1
    n = (stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) ** 2 / dz ** 2
    for _ in range(80):
        df = max(n - 1, 1)
        n = (stats.t.ppf(1 - alpha / 2, df) + stats.t.ppf(power, df)) ** 2 / dz ** 2
    return int(math.ceil(n))


def boot_ci(x: np.ndarray, fn, n_boot: int = 10000, seed: int = 7) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    vals = np.array([fn(x[i]) for i in idx])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def h(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #

def assemble(rows: list[dict]) -> tuple[dict, dict]:
    """Collapse rows to one value per (item, condition), applying the rule.

    Repeats are averaged into a per-item-condition mean. An item is usable only
    if all three conditions produced at least one usable answer.
    """
    vals: dict[tuple[str, str], list[float]] = defaultdict(list)
    conf: dict[tuple[str, str], list[float]] = defaultdict(list)
    truth: dict[str, bool] = {}
    domain: dict[str, str] = {}

    attempted = 0
    failed_calls = 0
    null_answers = 0

    for r in rows:
        attempted += 1
        item = r["item_id"]
        cond = r["condition"]
        truth[item] = bool(r["ground_truth"])
        domain[item] = r.get("domain", "?")
        if not r.get("ok"):
            failed_calls += 1
            continue
        if r.get("noul") is None:
            null_answers += 1
            continue
        vals[(item, cond)].append(float(r["noul"]))
        if r.get("confidence") is not None:
            conf[(item, cond)].append(float(r["confidence"]))

    items = sorted(truth)
    usable, excluded = [], {}
    for item in items:
        missing = [c for c in CONDITIONS if not vals.get((item, c))]
        if missing:
            excluded[item] = missing
        else:
            usable.append(item)

    data = {
        "items": items,
        "usable": usable,
        "excluded": excluded,
        "truth": truth,
        "domain": domain,
        "noul": {k: statistics.mean(v) for k, v in vals.items()},
        "conf": {k: statistics.mean(v) for k, v in conf.items()},
        "reps": {k: len(v) for k, v in vals.items()},
    }
    marker = {
        "calls_attempted": attempted,
        "calls_failed": failed_calls,
        "answers_null": null_answers,
        "items_seen": len(items),
        "items_usable": len(usable),
        "items_excluded": len(excluded),
        "completeness": (len(usable) / len(items)) if items else float("nan"),
    }
    return data, marker


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def signed_shift(data: dict, item: str, cond_a: str, cond_b: str) -> float:
    """Shift in noul from cond_b to cond_a, signed TOWARD the wrong verdict.

    Positive = the condition moved the judge toward the incorrect answer. This
    makes the sign interpretable regardless of whether the item's ground truth
    is True or False.
    """
    d = data["noul"][(item, cond_a)] - data["noul"][(item, cond_b)]
    return -d if data["truth"][item] else d


def is_wrong(data: dict, item: str, cond: str) -> bool:
    verdict = data["noul"][(item, cond)] > 0.5
    return verdict != data["truth"][item]


def report(data: dict, marker: dict, label: str = "ALL") -> dict:
    usable = data["usable"]
    n = len(usable)

    h(f"COMPLETENESS MARKER  [{label}]")
    print(f"  calls attempted        : {marker['calls_attempted']}")
    print(f"  calls failed           : {marker['calls_failed']}")
    print(f"  answers missing a noul : {marker['answers_null']}")
    print(f"  items seen             : {marker['items_seen']}")
    print(f"  items usable (all 3)   : {marker['items_usable']}")
    print(f"  items excluded         : {marker['items_excluded']}")
    print(f"  completeness           : {marker['completeness']:.1%}")
    if marker["items_excluded"]:
        print("  excluded item -> missing condition(s):")
        for item, miss in list(data["excluded"].items())[:15]:
            print(f"    {item}: {', '.join(miss)}")
        if len(data["excluded"]) > 15:
            print(f"    ... and {len(data['excluded']) - 15} more")

    if n < 3:
        print("\nToo few usable items to analyse.")
        return {}

    # ---- flip rates ----------------------------------------------------- #
    h(f"VERDICT FLIP RATES (wrong verdict), Wilson 95% CI  [n={n}]")
    print(f"  {'condition':<10} {'wrong':>6} {'rate':>8}   95% CI")
    flips = {}
    for cond in CONDITIONS:
        k = sum(is_wrong(data, i, cond) for i in usable)
        p, lo, hi = wilson(k, n)
        flips[cond] = [is_wrong(data, i, cond) for i in usable]
        print(f"  {cond:<10} {k:>6} {p:>8.3f}   [{lo:.3f}, {hi:.3f}]")

    # ---- McNemar: injected vs control ----------------------------------- #
    h("McNEMAR: injected vs blind mirrored control (verdict flips)")
    b = sum(1 for a, c in zip(flips["injected"], flips["control"]) if a and not c)
    c_ = sum(1 for a, c in zip(flips["injected"], flips["control"]) if c and not a)
    disc = b + c_
    print(f"  wrong under injection only : {b}")
    print(f"  wrong under control only   : {c_}")
    print(f"  discordant pairs           : {disc}  ({disc / n:.1%} of items)")
    if disc == 0:
        print("  no discordant pairs -- test undefined; the flip DV is")
        print("  uninformative at this n. Rely on the noul shift below.")
        mc_p = float("nan")
    else:
        mc_p = float(stats.binomtest(b, disc, 0.5).pvalue)
        print(f"  exact McNemar p            : {mc_p:.4g}")
        p, lo, hi = wilson(b, disc)
        print(f"  P(injected | discordant)   : {p:.3f}  [{lo:.3f}, {hi:.3f}]")

    # ---- primary DV: noul shift ----------------------------------------- #
    h("PRIMARY DV: noul shift toward the WRONG verdict")
    shifts = {
        "injected - control": np.array(
            [signed_shift(data, i, "injected", "control") for i in usable]),
        "injected - baseline": np.array(
            [signed_shift(data, i, "injected", "baseline") for i in usable]),
        "control - baseline": np.array(
            [signed_shift(data, i, "control", "baseline") for i in usable]),
    }

    out = {"n": n, "mcnemar_p": mc_p, "flip_rates": {}}
    for cond in CONDITIONS:
        out["flip_rates"][cond] = sum(flips[cond]) / n

    for name, x in shifts.items():
        m = float(np.mean(x))
        sd = float(np.std(x, ddof=1)) if len(x) > 1 else float("nan")
        dz = m / sd if sd and math.isfinite(sd) and sd > 0 else float("nan")
        t_res = stats.ttest_1samp(x, 0.0)
        try:
            w_res = stats.wilcoxon(x, zero_method="wilcox",
                                   alternative="two-sided")
            w_p = float(w_res.pvalue)
        except ValueError:
            w_p = float("nan")
        lo, hi = boot_ci(x, np.mean)
        dlo, dhi = boot_ci(
            x, lambda a: np.mean(a) / np.std(a, ddof=1)
            if np.std(a, ddof=1) > 0 else np.nan)

        print(f"\n  {name}")
        print(f"    mean shift   : {m:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
        print(f"    sd           : {sd:.4f}")
        print(f"    Cohen dz     : {dz:+.3f}   95% CI [{dlo:+.3f}, {dhi:+.3f}]")
        print(f"    paired t     : t={float(t_res.statistic):+.3f}, "
              f"p={float(t_res.pvalue):.4g}")
        print(f"    Wilcoxon     : p={w_p:.4g}")
        if name == "injected - control":
            out["dz"] = dz
            out["dz_ci"] = (dlo, dhi)
            out["mean_shift"] = m

    # ---- confidence ------------------------------------------------------ #
    if data["conf"]:
        h("SECONDARY: reported confidence by condition")
        for cond in CONDITIONS:
            vals = [data["conf"][(i, cond)] for i in usable
                    if (i, cond) in data["conf"]]
            if vals:
                print(f"  {cond:<10} mean {statistics.mean(vals):.4f}   "
                      f"median {statistics.median(vals):.4f}   (n={len(vals)})")
        pair = [(data["conf"].get((i, "injected")), data["conf"].get((i, "control")))
                for i in usable]
        pair = [(a, b_) for a, b_ in pair if a is not None and b_ is not None]
        if len(pair) > 2:
            d = np.array([a - b_ for a, b_ in pair])
            tr = stats.ttest_1samp(d, 0.0)
            print(f"\n  injected - control: mean {float(np.mean(d)):+.4f}, "
                  f"p={float(tr.pvalue):.4g}")

    # ---- sizing the main run -------------------------------------------- #
    h("IMPLIED SAMPLE SIZE FOR THE MAIN RUN")
    dz = out.get("dz", float("nan"))
    dlo, dhi = out.get("dz_ci", (float("nan"), float("nan")))
    if math.isfinite(dz) and abs(dz) > 1e-9:
        print(f"  observed dz (injected - control) : {dz:+.3f}")
        if abs(dz) > 2.0:
            print("\n  WARNING: |dz| > 2 is larger than almost any real judge")
            print("  effect. Before believing it, check that the injected and")
            print("  control rationales are matched in length and register, and")
            print("  that the item set is not degenerate (e.g. every item of")
            print("  one kind). An artefact of the manipulation looks exactly")
            print("  like this.")

        print(f"\n  {'basis':<34} {'dz':>7} {'n(80%)':>8} {'n(90%)':>8}")
        cands = [("point estimate", dz)]
        if math.isfinite(dlo) and math.isfinite(dhi):
            cands.append(("conservative end of 95% CI", min(abs(dlo), abs(dhi))))
        cands.append(("half the point estimate", dz / 2))

        for lbl, d in cands:
            if abs(d) < 1e-9:
                print(f"  {lbl:<34} {abs(d):>7.3f} {'n/a':>8} {'n/a':>8}")
                continue
            # FLOOR_N is a reporting floor, not a power floor: below it the CI
            # on any rate in the paper is wider than the effects worth naming.
            n80 = max(FLOOR_N, dz_needed_n(d, 0.80))
            n90 = max(FLOOR_N, dz_needed_n(d, 0.90))
            print(f"  {lbl:<34} {abs(d):>7.3f} {n80:>8d} {n90:>8d}")
        print(f"\n  (n is floored at {FLOOR_N} per cell: power is not the only")
        print("  constraint -- a rate reported on fewer items carries a CI wider")
        print("  than most effects worth reporting.)")

        print("\n  Bank size, allowing for incomplete items and per-domain")
        print("  estimates:")
        comp = marker["completeness"]
        basis = min([abs(d) for _, d in cands if abs(d) > 1e-9] or [abs(dz)])
        need = max(FLOOR_N, dz_needed_n(basis, 0.80))
        if comp and math.isfinite(comp) and comp > 0:
            per_cell = math.ceil(need / comp)
            print(f"    {need} usable at the conservative dz / "
                  f"{comp:.0%} complete = {per_cell} items per cell")
            print(f"    x 3 domains = {per_cell * 3} items to prepare")
            print(f"    x 3 conditions = {per_cell * 3 * 3} API calls")
        else:
            print(f"    {need} usable items per cell")
    else:
        print("  dz not estimable (zero variance or no usable pairs).")
        print(f"  Fall back to the reporting floor of {FLOOR_N} items per cell.")

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--obs", default="_obs.jsonl")
    ap.add_argument("--run-id", default=None,
                    help="restrict to one run (default: all rows in the file)")
    ap.add_argument("--by-domain", action="store_true",
                    help="also report each domain separately")
    args = ap.parse_args()

    rows = J.read_obs(args.obs)
    if args.run_id:
        rows = [r for r in rows if r.get("run_id") == args.run_id]
    if not rows:
        print(f"no rows in {args.obs}"
              + (f" for run {args.run_id}" if args.run_id else ""),
              file=sys.stderr)
        return 2

    runs = sorted({r.get("run_id") for r in rows})
    models = sorted({r.get("model_served") for r in rows if r.get("model_served")})
    print(f"rows {len(rows)}   runs {', '.join(map(str, runs))}")
    print(f"model(s) served: {', '.join(models) or 'n/a'}")
    total_cost = sum(float(r.get("cost") or 0.0) for r in rows)
    print(f"total cost: ${total_cost:.6f}")
    if any("SIMULATED" in (m or "") for m in models):
        print("\n*** SIMULATED DATA -- not a real measurement ***")

    data, marker = assemble(rows)
    report(data, marker, "ALL DOMAINS")

    if args.by_domain:
        domains = sorted({data["domain"][i] for i in data["items"]})
        for dom in domains:
            sub = [r for r in rows if r.get("domain") == dom]
            d2, m2 = assemble(sub)
            report(d2, m2, f"domain={dom}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
