#!/usr/bin/env python3
"""
Step 1: is Jev deterministic?

This decides the whole sample-size question and costs almost nothing, so it runs
first and nothing else is sized until it has an answer.

Jev exposes no temperature and no seed. If identical input gives identical
output, then repeated calls on the same item carry ZERO extra information and
your n is the number of distinct items. If output varies, repeats buy you
precision on a per-item mean and n can be thought of in calls -- but you then
also need to decide how many repeats per item, and the item-level variance
becomes part of the power calculation.

Usage
-----
    # offline, no key needed -- checks the harness itself
    python determinism.py --items items_demo.jsonl --simulate
    python determinism.py --items items_demo.jsonl --simulate --sim-jitter 0.05

    # real
    export OPENROUTER_API_KEY=...
    python determinism.py --items items_demo.jsonl --n 20 --repeats 5

Reads nothing but the item file; writes raw rows to determinism_obs.jsonl.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor

import jevlib as J

QID = "correct"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", required=True, help="items JSONL")
    ap.add_argument("--n", type=int, default=20, help="distinct items to probe")
    ap.add_argument("--repeats", type=int, default=5, help="calls per item")
    ap.add_argument("--model", default=J.DEFAULT_MODEL)
    ap.add_argument("--obs", default="determinism_obs.jsonl")
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--simulate", action="store_true",
                    help="offline simulator, no API key or network")
    ap.add_argument("--sim-jitter", type=float, default=0.0,
                    help="simulator only: sd of per-call noise (0 = deterministic)")
    args = ap.parse_args()

    items = J.load_items(args.items)
    select, = J.split_streams(args.seed, "select")
    probe = select.sample(items, min(args.n, len(items)))

    client = J.JevClient(model=args.model, simulate=args.simulate,
                         sim_jitter=args.sim_jitter)
    run_id = uuid.uuid4().hex[:12]
    questions = {QID: J.noul_question(J.JUDGE_INSTRUCTIONS)}

    print(f"run {run_id}: {len(probe)} items x {args.repeats} repeats "
          f"= {len(probe) * args.repeats} calls"
          f"{'  [SIMULATED]' if args.simulate else ''}", file=sys.stderr)

    jobs = [(it, r) for it in probe for r in range(args.repeats)]

    def call(job):
        item, rep = job
        state = J.build_state(item, "baseline")
        res = client.decide(J.strip_private(state) if not args.simulate else state,
                            questions)
        row = J.make_row(run_id=run_id, item=item, condition="baseline",
                         state=state, questions=questions, result=res,
                         qid=QID, repeat=rep)
        row["model_requested"] = args.model
        return row

    with J.ObsWriter(args.obs) as obs:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for row in pool.map(call, jobs):
                obs.write(row)

    report(args.obs, run_id, args.repeats, client)
    return 0


def report(obs_path: str, run_id: str, repeats: int, client) -> None:
    rows = [r for r in J.read_obs(obs_path) if r.get("run_id") == run_id]
    ok = [r for r in rows if r.get("ok") and r.get("noul") is not None]

    by_item: dict[str, list[float]] = {}
    for r in ok:
        by_item.setdefault(r["item_id"], []).append(float(r["noul"]))

    complete = {k: v for k, v in by_item.items() if len(v) == repeats}

    print()
    print("=" * 68)
    print("DETERMINISM CHECK")
    print("=" * 68)
    print(f"calls attempted      : {len(rows)}")
    print(f"calls succeeded      : {len(ok)}")
    print(f"items with all {repeats} reps: {len(complete)} / {len(by_item)}")
    print(f"cost                 : ${client.total_cost:.6f}")

    if not complete:
        print("\nNot enough complete items to judge. Check the errors in "
              f"{obs_path}.")
        return

    ranges = []
    identical_verdict = 0
    identical_value = 0
    for vals in complete.values():
        rng = max(vals) - min(vals)
        ranges.append(rng)
        if len({v > 0.5 for v in vals}) == 1:
            identical_verdict += 1
        if rng == 0.0:
            identical_value += 1

    n = len(complete)
    print()
    print(f"identical noul value across reps : {identical_value}/{n} "
          f"({identical_value / n:.0%})")
    print(f"identical verdict across reps    : {identical_verdict}/{n} "
          f"({identical_verdict / n:.0%})")
    print(f"noul range per item  median      : {statistics.median(ranges):.4f}")
    print(f"                     max         : {max(ranges):.4f}")
    if n > 1:
        within = [statistics.pstdev(v) for v in complete.values() if len(v) > 1]
        print(f"within-item sd       mean        : "
              f"{statistics.mean(within):.4f}")

    print()
    print("-" * 68)
    if identical_value == n:
        print("VERDICT: deterministic on identical input.")
        print("  -> Repeats buy you nothing. n = number of DISTINCT ITEMS.")
        print("  -> Do not report repeated calls as independent observations.")
    elif identical_verdict == n:
        print("VERDICT: verdict-stable but numerically noisy.")
        print("  -> Flip-rate analyses: n = distinct items.")
        print("  -> noul-shift analyses: repeats reduce measurement error; use")
        print("     3-5 reps per item-condition and analyse per-item means.")
    else:
        print("VERDICT: stochastic -- verdicts vary on identical input.")
        print("  -> Repeats are mandatory, and within-item variance must enter")
        print("     the power calculation. Re-run the pilot sizing with the")
        print("     within-item sd printed above before committing to an n.")
    print("-" * 68)
    print(f"\nRaw rows: {obs_path}")


if __name__ == "__main__":
    sys.exit(main())
