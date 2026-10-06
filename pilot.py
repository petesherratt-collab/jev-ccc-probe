#!/usr/bin/env python3
"""
Step 2: the paired pilot.

Three conditions per item, differing only in the `analysis_notes` field of the
state object:

    baseline   no notes
    injected   plausible reasoning that concludes the WRONG way
    control    blind mirrored control -- same shape and length, honest
               conclusion

The control matters. Without it, any effect of merely having notes in the state
is indistinguishable from the effect of the notes being misleading. The CCC
contrast is injected-vs-control, with baseline retained to show the judge is
competent on the item in the first place.

Two dependent variables come out of one Noul:
    noul        continuous probability the answer is correct -- PRIMARY
    verdict     noul > 0.5 -- secondary, much lower power

Missingness is strict: an item contributes to the paired analysis only if all
three conditions returned a usable answer. Failures are logged as rows with
ok=false and counted in a completeness marker; they are never retried into
existence or quietly replaced.

Usage
-----
    python pilot.py --items items_demo.jsonl --simulate
    export OPENROUTER_API_KEY=...
    python pilot.py --items items_demo.jsonl --n 50 --obs _obs.jsonl
    python analyse.py --obs _obs.jsonl
"""

from __future__ import annotations

import argparse
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor

import jevlib as J

QID = "correct"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", required=True)
    ap.add_argument("--n", type=int, default=50,
                    help="items to sample (0 = all)")
    ap.add_argument("--repeats", type=int, default=1,
                    help="calls per item-condition; leave at 1 if determinism.py "
                         "says the model is deterministic")
    ap.add_argument("--model", default=J.DEFAULT_MODEL)
    ap.add_argument("--obs", default="_obs.jsonl")
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--domain", action="append", default=None,
                    help="restrict to a domain; repeatable")
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--sim-effect", type=float, default=0.35,
                    help="simulator only: planted injection effect on noul")
    ap.add_argument("--sim-control-effect", type=float, default=0.03)
    ap.add_argument("--sim-jitter", type=float, default=0.0)
    ap.add_argument("--sim-fail-rate", type=float, default=0.0,
                    help="simulator only: fraction of calls that fail, to "
                         "exercise the missingness rule")
    args = ap.parse_args()

    items = J.load_items(args.items)
    if args.domain:
        want = set(args.domain)
        items = [it for it in items if it.domain in want]
        if not items:
            print(f"no items in domain(s) {sorted(want)}", file=sys.stderr)
            return 2

    # Independent streams. `select` picks items; `order` shuffles the call
    # schedule. Deriving both by name means neither can be reconstructed from
    # the other -- the pre-1.5.1 failure mode.
    select, order = J.split_streams(args.seed, "select", "order")

    if args.n and args.n < len(items):
        items = select.sample(items, args.n)

    client = J.JevClient(model=args.model, simulate=args.simulate,
                         sim_effect=args.sim_effect,
                         sim_control_effect=args.sim_control_effect,
                         sim_jitter=args.sim_jitter,
                         sim_fail_rate=args.sim_fail_rate)

    run_id = uuid.uuid4().hex[:12]
    questions = {QID: J.noul_question(J.JUDGE_INSTRUCTIONS)}

    jobs = [(it, cond, rep)
            for it in items
            for cond in J.CONDITIONS
            for rep in range(args.repeats)]
    order.shuffle(jobs)

    print(f"run {run_id}: {len(items)} items x {len(J.CONDITIONS)} conditions "
          f"x {args.repeats} rep(s) = {len(jobs)} calls"
          f"{'  [SIMULATED]' if args.simulate else ''}", file=sys.stderr)

    def call(job):
        item, cond, rep = job
        state = J.build_state(item, cond)
        payload = state if args.simulate else J.strip_private(state)
        res = client.decide(payload, questions)
        row = J.make_row(run_id=run_id, item=item, condition=cond,
                         state=state, questions=questions, result=res,
                         qid=QID, repeat=rep)
        row["model_requested"] = args.model
        row["seed"] = args.seed
        return row

    n_ok = n_fail = 0
    with J.ObsWriter(args.obs) as obs:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i, row in enumerate(pool.map(call, jobs), 1):
                obs.write(row)
                if row["ok"]:
                    n_ok += 1
                else:
                    n_fail += 1
                if i % 25 == 0 or i == len(jobs):
                    print(f"  {i}/{len(jobs)} calls  "
                          f"ok={n_ok} fail={n_fail}  "
                          f"cost=${client.total_cost:.5f}", file=sys.stderr)

    print(f"\nrun_id {run_id}", file=sys.stderr)
    print(f"calls {len(jobs)}  ok {n_ok}  failed {n_fail}  "
          f"cost ${client.total_cost:.6f}", file=sys.stderr)
    print(f"raw rows appended to {args.obs}", file=sys.stderr)
    print(f"\nnext: python analyse.py --obs {args.obs} --run-id {run_id}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
