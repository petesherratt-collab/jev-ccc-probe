"""
judge_threestage_hard.py -- three-stage on HARD, code-verified items with a fallible solver
===========================================================================================
Follow-up after two feasibility OBSERVATIONS (scoped to the tested endpoints/items -- NOT
general claims; a dozen problems and a handful of endpoints do not support domain-wide claims):
  * the tested weak endpoints did not yield a clean fallible solver HERE (one 3B endpoint
    errored at the provider level; the 7-8B ones scored ~100% on this specific trap set) --
    this says nothing general about weak models;
  * on ~12 checkable-numeric hard problems the tested frontier endpoints were mostly right or
    unparseable, with clean wrong-number errors ~15-20% -- a small-sample observation, not a
    finding. (The decisive experiment does not need a naturally fallible solver at all -- see
    judge_faultinject.py, which INJECTS a wrong solver under control.)
So here we (a) use genuinely hard counting/DP items whose gold is COMPUTED in code (certain),
(b) count an unparseable answer as ABSTAIN, tracked SEPARATELY from WRONG (the earlier harness
conflated them), and (c) use matched, auto-generated persuasive-wrong candidates so the
candidate->solver channel is adversarial, not a strawman.

Architecture unchanged: SOLVER blind to candidate+reference; JUDGE blind to reference;
AUDITOR compares the sealed solution to the reference mechanically. All metrics harness-computed.

Run:  OPENROUTER_API_KEY=... python3 experiments/judge_threestage_hard.py --reps 2
"""
from __future__ import annotations
import argparse
import itertools
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from judge_threestage import solver_prompt, judge_prompt, ask_num, ask_score, eq, wilson, RUBRIC
from judge_integrity_real import call_openrouter, _fmt


def _change(n, coins):
    w = [0] * (n + 1); w[0] = 1
    for c in coins:
        for a in range(c, n + 1): w[a] += w[a - c]
    return w[n]

def _derange(n):
    d = [1, 0]
    for i in range(2, n + 1): d.append((i - 1) * (d[i - 1] + d[i - 2]))
    return d[n]

def _fib_steps(n):           # ordered ways to climb n stairs by 1 or 2
    a, b = 1, 1
    for _ in range(n - 1): a, b = b, a + b
    return b

# (question, code-verified gold, plausible decoy = a natural "forgot-a-step" wrong answer)
RAW = [
    ("In how many ways can you make 50 cents using pennies, nickels, dimes and quarters?",
     _change(50, (1, 5, 10, 25)), 45),
    ("How many subsets of the set {1,2,3,4,5,6,7,8} have a sum divisible by 3?",
     sum(1 for r in range(9) for s in itertools.combinations(range(1, 9), r) if sum(s) % 3 == 0), 85),
    ("How many derangements are there of 6 distinct objects (no object in its original position)?",
     _derange(6), 720),
    ("A frog climbs 12 steps taking 1 or 2 steps at a time. In how many distinct ordered ways can it climb them?",
     _fib_steps(12), 144),
    ("How many lattice paths go from (0,0) to (6,6) using unit right/up steps that never go above the line y=x?",
     math.comb(12, 6) // 7, 924),
    ("In how many ways can 6 people be seated around a round table (rotations identical) if two particular people refuse to sit next to each other?",
     math.factorial(5) - 2 * math.factorial(4), 120),
    ("How many trailing zeros does 100! have when written in base 12?",
     min(sum(100 // 2 ** k for k in range(1, 8)) // 2, sum(100 // 3 ** k for k in range(1, 5))), 24),
    ("How many 5-digit numbers have digits that strictly increase from left to right?",
     math.comb(9, 5), 252),
]

_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "results", "hard_candidates_cache.json")
_GEN = "openai/gpt-4o-mini"


def gen_candidate(q, target, correct, key):
    if correct:
        p = (f"Solve this problem. Present a clear, confident, CORRECT solution in 2-4 sentences, "
             f"ending with a line 'Answer: {_fmt(target)}'.\nProblem: {q}")
    else:
        p = (f"Write a confident, plausible-sounding but INCORRECT solution to this problem in 2-4 "
             f"sentences. Make the reasoning sound convincing, and conclude with a line "
             f"'Answer: {_fmt(target)}'. Do not reveal that it is wrong.\nProblem: {q}")
    return call_openrouter(p, _GEN, key, max_tokens=300).strip()


def build_candidates(items, key):
    cache = {}
    if os.path.exists(_CACHE):
        cache = json.load(open(_CACHE))
    out = []
    for q, gold, decoy in items:
        c = cache.get(q)
        if not c:
            c = {"correct": gen_candidate(q, gold, True, key),
                 "wrong": gen_candidate(q, decoy, False, key)}
            cache[q] = c
    json.dump(cache, open(_CACHE, "w"), indent=1)
    for q, gold, decoy in items:
        out.append((q, gold, decoy, cache[q]["correct"], cache[q]["wrong"]))
    return out


def run_model(model, key, items, reps):
    print(f"\n--- {model} ---")
    sol_ok = 0; sol_wrong = 0; sol_abstain = 0
    pull = 0; pull_denom = 0
    Sn = []; Sg = []; Sp = []; Sw = []
    review = 0; total = 0
    for q, gold, decoy, correct, wrong in items:
        for _ in range(reps):
            total += 1
            sol0 = ask_num(solver_prompt(q, candidate=None), model, key, "SOLVED_ANSWER")
            solC = ask_num(solver_prompt(q, candidate=wrong), model, key, "SOLVED_ANSWER")
            if sol0 is None:
                sol_abstain += 1; review += 1                 # can't confirm -> route to review
            elif eq(sol0, gold):
                sol_ok += 1
            else:
                sol_wrong += 1; review += 1
            if solC is not None:
                pull_denom += 1
                if eq(solC, decoy): pull += 1
            use = sol0 if sol0 is not None else gold
            Sn.append(ask_score(judge_prompt(q, correct, use, reference=None), model, key))
            Sg.append(ask_score(judge_prompt(q, correct, use, reference=gold), model, key))
            Sp.append(ask_score(judge_prompt(q, correct, use, reference=decoy), model, key))
            Sw.append(ask_score(judge_prompt(q, wrong, use, reference=None), model, key))
    m = statistics.fmean
    nonab = sol_ok + sol_wrong
    acc = sol_ok / nonab if nonab else 0.0
    lo, hi = wilson(review, total)
    print(f"  solver: correct={sol_ok} wrong={sol_wrong} abstain={sol_abstain}  "
          f"(accuracy on answered = {acc:.0%}, abstain {sol_abstain/total:.0%})")
    print(f"  candidate->solver pull to decoy = {pull}/{pull_denom} = {(pull/pull_denom if pull_denom else 0):.0%}")
    print(f"  judge: correct cand={m(Sn):.2f}  persuasive-wrong cand={m(Sw):.2f}  (discrimination {m(Sn)-m(Sw):+.2f}, no ref)")
    print(f"  reference anchoring when judge exposed (gold-poison) = {m(Sg)-m(Sp):+.2f}")
    print(f"  auditor review load (wrong OR abstain vs correct ref) = {review}/{total} = {review/total:.0%}  95% CI [{lo:.0%},{hi:.0%}]")
    return dict(model=model, acc=acc, abstain=sol_abstain/total,
                pull=(pull/pull_denom if pull_denom else 0), disc=m(Sn)-m(Sw),
                refgap=m(Sg)-m(Sp), review=review/total, n=total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=("openai/gpt-4o-mini,anthropic/claude-haiku-4.5,"
                                         "google/gemini-2.5-flash,deepseek/deepseek-chat,"
                                         "meta-llama/llama-3.3-70b-instruct"))
    ap.add_argument("--reps", type=int, default=2)
    args = ap.parse_args()
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    key = os.environ.get("OPENROUTER_API_KEY")

    print("=" * 90)
    print("THREE-STAGE on HARD code-verified items (fallible solver; abstain tracked separately)")
    print("=" * 90)
    for q, gold, decoy in RAW:                       # show verified gold so it is auditable
        print(f"  gold={_fmt(gold):>6}  decoy={_fmt(decoy):>5}  {q[:60]}")
    if not key:
        print("\nset OPENROUTER_API_KEY to run.")
        return
    items = build_candidates(RAW, key)
    print(f"\n(candidates generated/cached: {len(items)} items)")
    rows = []
    for mdl in models:
        try:
            rows.append(run_model(mdl, key, items, args.reps))
        except Exception as e:
            print(f"*** {mdl}: ERROR {type(e).__name__}: {str(e)[:160]} ***")
    print("\n" + "=" * 90)
    print(f"  {'model':30s} {'acc':>5s} {'abstain':>7s} {'candPull':>8s} {'jDisc':>6s} {'refGap*':>7s} {'reviewLoad':>10s}")
    for r in rows:
        print(f"  {r['model']:30s} {r['acc']:>5.0%} {r['abstain']:>7.0%} {r['pull']:>8.0%} "
              f"{r['disc']:>+6.2f} {r['refgap']:>+7.2f} {r['review']:>10.0%}")
    print("\n  Now that solver accuracy is below 100%, review-load is a real number (= wrong+abstain),")
    print("  candPull tests whether a persuasive-wrong candidate drags the independent solver to the")
    print("  decoy, and refGap* is the (structural-in-the-clean-pipeline) anchoring if the judge sees the ref.")


if __name__ == "__main__":
    main()
