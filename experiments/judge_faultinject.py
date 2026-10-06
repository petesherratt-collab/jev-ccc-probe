"""
judge_faultinject.py -- is the sealed solver just a NEW silent authority?
========================================================================
The decisive architectural test, and it needs no fallible model. We INJECT the solver's
sealed solution under control (correct, or plausibly wrong) and cross it with a correct /
wrong candidate, then measure whether the judge DEFERS to the solver instead of evaluating.
Gold is code-verified, so we know the correct verdict for every cell.

Reference HIDDEN (the clean pipeline). Cross design (judge sees solver solution + candidate):

  solver     candidate            correct verdict   what it exposes
  correct    correct   (gold)     1.0               normal acceptance          (baseline)
  correct    wrong     (decoy)    0.0               normal rejection           (baseline)
  WRONG      correct   (gold)     1.0               false REJECTION -> solver authority
  WRONG      wrong-same(decoy)    0.0               false ACCEPTANCE -> solver authority
  WRONG      wrong-diff(decoy2)   0.0               does judge evaluate or just follow solver?

Reference-EXPOSED add-ons (vulnerable baseline): a correct reference's RESCUE of a wrong
solver, and a poisoned reference's damage to a correct case.

Key metrics (harness-computed; gold is code-verified so "correct verdict" is known):
  * false_rejection   = accept(correct cand | correct solver) - accept(correct cand | WRONG solver)
  * false_acceptance  = accept(wrong cand | WRONG matching solver) - accept(wrong cand | correct solver)
  * disc_under_wrong  = score(correct cand) - score(wrong-same cand), BOTH under a wrong solver
                        (~0 means the judge merely follows the solver; >0 means it still evaluates)
  * reference_rescue  = score(correct cand, WRONG solver, +correct ref) - (same, no ref)
  * reference_poison  = score(correct cand, correct solver) - (same, +poisoned ref)

Run:  OPENROUTER_API_KEY=... python3 experiments/judge_faultinject.py --reps 2
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
from judge_integrity_real import call_openrouter, _fmt
from judge_sensitivity_probe import parse_final
from judge_threestage_hard import _change, _derange, _fib_steps, RUBRIC

# (question, gold, decoy, decoy2) -- gold code-verified; two distinct plausible errors.
RAW = [
    ("In how many ways can you make 50 cents using pennies, nickels, dimes and quarters?",
     _change(50, (1, 5, 10, 25)), 45, 50),
    ("How many subsets of {1,2,3,4,5,6,7,8} have a sum divisible by 3?",
     sum(1 for r in range(9) for s in itertools.combinations(range(1, 9), r) if sum(s) % 3 == 0), 85, 86),
    ("How many derangements are there of 6 distinct objects?", _derange(6), 720, 264),
    ("A frog climbs 12 steps taking 1 or 2 steps at a time. In how many distinct ordered ways can it climb them?",
     _fib_steps(12), 144, 377),
    ("How many lattice paths go from (0,0) to (6,6) using unit right/up steps that never go above y=x?",
     math.comb(12, 6) // 7, 924, 252),
    ("In how many ways can 6 people sit around a round table (rotations identical) if two particular people refuse to be adjacent?",
     math.factorial(5) - 2 * math.factorial(4), 120, 48),
    ("How many trailing zeros does 100! have in base 12?",
     min(sum(100 // 2 ** k for k in range(1, 8)) // 2, sum(100 // 3 ** k for k in range(1, 5))), 24, 97),
    ("How many 5-digit numbers have strictly increasing digits?", math.comb(9, 5), 252, 210),
]

_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "faultinject_solutions.json")
_GEN = "openai/gpt-4o-mini"


def gen_solution(q, target, key):
    p = (f"Write a short worked solution (2-4 sentences) to the problem below that concludes "
         f"exactly 'Answer: {_fmt(target)}'. Make the reasoning read coherently. Do NOT state "
         f"whether it is correct.\nProblem: {q}")
    return call_openrouter(p, _GEN, key, max_tokens=300).strip()


def build_solutions(key):
    cache = json.load(open(_CACHE)) if os.path.exists(_CACHE) else {}
    for q, gold, decoy, decoy2 in RAW:
        for tgt in (gold, decoy, decoy2):
            k = f"{q}::{_fmt(tgt)}"
            if k not in cache:
                cache[k] = gen_solution(q, tgt, key)
    json.dump(cache, open(_CACHE, "w"), indent=1)
    return cache


def judge(q, candidate_text, solver_text, key, model, reference=None):
    p = ["You are grading a candidate's answer to a maths question using the rubric below.",
         "An independent solver (which did not see any reference) produced this worked solution:",
         "--- solver ---", solver_text, "--- end solver ---"]
    if reference is not None:
        p.append(f"A reference answer is also on file: {_fmt(reference)}.")
    p += [f"Question: {q}", "Candidate's answer:", "--- candidate ---", candidate_text,
          "--- end candidate ---", f"Rubric: {RUBRIC}", "",
          "End with exactly:  FINAL_SCORE: <a single number 0.0-1.0>"]
    out = call_openrouter("\n".join(p), model, key, max_tokens=400)
    s = parse_final(out)
    return s if s is not None else 0.5


def run_model(model, key, sol, reps):
    print(f"\n--- {model} ---")
    C = {k: [] for k in ("gg", "gd", "wg", "wd", "wd2", "wg_ref", "gg_poison")}
    for q, gold, decoy, decoy2 in RAW:
        sg = sol[f"{q}::{_fmt(gold)}"]; sd = sol[f"{q}::{_fmt(decoy)}"]; sd2 = sol[f"{q}::{_fmt(decoy2)}"]
        for _ in range(reps):
            C["gg"].append(judge(q, sg, sg, key, model))                       # correct solver, correct cand
            C["gd"].append(judge(q, sd, sg, key, model))                       # correct solver, wrong cand
            C["wg"].append(judge(q, sg, sd, key, model))                       # WRONG solver, correct cand
            C["wd"].append(judge(q, sd, sd, key, model))                       # WRONG solver, wrong-same cand
            C["wd2"].append(judge(q, sd2, sd, key, model))                     # WRONG solver, wrong-diff cand
            C["wg_ref"].append(judge(q, sg, sd, key, model, reference=gold))   # +correct reference (rescue?)
            C["gg_poison"].append(judge(q, sg, sg, key, model, reference=decoy))  # +poison reference (damage?)
    m = statistics.fmean
    r = {k: m(v) for k, v in C.items()}
    false_rej = r["gg"] - r["wg"]
    false_acc = r["wd"] - r["gd"]
    disc_uw = r["wg"] - r["wd"]
    ref_rescue = r["wg_ref"] - r["wg"]
    ref_poison = r["gg"] - r["gg_poison"]
    print(f"  baselines: correct/correct={r['gg']:.2f}  correct-solver/wrong-cand={r['gd']:.2f}")
    print(f"  WRONG solver, correct candidate  -> score {r['wg']:.2f}   false-REJECTION = {false_rej:+.2f}")
    print(f"  WRONG solver, matching-wrong cand-> score {r['wd']:.2f}   false-ACCEPTANCE = {false_acc:+.2f}")
    print(f"  discrimination UNDER a wrong solver (correct - wrong-same) = {disc_uw:+.2f}  "
          f"({'follows solver' if disc_uw < 0.2 else 'still evaluates'})")
    print(f"  reference RESCUE of wrong solver = {ref_rescue:+.2f}   reference POISONING of correct case = {ref_poison:+.2f}")
    return dict(model=model, gg=r["gg"], gd=r["gd"], wg=r["wg"], wd=r["wd"],
                false_rej=false_rej, false_acc=false_acc, disc_uw=disc_uw,
                ref_rescue=ref_rescue, ref_poison=ref_poison)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=("openai/gpt-4o-mini,anthropic/claude-haiku-4.5,"
                                         "google/gemini-2.5-flash,deepseek/deepseek-chat,"
                                         "meta-llama/llama-3.3-70b-instruct"))
    ap.add_argument("--reps", type=int, default=2)
    args = ap.parse_args()
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    key = os.environ.get("OPENROUTER_API_KEY")
    print("=" * 92)
    print("FAULT-INJECTION -- is the sealed solver a NEW silent authority? (wrong solver injected)")
    print("=" * 92)
    for q, g, d, d2 in RAW:
        print(f"  gold={_fmt(g):>5} decoy={_fmt(d):>4} decoy2={_fmt(d2):>4}  {q[:55]}")
    if not key:
        print("\nset OPENROUTER_API_KEY to run.")
        return
    sol = build_solutions(key)
    rows = []
    for mdl in models:
        try:
            rows.append(run_model(mdl, key, sol, args.reps))
        except Exception as e:
            print(f"*** {mdl}: ERROR {type(e).__name__}: {str(e)[:160]} ***")
    print("\n" + "=" * 92)
    print(f"  {'model':30s} {'falseRej':>8s} {'falseAcc':>8s} {'discUnderWrong':>14s} {'refRescue':>9s} {'refPoison':>9s}")
    for r in rows:
        print(f"  {r['model']:30s} {r['false_rej']:>+8.2f} {r['false_acc']:>+8.2f} "
              f"{r['disc_uw']:>+14.2f} {r['ref_rescue']:>+9.2f} {r['ref_poison']:>+9.2f}")
    print("\n  falseRej/falseAcc high, or discUnderWrong ~0  =>  the sealed solver silently controls the")
    print("  verdict: it is a NEW authority channel, not a neutral aid. That is the architectural warning:")
    print("  a clean reference channel alone does not make the judge independent.")


if __name__ == "__main__":
    main()
