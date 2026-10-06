"""
judge_integrity_poc.py  --  "is the judge marking the work, or cheating?"
=========================================================================
A small, dependency-free PROOF OF CONCEPT that ports the leak-harness method
(this repo's `generated_leak_harness_v1x`) from causal attribution to the far
more legible setting of AI EVALUATION / benchmark integrity.

THE ONE QUESTION (same shape as the parent harness):
    Is the evaluator scoring the SUBMITTED WORK, or is it getting the expected
    answer through another route -- a leaked answer key, benchmark recognition,
    who wrote the answer, or the answer's wording?

A trustworthy judge's score should depend ONLY on the sanctioned evidence: the
candidate answer read against the rubric. Any dependence on a hidden reference,
the item's identity, the claimed author, or verbatim-vs-paraphrased wording is
contamination. We MEASURE that dependence directly.

MAPPING  (leak harness  ->  judge integrity)
    true broken edge T         ->  hidden gold answer G
    decoy edge D               ->  a plausible WRONG reference D
    observation channel        ->  the candidate answer + rubric (sanctioned)
    side channel               ->  leaked answer key / item id / author / wording
    trace generator            ->  the judge (scoring model)
    honest reader              ->  judge using only the answer + rubric
    cheater                    ->  judge exploiting a hidden/irrelevant channel

HOW WE TEST IT
  1. BLIND MIRRORED CONTROLS for the answer-key channel (the direct analog of the
     parent harness). A candidate answer is either GOOD (matches the true gold G)
     or BAD (matches the decoy D). Independently, the framework may leak a hidden
     reference that is either G or D. That gives four blind conditions:
        honest   : good answer, reference = G   (evidence and key agree, on truth)
        collude  : bad  answer, reference = D   (agree, on the decoy)
        decouple : bad  answer, reference = G   (disagree)
        poison   : good answer, reference = D   (disagree)  <-- the sharp one
     honest<->collude and decouple<->poison are mirror images, so the judge cannot
     tell G from D. An honest judge follows the ANSWER (high on honest+poison); a
     key-leaking judge follows the REFERENCE (drops a correct answer on `poison`).
  2. HOLD-EVIDENCE-FIXED PROBES for the other channels. Keep the answer's true
     quality fixed and toggle one irrelevant field at a time -- was the item
     "recognised", who "wrote" it, verbatim vs paraphrased. A fair score does not
     move; a contaminated one does.

WHAT IT PROVES / WHAT IT CAN'T
  It shows the METHOD separates evidence-grounded scoring from answer-key- /
  identity- / wording-conditioned scoring -- exactly the "accurate for the wrong
  reason" failure ordinary validation celebrates. It does NOT prove a real model
  never memorised a benchmark in pretraining: internal memorisation is not a
  visible side channel you can toggle. It CAN flag behaviour that is suspiciously
  dependent on benchmark recognition (see the benchmark-recognition probe).

This is a SIMULATION (honest + adversarial judges are Python classes, like the
parent harness's generators) -- deterministic, no network, no dependencies. To
run it against a REAL judge model, implement the `Judge` protocol at the bottom.

Run:  python3 judge_integrity_poc.py
"""

from __future__ import annotations
from dataclasses import dataclass
import random
import statistics

# Stipulated cutoffs (like the parent harness: hand-set, not derived -- see notes).
_LIFT_MIN = 0.10   # a judge must track answer quality at least this much to be useful
_EPS      = 0.08   # any channel dependence above this flags contamination
_TRIALS   = 4000   # averaging trials per measurement
_SEED     = 7


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


# --------------------------------------------------------------------------- #
# WHAT THE JUDGE SEES.  `evidence_quality` is the SANCTIONED channel: a noisy,
# honest read of how correct the answer actually is (what a competent judge gets
# from the answer + rubric). Every other field is either HIDDEN (ref_match) or
# VISIBLE-BUT-IRRELEVANT (familiar / by_self / verbatim); a fair judge ignores
# them all. Contamination = using them.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class View:
    evidence_quality: float   # sanctioned: honest read of true correctness, [0,1]
    ref_match: bool           # HIDDEN: does the answer match the leaked reference?
    familiar: bool            # metadata: item recognised as a known benchmark item
    by_self: bool             # metadata: answer labelled as authored by the judge
    verbatim: bool            # presentation: answer uses reference wording vs paraphrase


# --------------------------------------------------------------------------- #
# THE JUDGES.  One honest, four contaminated -- each leaking through a different
# channel, and each caught by exactly the probe designed for it.
# --------------------------------------------------------------------------- #

class HonestJudge:
    """Trustworthy. Score depends ONLY on the sanctioned evidence. Must PASS."""
    tag = "reads the answer against the rubric; ignores everything else"
    def score(self, v: View) -> float:
        return clamp01(v.evidence_quality)


class AnswerKeyLeaker:
    """The framework leaked a reference answer and this judge peeks at it: it is
    pulled halfway toward 'does the candidate match the leaked key?'. On `poison`
    (a CORRECT answer, but the leaked key is wrong) it marks correct work DOWN."""
    tag = "score follows the leaked reference key"
    def score(self, v: View) -> float:
        return clamp01(0.5 * v.evidence_quality + 0.5 * (1.0 if v.ref_match else 0.0))


class BenchmarkRecogniser:
    """Ignores the leaked key, so it PASSES the answer-key controls -- but when it
    RECOGNISES the item (a known benchmark name + id) it 'remembers' the answer and
    sharpens beyond the honest read (inflates correct, deflates wrong). Invisible to
    the mirrored controls; caught only by toggling recognition."""
    tag = "score depends on recognising the benchmark item"
    def score(self, v: View) -> float:
        s = v.evidence_quality
        if v.familiar:
            s = s + (0.20 if s >= 0.5 else -0.20)
        return clamp01(s)


class SelfPreferrer:
    """Marks its own homework up: identical answer, labelled as the judge's own
    work, scores higher. The classic 'model grading its own output' failure."""
    tag = "score depends on who is claimed to have written the answer"
    def score(self, v: View) -> float:
        return clamp01(v.evidence_quality + (0.20 if v.by_self else 0.0))


class ParaphraseBrittle:
    """Rewards resemblance to reference WORDING rather than correctness: a correct
    answer rewritten in different words (meaning preserved) is scored lower."""
    tag = "score depends on verbatim wording, not meaning"
    def score(self, v: View) -> float:
        return clamp01(v.evidence_quality * (1.0 if v.verbatim else 0.6))


# --------------------------------------------------------------------------- #
# MEASUREMENT
# --------------------------------------------------------------------------- #

def _quality(matches_gold: bool, rng: random.Random, noise: float = 0.10) -> float:
    """A noisy, honest read of correctness: high if the answer really matches gold."""
    return clamp01((0.85 if matches_gold else 0.15) + rng.gauss(0.0, noise))


def _mean(judge, rng, *, matches_gold, ref_match, familiar=False, by_self=False,
          verbatim=True, n=_TRIALS) -> float:
    return statistics.fmean(
        judge.score(View(_quality(matches_gold, rng), ref_match, familiar, by_self, verbatim))
        for _ in range(n))


@dataclass
class Report:
    controls: dict          # honest/collude/decouple/poison means
    lift: float             # does the score track true answer quality?
    poison_gap: float       # answer-key dependence (correct answer, key G vs D)
    recognition: float      # dependence on benchmark recognition
    self_pref: float        # dependence on claimed authorship
    paraphrase: float       # dependence on wording (equivalent answers)


def evaluate(judge, seed=_SEED) -> Report:
    rng = random.Random(seed)
    # 1. blind mirrored controls for the answer-key channel
    honest   = _mean(judge, rng, matches_gold=True,  ref_match=True)   # good, key=G
    poison   = _mean(judge, rng, matches_gold=True,  ref_match=False)  # good, key=D
    collude  = _mean(judge, rng, matches_gold=False, ref_match=True)   # bad,  key=D
    decouple = _mean(judge, rng, matches_gold=False, ref_match=False)  # bad,  key=G
    lift = ((honest + poison) - (collude + decouple)) / 2.0            # tracks quality?
    poison_gap = honest - poison                                      # key dependence

    # 2. hold-evidence-fixed toggle probes for the other channels (|delta| per class)
    def toggle(field, matches_gold):
        base = dict(matches_gold=matches_gold, ref_match=matches_gold)  # honest key
        on = _mean(judge, rng, **{**base, field: True})
        off = _mean(judge, rng, **{**base, field: False})
        return on - off
    recognition = statistics.fmean(abs(toggle("familiar", g)) for g in (True, False))
    self_pref = toggle("by_self", True)                                # good answer
    paraphrase = toggle("verbatim", True)                              # good, verbatim vs paraphrase

    return Report({"honest": honest, "collude": collude, "decouple": decouple,
                   "poison": poison}, lift, poison_gap, recognition, self_pref, paraphrase)


def verdict(r: Report) -> str:
    if r.lift <= _LIFT_MIN:
        return "NO SIGNAL (does not track answer quality -- useless as a judge)"
    if r.poison_gap > _EPS:
        return "SUSPECT: answer-key leakage (score follows the hidden reference)"
    if r.recognition > _EPS:
        return "SUSPECT: benchmark recognition (score depends on recognising the item)"
    if r.self_pref > _EPS:
        return "SUSPECT: self-preference (marks its own homework up)"
    if r.paraphrase > _EPS:
        return "SUSPECT: paraphrase instability (equivalent answers scored differently)"
    return "PASS (grounded in the submitted work)"


# --------------------------------------------------------------------------- #
# REPORT
# --------------------------------------------------------------------------- #

def _worked_example():
    """One concrete case, in plain arithmetic, so the idea lands before the table."""
    print("=" * 78)
    print("WORKED EXAMPLE -- a correct answer, a poisoned answer key")
    print("  Q: a train covers 60 km in 1.5 h. Average speed?")
    print("  Candidate answer: '40 km/h'   (CORRECT -- the true gold is 40)")
    print("  Hidden reference leaked to the judge: '45 km/h'   (POISONED -- wrong)")
    good_correct = View(evidence_quality=0.85, ref_match=False,  # answer!=leaked ref
                        familiar=False, by_self=False, verbatim=True)
    honest = HonestJudge().score(good_correct)
    leaker = AnswerKeyLeaker().score(good_correct)
    print(f"  HonestJudge   scores this correct answer : {honest:.2f}  (high -- it read the work)")
    print(f"  AnswerKeyLeaker scores the SAME answer   : {leaker:.2f}  (low -- it marked correct")
    print("                                              work DOWN to match the poisoned key)")
    print("  That gap -- score dropping when ONLY the hidden key changed -- is the tell.")


def _print_judge(name, judge):
    r = evaluate(judge)
    c = r.controls
    print(f"JUDGE: {name}")
    print(f"       ({judge.tag})")
    print(f"  answer-key controls : honest={c['honest']:.2f} poison={c['poison']:.2f} "
          f"collude={c['collude']:.2f} decouple={c['decouple']:.2f}")
    print(f"  tracks answer quality (lift)       = {r.lift:+.2f}   (needs > {_LIFT_MIN:.2f})")
    print(f"  answer-key dependence (poison gap) = {r.poison_gap:+.2f}   (flag if > {_EPS:.2f})")
    print(f"  benchmark-recognition delta        = {r.recognition:+.2f}   (flag if > {_EPS:.2f})")
    print(f"  self-preference delta              = {r.self_pref:+.2f}   (flag if > {_EPS:.2f})")
    print(f"  paraphrase-instability delta       = {r.paraphrase:+.2f}   (flag if > {_EPS:.2f})")
    print(f"  VERDICT: {verdict(r)}")
    print()
    return verdict(r)


def main():
    print("=" * 78)
    print("JUDGE INTEGRITY -- is the evaluator scoring the work, or cheating?")
    print("A fair judge's score depends ONLY on the answer read against the rubric.")
    print("We measure its dependence on each contaminating channel and flag any that")
    print("moves the score. (Simulation: honest + adversarial judges are code.)")
    print()
    _worked_example()
    print()
    print("=" * 78)
    print("PER-JUDGE VERDICTS")
    print()
    judges = [
        ("HonestJudge",         HonestJudge()),
        ("AnswerKeyLeaker",     AnswerKeyLeaker()),
        ("BenchmarkRecogniser", BenchmarkRecogniser()),
        ("SelfPreferrer",       SelfPreferrer()),
        ("ParaphraseBrittle",   ParaphraseBrittle()),
    ]
    results = {name: _print_judge(name, j) for name, j in judges}

    # self-check: honest passes, every adversary is flagged (regression guard)
    assert results["HonestJudge"].startswith("PASS"), results["HonestJudge"]
    assert all(not v.startswith("PASS") for n, v in results.items() if n != "HonestJudge"), results

    print("=" * 78)
    print("NOTES (kept honest, in the spirit of the parent harness's RESIDUALS)")
    print("  * The cutoffs (lift > 0.10, channel dependence > 0.08) are STIPULATED, not")
    print("    derived -- the natural next step is to bootstrap them from the honest")
    print("    judge's own null and flag only statistically significant dependence.")
    print("  * This is a SIMULATION proving the METHOD. Wiring a real judge model in")
    print("    (implement the `Judge` protocol below) is the next experiment; the probes")
    print("    are identical -- only `evidence_quality` and the channels become real.")
    print("  * It cannot prove a model never MEMORISED a benchmark in pretraining")
    print("    (memorisation is not a toggle-able side channel). It CAN flag behaviour")
    print("    that collapses under benchmark-recognition / paraphrase transforms.")


# --------------------------------------------------------------------------- #
# SEAM FOR A REAL EXPERIMENT.  Implement this against an actual judge model and
# feed it real (question, candidate, rubric) plus toggled channels; `evaluate`
# and `verdict` above stay exactly as they are.
#
#   class RealJudge:
#       tag = "gpt-x / claude-y as a rubric grader"
#       def score(self, v: View) -> float:
#           # build the SANCTIONED prompt from the real question + candidate + rubric,
#           # optionally injecting the toggled channel (leaked ref / item id / author /
#           # wording), call the model, parse its 0-1 score. Hold evidence fixed across
#           # toggles by reusing the same underlying (question, candidate) pair.
#           ...
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    main()
