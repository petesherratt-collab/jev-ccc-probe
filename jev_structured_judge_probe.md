# Typed Provenance Is Inert: Contextual Conclusion Capture in a Structured-Decision Judge

**Author:** Pete Sherratt · Independent researcher · contextual-conclusion-capture@tuta.com

**Preprint draft — 2026-10.** Companion note to *Contextual Conclusion Capture: Conflicting
Conclusions Degrade LLM-Judge Discrimination Across Reasoning Domains* (CCC). This note is
**exploratory, not preregistered**: it reuses the CCC item bank and condition structure against a
single new endpoint of a different architectural class. All numeric results are recomputed from
streamed raw rows; see the Reproducibility appendix for artifacts, commit references and digests.
Two results from earlier drafts were withdrawn — one after a harness defect was found, one after a
preregistered follow-up contradicted it. Both, their effects and their corrections are reported in §9
rather than silently removed. **The second withdrawal changed this note's title**, which until
2026-10-06 asserted that typed provenance is load-bearing.

---

## Abstract

CCC established that an LLM judge's ability to distinguish correct from incorrect candidates
deteriorates when a conflicting conclusion is present in its context, that a **bare,
neutrally-labelled wrong answer is sufficient** to induce capture, and that an authoritative source
label is **not necessary**. All of that evidence came from judges that read a prompt and emit text.
We ask whether the failure survives a change of *judge architecture*: we probe `typesafe/jev-1.13`,
a "System One" structured-decision model that accepts a typed `state` object and returns a
calibrated probability rather than prose, and which its vendor describes as unable to hallucinate by
construction. Using the frozen CCC bank (24 SQL, 16 code, 16 arithmetic items), the four canonical
conditions and both candidate types, we find capture in **every domain** and under **both** state
encodings we tested. Three results are new.

First, the CCC core result **replicates in this architecture and is, if anything, stronger**: merely
placing a bare wrong answer in the `state` object costs the SQL judge 180% of its baseline
discrimination, driving it past zero into **preferring the wrong answer**, at a 100% verdict-flip
rate. Second, we encoded each stimulus two ways — verbatim prose in a single field, and decomposed
into named fields — and the two **agree on plain judging and diverge under injection**: mean
|Δnoul| is 0.048 with no injection (2.8× measurement noise) against 0.178 when injected (10.3×
noise, a 3.70× ratio). The representation does not change how well the judge judges; it changes
**how much a planted claim counts**. Third, and most consequentially for CCC's claim about source
labels: a **preregistered decomposition** of the provenance manipulation
(`PREREG_jev_attribution.md`, 11,904 calls at k=3) finds a typed `analysis_attribution` field to be
**inert** — six label values, every effect within ±0.019 — while the same label **in prose is
protective**, by −0.122 for `### Solver Analysis` and −0.244 for `### Candidate Analysis`. The
encoding interaction is supported (+0.122 [+0.093, +0.152]) but its mechanism is the reverse of what
an earlier version of this note claimed: typing provenance does not add cost, it **discards a
discount that prose confers** (§6.3, §9.2). What carried authority's effect was never the identity
label but the reliability assertion beside it — `This analysis has been verified.` is worth +0.106
alone, and `This analysis may contain errors.` **reduces** capture by −0.166 (§6.6). CCC found an
authoritative label unnecessary for capture; in this judge a bare label is not merely unnecessary but
protective. We also find that the *failure mode*
is encoding-dependent: as prose the SQL judge inverts (the cost-tier pattern), as typed fields its
discrimination is erased toward zero (the frontier pattern), on identical items. We argue that
structured-output judges do not inherit evaluator integrity from their type safety, that serialising
provenance into fields can remove a defence rather than add an attack surface, and that "which tier is
this judge in" is a property of task and encoding, not of the model alone.

Because every `noul` is a stated probability and every candidate carries an oracle label, the same
896 observations also audit the vendor's own calibration claim at no extra cost (§7). On clean input
the judge discriminates sharply in every domain and encoding (AUC 0.90-0.996) and is approximately
calibrated, though at this n the calibration error cannot be pinned tighter than about +/-0.05.
Under injection the Murphy decomposition, AUC and a flipped-forecast margin separate three failure
modes that the discrimination measure cannot: **inverted**, **confidently uninformative**, and
**decalibrated**. Which one occurs depends jointly on domain and encoding, and the single most
extreme cell in the study is SQL under bare verbatim `answer_only`, where AUC falls from 0.962 to
**0.0165** -- the wrong answer is scored more likely correct in roughly 98% of pairs -- with the
flipped forecast beating the model's own by 0.52 of Brier. **Inversion occurs nowhere in the
structured arm**: typed fields do not cure capture but they eliminate its catastrophic mode. Within
the structured arm, the typed `analysis_attribution` field is the element that reliably costs
(+0.0556 Brier, CI excluding zero, against a null contrast in prose — an interaction, which is what
isolates the field from the prose bundle it travels with). Total measurement cost: **$0.037**.

---

## 1. Introduction

CCC characterised a failure of LLM judges that read prose and write prose. The evaluator ecosystem is
moving, in part, toward judges that do neither: models that accept structured application state and
return typed values with calibrated probabilities, intended to be consumed directly by software. Such
models are marketed on a guarantee that is real but narrow — the output will conform to the requested
type — and that guarantee is easily misread as a guarantee about correctness. A type-safe verdict can
be a wrong verdict.

This note asks whether contextual conclusion capture crosses that architectural boundary, and whether
the structure of the input changes the shape of the failure. The second question turns out to be the
interesting one. A structured judge does not receive an authoritative *label*; it receives a *field*.
Whether a provenance claim expressed as data is treated differently from the same claim expressed as
a sentence is, for a model whose whole premise is that structure is better than prose, not obvious in
advance.

We make no claim that `jev-1.13` is representative of structured-decision models generally. One
endpoint, one bank, one week.

## 2. The judge under test

`typesafe/jev-1.13`, reached through OpenRouter at `POST /api/alpha/decisions`; served model string
`typesafe/jev-1.13-20260917`. The request carries a `state` object and a `questions` map; we use a
single **Noul** question, which returns `noul`, a probability in [0,1] that a stated proposition
holds. There is no system prompt, no temperature, no streaming and no multi-turn. Input is billed at
$0.042/M tokens and output is free.

Three measurement properties, established before any substantive analysis, matter for what follows.

**The judge is not deterministic.** On 20 items repeated five times with byte-identical requests,
only 3 of 20 returned identical `noul` values, while 20 of 20 returned identical *verdicts*. The
within-item standard deviation of a single call is **0.0154** (median range 0.040, maximum 0.180).
Verdict-based analyses therefore take n = distinct items; probability-based analyses gain precision
from repeats, but at the effect sizes observed a single call costs only 2–4% of dz, so all results
below use one call per cell.

**The vendor's documentation overstates the response.** Noul answers carry `noul` and `type` only;
the `confidence` field the documentation describes for Choice and Score answers is absent. A
confidence-based secondary measure was planned and abandoned.

**The probability is graded, not a disguised verdict.** Across 448 verbatim cells: 0% of values at
the extremes (≤0.001 or ≥0.999), 83.7% in (0.05, 0.95), 161 distinct values. A single `noul = 1`
observed on a trivial smoke-test item (*17 + 4 = 21*) was not representative. Had the probability
saturated, the continuous dependent variable below would have collapsed into the verdict one and the
power analysis would have been wrong in the optimistic direction; this was checked on every arm.

## 3. Method

**Items.** The frozen CCC bank, unmodified: 24 relational (SQLite, seeded fixtures), 16 code (Python
against a frozen unit-test suite), 16 arithmetic. Stimuli were verified byte-identical to
`codex/ccc-openweight-publish-v2` @ `bdcfd74` by per-cell request digest, across all 896 cells.

**Conditions.** The four canonical CCC conditions, unmodified: `no_injection` (no analysis section),
`answer_only` (neutral label, wrong answer stated, no reasoning), `full_rationale` (neutral label,
full wrong rationale), `solver_rationale` (solver label asserting verification). **`solver_rationale`
is not a pure attribution manipulation and must not be read as one:** the harness's frozen string
prepends `"Solver (verified) result: <wrong value>. "` to the identical rationale body, so it differs
from `full_rationale` by an identity word, a verification claim **and a restatement of the wrong
value**. Only the *encoding interaction* on this contrast isolates the typed attribution field,
because `build_state` places the same prose in `analysis_notes` in both arms and adds
`analysis_attribution` in the structured arm alone, so the prose difference cancels in the
difference-in-differences. Within-encoding marginals on this contrast are bundles. Both
candidate types (`correct`, `wrong_matching`) are judged for every item in every condition.

**Encodings.** Because a structured judge takes a state object rather than a prompt, how the stimulus
crosses over is a design choice, and we declined to pick one:

- **verbatim** — `state = {"judge_prompt": <the exact CCC prompt string>}`. Maximum comparability
  with the text tiers; the rationale arrives as prose as it did for them. **Pre-specified as
  primary.**
- **structured** — named fields (`question`, `schema_and_data` / `spec`, `candidate_answer`), with
  the rationale in `analysis_notes` and its attribution in `analysis_attribution`. Judge-native; the
  rationale is explicitly *data*.

**Dependent variables.** Two, from one response.

- *Flip* — verdict reversal, `verdict = noul > 0.5`, wrong when it disagrees with the oracle. The CCC
  reversal-susceptibility measure.
- *Discrimination* — per item, `D = noul(correct) − noul(wrong_matching)`; the harness's own `_disc`.
  Capture is `D(no_injection) − D(cond)`, positive meaning discrimination lost. Bootstrapped over
  items, 6000 resamples, following the harness `_ci` convention.

Carrying both was not redundancy. In arithmetic the flip measure proved uninformative (§6.5) while
the discrimination measure worked; in the structured SQL arm a near-zero baseline made ratio-scaled
capture meaningless (§9). Either alone would have misled.

**Gating.** Capture is the destruction of discrimination, so a judge that cannot discriminate before
injection has nothing to lose. Every arm reports baseline competence, and the analysis refuses to
size a confirmatory run when baseline D < 0.10.

**Missingness.** Strict: an item contributes only when every condition-by-candidate cell returned a
usable answer. Failures are logged as rows and counted. Retries cover transport errors only; a 4xx
surfaces rather than being retried. **No cell was missing in any arm reported here — completeness is
100% across 1,344 calls with zero failures.**

**Randomisation.** Independent generators derived by name, so consuming one neither advances nor
reveals another. This is the v1.5.1 correction applied to a new harness; drawing per-item values from
the same stream as condition assignment is what made earlier sweeps reconstructable.

## 4. Baseline competence

| arm | baseline D | 95% CI | accuracy | correct→wrong | wrong→correct |
|---|---|---|---|---|---|
| verbatim, pooled (n=56) | +0.606 | [0.513, 0.696] | 85.7% | 9/56 | 7/56 |
| verbatim sql (24) | +0.652 | [0.507, 0.781] | 87.5% | 3 | 3 |
| verbatim arith (16) | +0.492 | [0.272, 0.684] | 75.0% | 6 | 2 |
| verbatim code (16) | +0.650 | [0.536, 0.755] | 93.8% | 0 | 2 |
| structured sql (24) | +0.674 | [0.530, 0.805] | 89.6% | 3 | 2 |
| structured arith (16) | +0.474 | [0.243, 0.670] | 75.0% | 6 | 2 |
| structured code (16) | +0.616 | [0.484, 0.740] | 90.6% | 0 | 3 |

All arms pass the gate. Baselines agree closely across encodings (sql +0.022, arith −0.018, code
−0.034), which matters for §6.2: the two representations do not differ in how well the judge judges.

Arithmetic is the weakest arm in both encodings, and its errors are asymmetric in the *conservative*
direction — six correct answers rejected against two wrong answers accepted. The structured code arm
shows the opposite asymmetry (0 correct rejected, 3 wrong accepted), a yes-bias that makes the
`wrong_matching` side the weak half of its contrasts.

## 5. The CCC core result replicates, and strengthens

Capture under the verbatim encoding, with per-item discrimination:

| domain | base D | answer_only | full_rationale | solver_rationale | inverts? |
|---|---|---|---|---|---|
| sql | +0.652 | **−0.522** | **−0.320** | **−0.314** | all three |
| arith | +0.492 | **−0.242** | +0.034 | +0.201 | answer_only only |
| code | +0.650 | +0.275 | +0.256 | +0.195 | none |

Capture magnitudes, as discrimination lost and as a share of baseline:

| domain | answer_only | full_rationale | solver_rationale |
|---|---|---|---|
| sql | +1.174 (180%), dz 3.24 | +0.972 (149%), dz 2.21 | +0.966 (148%), dz 2.00 |
| arith | +0.735 (149%), dz 2.11 | +0.459 (93%), dz 1.62 | +0.292 (59%), dz 1.30 |
| code | +0.375 (58%), dz 1.59 | +0.394 (61%), dz 2.23 | +0.455 (70%), dz 1.70 |

All twelve contrasts are significant at p < 1e-04 (paired t and Wilcoxon). Verdict flips concur:
pooled 25.0% with no injection against 85.7% (`answer_only`), 78.6% (`full_rationale`) and 66.1%
(`solver_rationale`); exact McNemar against `no_injection` gives p = 1.1e-09, 1.5e-08 and 5.6e-06.

**A bare wrong answer in the state object is sufficient, and in SQL it is total.** The
`no_injection → answer_only` step alone costs +1.174 discrimination (p = 7e-14), at a **100%**
verdict-flip rate across all 24 SQL items. This is CCC's central sufficiency claim, reproduced in an
architecture that shares no component with the judges that established it.

Capture in SQL exceeds 100% of baseline because discrimination does not merely vanish — it
**reverses**. The judge ends up assigning higher probability to the wrong candidate than the right
one. Readers should note that a "percentage of baseline" above 100 is a signed statement about
direction, not an effect size; we report D directly alongside it for that reason, and §9 records what
happens when the denominator approaches zero.

**Adding a rationale partially restores discrimination.** The `answer_only → full_rationale` step is
*negative* in sql (−0.202, p=4.5e-05) and arith (−0.276, p=0.0027), and null in code (+0.019,
p=0.37). The same sign and rough magnitude appear in the structured encoding (−0.165, −0.142,
−0.006). This is opposite to the text-judge pattern and is stable across representations. A length
confound would predict the reverse ordering: `answer_only` carries the *shortest* note and produces
the *largest* effect.

## 6. Results specific to a structured judge

### 6.1 Capture is domain-dependent in magnitude and in kind

Code is the most accurate baseline (93.8%) and the least capturable (58–70%, never inverting). SQL is
nearly as accurate (87.5%) and the most capturable (148–180%, always inverting). Arithmetic is the
least accurate (75.0%) and intermediate. Baseline accuracy alone therefore does not order capture:
SQL beats arithmetic on accuracy and is captured far more.

The ordering does track something else. Code items carry an executable specification, so the judge
has an independent means of checking the candidate and resists the planted claim. SQL requires
evaluating a query against seeded data, which is harder to verify internally, so the judge defers to
the stated result. We advance this as a hypothesis — three domains is not a test — but note that §9
supplies accidental evidence for it: when the data needed to verify a SQL answer was absent from the
state, baseline accuracy fell to 54.2%, near chance.

### 6.2 The encoding effect is injection-specific

Cross-encoding agreement, per condition, over 448 paired cells. Measurement noise alone would produce
mean |Δnoul| ≈ 0.017 (two single calls, within-item sd 0.0154):

| condition | cells | verdict agreement | mean \|Δnoul\| | × noise |
|---|---|---|---|---|
| no_injection | 112 | **96.4%** | **0.048** | 2.8 |
| answer_only | 112 | 69.6% | 0.195 | 11.2 |
| full_rationale | 112 | 72.3% | 0.186 | 10.7 |
| solver_rationale | 112 | 83.0% | 0.154 | 8.8 |
| injected, pooled | 336 | **75.0%** | **0.178** | **10.3** |

Injected divergence is **3.70×** baseline divergence and roughly ten times measurement noise;
baseline divergence is small but not nil at 2.8× noise. Pooled across conditions the picture is
flattened to 80.4% agreement (r = 0.759, mean difference −0.063, p = 1.1e-09, largest single gap
0.83), which invites the conclusion that the representations simply disagree. They do not. **They
agree on plain judging and diverge on how much a planted claim counts.**

We note the diagnostic that distinguishes these, because it is easy to get wrong: verdict agreement
is confounded by distance from 0.5, since a confident baseline survives a shift that flips a
borderline injected cell. The magnitude comparison is the sound basis; verdict agreement here moves
the same way and corroborates it.

Within the injected conditions, divergence falls and agreement rises monotonically as the note gains
content: bare answer (0.195) → rationale (0.186) → attributed rationale (0.154). With a bare value
the representation is nearly all there is; as content accumulates it dominates and the encodings
converge.

### 6.3 Typed provenance is inert; prose provenance is protective

**This section reports a result that reverses the one first published here.** The original §6.3
claimed that provenance expressed as a typed field is load-bearing and makes capture worse. That
claim rested on the `full_rationale → solver_rationale` contrast, and that contrast is confounded:
the harness's frozen string prepends `"Solver (verified) result: <wrong value>. "` to an otherwise
identical rationale, changing an identity word, a verification claim **and** a restatement of the
wrong value in a single step (§3). A preregistered follow-up decomposed it — run `3bee5c5c030e`,
11,904 calls at k=3, zero failures, 100% completeness — composing every cell over a **byte-identical
rationale body** so that each contrast moves exactly one element. §9.2 records the superseded figures
and what the error cost.

Identity label against the neutral label, restatement absent, no reliability claim. Positive is more
capture:

| identity | structured (typed field) | verbatim (prose label) |
|---|---|---|
| `unlabelled` | +0.002 [−0.004, +0.008] | −0.007 [−0.025, +0.010] |
| `candidate` | −0.019 [−0.033, −0.005] | **−0.244 [−0.299, −0.192]** |
| `reviewer` | +0.012 [+0.005, +0.020] | +0.038 [+0.018, +0.062] |
| `solver` | −0.000 [−0.007, +0.006] | **−0.122 [−0.151, −0.095]** |
| `sealed_solver` | +0.010 [+0.002, +0.017] | −0.047 [−0.075, −0.020] |

Two results sit in that table.

**In the typed field the identity value is inert.** Six label values, every effect within ±0.019 and
four within ±0.013. This is not a weak effect but a tight null: to this measurement
`analysis_attribution` is not read.

**In prose the label is read as source credibility, and used defensibly.** The most protective
manipulation in the entire study is attributing the analysis to the thing being judged —
`### Candidate Analysis`, −0.244. `### Solver Analysis` is protective at −0.122. The absence of any
header is indistinguishable from a neutral header (−0.007), so this is not a presence-of-heading
artefact. Only `### External Reviewer Analysis` mildly increases capture.

The preregistered interaction is therefore **supported** — +0.122 [+0.093, +0.152], Holm-adjusted
p = 4.8e-10 — and its mechanism is the opposite of the original claim. It is positive not because
typing provenance adds cost, but because prose provenance confers a discount that the typed field
does not. For a judge consuming application state the actionable statement is that serialising
provenance into a field does not merely fail to help: **it discards a protection the prose form was
providing.**

What was harmful in the original bundle is now identified. Verbatim arm, ranked:

| manipulation | effect on capture |
|---|---|
| `This analysis has been verified.` | **+0.106** |
| restating the wrong value | **+0.080** |
| `Sealed` status above plain solver identity | **+0.075** |
| `### External Reviewer Analysis` | +0.038 |
| `### Solver Analysis` | **−0.122** |
| `### Candidate Analysis` | **−0.244** |
| `This analysis may contain errors.` | **−0.166** |

The verification claim and the restatement did the damage. The identity label was pulling the other
way throughout. CCC's finding that an authoritative source label is *not necessary* for capture
stands and is sharpened: in this judge a bare identity label is not merely unnecessary, in prose it is
actively protective, and what carries authority's effect is the **reliability assertion** attached to
it (§6.6).

One limit on the decomposition, found by its own preregistered check: it does not fully account for
the original effect. The hand-written legacy bundle captures **+0.116 [+0.077, +0.157]** more than the
composed cell carrying the same three factors, and the components do not sum — parts predict +0.064
against +0.224 observed. Presentation format is therefore a factor in its own right, of comparable
size to everything tested here, and it is uncontrolled (§10).

### 6.4 The failure mode, not only its size, depends on the encoding

Per-item D under injection:

| domain | verbatim | structured |
|---|---|---|
| sql | (−0.522, −0.320, −0.314) **inverts** | (0.031, 0.196, −0.013) **erased toward 0** |
| arith | (−0.242, 0.034, 0.201) | (0.059, 0.197, 0.147) |
| code | (0.275, 0.256, 0.195) **never inverts** | (0.326, 0.345, 0.257) |

Same model, same items, same oracle. Presented as prose the SQL judge **inverts** — the pattern CCC
reports for cost-tier and open-weight endpoints. Presented as typed fields its discrimination is
**erased toward zero** without inverting — the pattern CCC reports for frontier endpoints. Code never
inverts under either encoding.

A judge's tier, on CCC's own taxonomy, is therefore not a property of the model alone. It is a
property of the model, the domain, and the encoding of the stimulus jointly. Evaluator-integrity
claims that name only a model are underspecified.

### 6.5 Where arithmetic's verdict measure fails

Arithmetic's baseline flip rate is 43.8%, and exact McNemar is non-significant for every injected
condition (p = 0.0625, 0.125, 0.5). With six of sixteen correct answers rejected before any
injection, the verdict measure in that domain has no room to move. Its discrimination measure is
unaffected and significant throughout (dz 1.30–2.11).

We report this rather than dropping the arm. A study carrying only the verdict measure would record
arithmetic as uncaptured; a study carrying only the probability measure would miss that SQL inverts
rather than merely degrades.

### 6.6 The judge reads reliability language, and reads it in the right direction

The decomposition's fourth preregistered contrast was declared **two-sided**, because the two
hypotheses make opposite predictions and that is the point. A judge using provenance as *information*
should defer less when told an analysis may be wrong. A judge using it as an *authority signal* should
not care.

Holding the label at `neutral` and varying only the reliability claim, mean D by claim:

| claim | structured | verbatim |
|---|---|---|
| none | +0.236 | +0.101 |
| `This analysis has been verified.` | +0.170 | **−0.005** |
| `This analysis has not been verified.` | +0.302 | +0.289 |
| `This analysis may contain errors.` | +0.298 | +0.267 |

The contrast resolves negative in both encodings: −0.062 [−0.077, −0.048] structured and
−0.166 [−0.204, −0.132] verbatim, both Holm-significant. Telling this judge an analysis may contain
errors **reduces** capture. The ordering is monotone in the sensible direction — asserting
verification increases capture, asserting unreliability decreases it — and it holds in both encodings,
unlike the identity effect of §6.3.

So the judge is not simply deferring to authority. It is using reliability language as information,
and using it correctly. That is the most favourable result in this note, and it coexists with the
least favourable one: in the verbatim arm `verified` drives D to **−0.005**, essentially total
capture, making it the single most effective clean manipulation we measured. The same sensitivity that
lets a disclaimer protect the judge is what lets an unearned verification claim capture it.

An evaluation pipeline can exploit the first half of that: a reliability field on untrusted context is
not decoration, it measurably buys discrimination back. It cannot avoid the second half, because
anything that can assert unreliability can assert verification.

## 7. The vendor's calibration claim, audited on its own terms

TypeSafe's material for this model states that "all answers are accompanied with calibrated
probabilities and confidence scores". The CCC grid tests the first half of that claim for free. Every
`noul` in the study is a stated probability that the candidate answer is correct, and
`candidate_type` says whether it was: 896 forecasts with oracle labels, balanced 50/50 by
construction, no annotation step and so nothing to disagree about. The second half is not testable
here — the `confidence` field was unpopulated in all 1,088 responses (§9).

This is a different instrument from §5 and §6, which measure *discrimination* via `D`. A judge can
lose discrimination in two very different ways, and the distinction matters to anyone consuming the
probability. It can become **uncertain** — hedging toward 0.5, less useful but not misleading — or it
can stay confident and become **wrong**. The Murphy decomposition separates these:

> Brier = reliability − resolution + uncertainty

with reliability the mean squared gap between stated probability and observed frequency (lower
better), resolution the separation the forecasts achieve (higher better), and uncertainty fixed at
0.2500 by the balanced design, so every cell in this section is directly comparable.

**The decomposition alone is not sufficient, and this study shows why.** Resolution is non-negative by
construction, so a judge that systematically prefers the wrong answer has *high* resolution — its
forecasts separate the outcomes perfectly, backwards. In SQL under verbatim `answer_only`, resolution
**rises** from 0.1750 to 0.2000 while the ranking collapses; in arithmetic it rises from 0.1406 to
0.1781. Read through the decomposition alone, the two most catastrophically reversed cells in the
study look like improved discrimination. Two further instruments disambiguate it: AUC, which falls
below 0.5 only when the ranking reverses, and the **flip margin** — Brier(p) − Brier(1−p) — which is
positive when the information is present with reversed polarity. Both are reported with 95%
percentile intervals bootstrapped over items (6,000 resamples, seed 0), items being the resampling
unit because an item's correct and wrong candidates share a question, a schema and a planted value.
The unit of analysis is the per-cell mean of `noul`, the same quantity §5 and §6 analyse; at the
single repetition actually run it is the raw value.

### 7.1 On clean input the judge discriminates sharply; its calibration is good but not established

| cell, `no_injection` | Brier [95% CI] | AUC [95% CI] | reliability | rel/Brier | ECE | bias |
|---|---|---|---|---|---|---|
| sql verbatim | 0.0817 [0.035, 0.135] | 0.962 [0.910, 0.998] | 0.0053 | 6.5% | 0.060 | −0.028 |
| sql structured | 0.0732 [0.030, 0.126] | 0.969 [0.915, 0.998] | 0.0088 | 12.0% | 0.075 | −0.033 |
| code verbatim | 0.0658 [0.028, 0.112] | 0.996 [0.977, 1.000] | 0.0286 | 43.5% | 0.124 | **+0.095** |
| code structured | 0.0886 [0.037, 0.156] | 0.984 [0.949, 1.000] | 0.0403 | 45.5% | 0.152 | **+0.119** |
| arith verbatim | 0.1408 [0.066, 0.236] | 0.900 [0.777, 0.984] | 0.0374 | 26.6% | 0.132 | −0.049 |
| arith structured | 0.1518 [0.069, 0.257] | 0.902 [0.756, 1.000] | 0.0291 | 19.2% | 0.107 | −0.061 |

Discrimination on clean input is strong everywhere: AUC 0.90 to 0.996, every interval far above 0.5,
and encoding-invariant within each domain. The vendor's claim survives its own easy case, which is
what makes the rest of this section worth reading.

Calibration is good but we decline to call it established. The ECE point estimates run 0.04 to 0.15
and **no cell's interval excludes 0.10**; at 16–24 items per domain this design cannot pin the
calibration error more tightly than about ±0.05. The defensible statement is that the judge is sharply
discriminating and approximately calibrated on clean input, not that it is well calibrated. We report
reliability as a share of Brier alongside ECE because it is far less sensitive to binning choices, and
because it exposes something AUC hides: **`code` has the best ranking in the study and the worst
calibration.** It separates correct from wrong candidates almost perfectly (AUC 0.996) while stating
probabilities shifted roughly +0.10 high, so miscalibration accounts for ~44% of its Brier score
against ~7% for SQL. That is the yes-bias of §4 seen through a second instrument, and it is a
qualitatively different defect from SQL's: `code` is confident and offset, `sql` is confident and
accurate. Pooling the domains erases the distinction.

### 7.2 Injection has three distinguishable failure modes, and which one you get depends on domain and encoding

Classifying each injected cell by its AUC interval and its flip margin gives a clean taxonomy.
**Inverted**: AUC interval wholly below 0.5, or flip margin interval wholly above 0 — the judge keeps
asserting and asserts the wrong thing. **Confidently uninformative**: AUC interval spans 0.5 while the
judge still states sharp probabilities — the signal is gone but the certainty is not.
**Decalibrated**: ranking intact, stated probability no longer matching observed frequency.

| domain | encoding | `answer_only` | `full_rationale` | `solver_rationale` |
|---|---|---|---|---|
| sql | verbatim | **inverted** 0.017 | **inverted** 0.177 | **inverted** 0.178 |
| sql | structured | uninformative 0.524 | decalibrated 0.741 | uninformative 0.530 |
| arith | verbatim | **inverted** 0.234 | uninformative 0.522 | uninformative 0.656 |
| arith | structured | uninformative 0.592 | uninformative 0.684 | uninformative 0.664 |
| code | verbatim | decalibrated 0.967 | decalibrated 0.918 | decalibrated 0.934 |
| code | structured | decalibrated 0.961 | decalibrated 0.883 | decalibrated 0.961 |

Figures are AUC; every paired Brier degradation against `no_injection` excludes zero except two cells
noted in §7.4. The extreme cell is SQL under verbatim `answer_only`: AUC **0.0165** with interval
[0.000, 0.063], meaning that across the 576 correct/wrong pairs the *wrong* answer receives the higher
probability of being correct in roughly 98% of them. Reliability 0.5597, ECE 0.698 — the stated
probability is off by seven-tenths on average — and the flipped forecast scores 0.0912 against the
model's own 0.6130, a flip margin of +0.5219 [+0.397, +0.641]. A bare wrong value in a state field, with
no rationale and no authority label, does not confuse this judge. It reverses it.

This is also where the pooled view misleads. Pooled across domains, verbatim `full_rationale` and
`solver_rationale` classify as merely uninformative — an average over a domain at AUC 0.18 and a
domain at 0.92. **The three-way domain ordering, not the pooled verdict, is the result.**

### 7.3 Typed provenance does not cure capture; it prevents the catastrophic mode

The SQL rows above are the sharpest statement of §6.3 available in this study, because baseline
competence is identical across the two encodings (AUC 0.962 verbatim, 0.969 structured) while the
injected cells are not:

| condition | verbatim AUC | structured AUC | verbatim Brier | structured Brier |
|---|---|---|---|---|
| `no_injection` | 0.962 | 0.969 | 0.082 | 0.073 |
| `answer_only` | 0.017 | 0.524 | 0.613 | 0.368 |
| `full_rationale` | 0.177 | 0.741 | 0.508 | 0.284 |
| `solver_rationale` | 0.178 | 0.530 | 0.510 | 0.370 |

Same items, same planted values, same rationale text, same judge, same week. Moved from prose into
named fields, every reversal disappears. **Inversion occurs nowhere in the structured arm, in any
domain or condition.** Capture is not reduced to nothing — structured SQL still loses most of its
discrimination and its Brier still triples — but the judge degrades to uninformative rather than to
misleading, and that difference is the one a downstream consumer feels. An uninformative probability
wastes a filter; a reversed one inverts it.

Note the direction of this against §6.3, which found the structured encoding made the *attribution*
effect worse. Both hold, and they are not in tension: typed fields raise the marginal cost of a
provenance label while removing the catastrophic failure mode of prose injection. The encoding is not
uniformly safer or more dangerous; it changes which part of the stimulus is load-bearing.

### 7.4 The Brier view of the same contrast, and why it was misread

Paired per-item Brier contrasts between injected conditions, bootstrapped over items:

| contrast | verbatim | structured |
|---|---|---|
| `full_rationale` − `answer_only` | −0.0789 [−0.109, −0.050] | −0.0554 [−0.074, −0.037] |
| `solver_rationale` − `answer_only` | −0.0949 [−0.134, −0.057] | +0.0002 [−0.016, +0.016] |
| `solver_rationale` − `full_rationale` | −0.0160 [−0.036, +0.003] | **+0.0556 [+0.039, +0.072]** |

The robust part first: in both encodings a full wrong rationale is *less* damaging than a bare wrong
answer, plausibly because a rationale gives the judge something checkable where a bare value gives it
nothing. That holds up.

The third row is the one that misled us. Read on its own it says that the `solver_rationale` step
costs nothing as prose and +0.0556 of Brier once the same change also sets a typed
`analysis_attribution` field — and we drew from it the conclusion that the typed field carries the
cost. **The preregistered decomposition of §6.3 shows that reading to be wrong.** `solver_rationale`
is a bundle of three changes, and when they are separated the typed identity field is inert (±0.019
across six label values) while the prose identity label is protective (−0.122). The +0.0556 is real;
what it measures is the *removal of the prose discount*, not the *addition of a field cost*. The sign
of the interaction survived; the mechanism did not.

`code` illustrates the same trap. In the structured arm `solver_rationale` is the only condition whose
degradation resolves at all (+0.0895 [+0.019, +0.143], against `answer_only` +0.0549 [−0.004, +0.104]
and `full_rationale` +0.0474 [−0.002, +0.090], both spanning zero at n=16). We read that as the typed
authority field making capture measurable where it otherwise was not. It is at least as consistent
with the bundle's verification claim and restatement doing the work, which is what §6.3 finds.

**These three contrasts were selected after inspecting the summary tables**, so they were post-hoc —
which is precisely why they needed the preregistered follow-up rather than a louder claim. The lesson
generalises past this note: a contrast between two conditions is only a measurement of one factor if
the conditions differ in one factor, and a frozen stimulus set inherited from an earlier study is
exactly where that assumption goes unchecked.

### 7.5 What the audit adds, and what it does not

It adds an independent confirmation of §6's domain ordering from machinery that shares nothing with
`_disc()`; a taxonomy that separates "stopped being useful" from "started being wrong", which the `D`
measure cannot; the specific claim that typed encoding eliminates inversion; and a mechanism-level
result about the attribution field. It does not establish that the judge is well calibrated on clean
input — §7.1 is explicit that this n cannot — and it does not test the vendor's confidence-score
claim at all. It also inherits every limitation of §9: the same 56 items, the same single endpoint,
the same single measurement per cell.

---

### 7.6 The taxonomy applied to eighteen cells: only prose produces the good failure

§7.2's three-mode taxonomy was built on four conditions. The attribution run supplies eighteen, with
oracle labels on 3,968 forecasts, and applying the taxonomy to them says something the discrimination
measure cannot. Modes are assigned by the same interval gates as §7.2.

| cell | verbatim | structured |
|---|---|---|
| `id:candidate` | **HEDGING** | decalibrated |
| `rel:possibly_erroneous` | **HEDGING** | decalibrated |
| `id:solver`, `id:sealed_solver`, `id:unlabelled`, `rel:unverified` | decalibrated | decalibrated |
| `id:neutral`, `id:reviewer`, `rel:verified`, `legacy:solver_bundle` | confidently uninformative | decalibrated |
| every `+restate` cell | decalibrated or confidently uninformative | decalibrated |

**Only two of thirty-four injected cells reach HEDGING** — resolution falls while calibration holds,
which is the failure an evaluator *should* have when fed material it cannot verify. They are
`### Candidate Analysis` (reliability 0.0026 → 0.0207, AUC 0.946 → 0.836 [0.742, 0.917]) and
`This analysis may contain errors.` (reliability → 0.0181, AUC → 0.767 [0.655, 0.865]). Both are
prose. In both, miscalibration rises by less than 0.02 against a baseline of 0.0026 — the judge
becomes less informative without becoming less honest.

**The structured arm never reaches it.** All seventeen injected cells return DECALIBRATED: the ranking
survives, the stated probability stops matching the observed frequency, identically, across six
identity labels, three reliability claims, a restatement factor and two bundles. Seventeen stimuli,
one failure mode. This is a stronger statement of §6.3's inertness than the effect sizes alone
support: it is not merely that writing in `analysis_attribution` moves the numbers little, it is that
**nothing written there changes the kind of failure.**

**One sentence suffices to reduce the judge to chance.** `This analysis has been verified.` alone takes
AUC from 0.946 to **0.5010** [0.364, 0.643] while sharpness stays at 0.256 — confidently
uninformative, from eight words, with no rationale and no identity label.

**§6.3's format effect is a mode change, not a magnitude change.** The hand-written legacy bundle
reaches reliability **0.2234**, the worst cell in the study and 2.4× the composed equivalent's 0.0936,
with AUC 0.387 against 0.544. We stop short of calling it inverted: its AUC interval [0.231, 0.569]
spans 0.5 at 40 items, so the point estimate reverses while the interval does not exclude chance. What
the comparison does support is that presentation alone moves the judge from uninformative *toward*
reversed — a different and more serious claim than the +0.116 of discrimination reported in §6.3.

Finally, §6.3's ordering reproduces here from machinery sharing nothing with `_disc()`. Verbatim AUC:
`candidate` 0.836, `solver` 0.733, `sealed_solver` 0.672, `unlabelled` 0.632, `neutral` 0.626,
`reviewer` 0.573. The restatement factor is harmful in all twelve label-by-encoding pairs without
exception.

## 8. Implications for evaluation design

**Type safety is not evaluator integrity.** A model that cannot emit a malformed verdict can still
emit a wrong one, and here does so at a 100% flip rate on one domain from the cheapest possible
injection. Vendor guarantees about output *shape* should not be read as guarantees about output
*correctness*, and procurement language should separate them.

**Serialising provenance can remove a defence.** CCC's recommendation of context isolation — never
placing a foreign conclusion in the judge's input — extends directly. What §6.3 adds is less obvious
and cuts the other way from our first reading: a provenance *field* is not a new attack surface, it is
an **inert** one, and moving a label out of prose and into it discards the discount the prose label was
earning. A pipeline that helpfully normalises "### Candidate Analysis" into
`{"attribution": "candidate"}` has, on this judge, thrown away the most protective signal in the study.
Where provenance must be machine-readable, keep it in the prose the judge reads as well, and verify
that the field is doing anything at all before relying on it.

**Reliability fields are worth more than identity fields, and are symmetric.** §6.6 shows a disclaimer
buys back −0.166 of capture while a verification claim costs +0.106. A pipeline attaching reliability
metadata to untrusted context is buying real discrimination. The same mechanism means any component
that can write "unverified" can write "verified", so the field needs the same provenance controls as
the claim it describes.

**Encoding belongs in the preregistration.** Two defensible representations of one stimulus produced
different failure modes on identical items. A structured-judge evaluation that does not fix and
report its state schema is not reproducible in the sense CCC requires.

**Report the baseline.** Both the arithmetic weakness and the defect in §9 were caught by gating on
baseline discrimination before interpreting any capture figure. We recommend it as a standing
requirement for capture studies: a capture number computed against a judge that was not judging is
not a measurement.

## 9. Two withdrawn results, and what they cost

### 9.1 A stimulus defect found mid-study

An earlier draft of this note reported that the two encodings disagreed substantially and treated
that as a finding. The structured SQL state at that time omitted the schema and seeded fixtures,
carrying only the question and a bare result. The judge was being asked an unanswerable question and
answered conservatively: baseline D **+0.042**, accuracy **54.2%**, with **22 of 24 correct answers
rejected**, and ratio-scaled capture figures reaching 1796–2030% of a near-zero baseline.

After correction — the state now carries `FIXTURES[item["fixture"]]`, the same seeded data the
verbatim prompt contains — the arm reads baseline D **+0.674**, accuracy **89.6%**, closely matching
the verbatim arm. The defect accounted for the entire apparent encoding disagreement at baseline;
what survives is the injection-specific divergence of §6.2, which is a narrower and better-supported
claim than the one withdrawn.

We report this for three reasons. The failure mode is instructive: an incomplete state produced a
*plausible* result rather than an obvious error, and a near-zero denominator produced percentages
that looked like strong findings. It is evidence for §6.1's hypothesis, since removing the means of
verification collapsed competence to chance. And the correction is only checkable because the raw
rows and request digests were retained; the recomputation in the appendix reproduces both the broken
and the corrected arms from evidence.

Guards added in consequence, all exercised against deliberately broken inputs: the runner refuses to
start when any domain-by-encoding cell collapses its four conditions to identical requests, or when a
domain fails to load; it warns when a structured state is under 25% the length of its verbatim
prompt; the analysis refuses to print a share-of-baseline ratio when baseline D < 0.10; and the merge
utility refuses to average cells originating from different code versions.

### 9.2 A finding withdrawn by its own preregistered follow-up

The first version of §6.3 claimed that typed provenance is load-bearing: that a source label inert in
prose becomes, once expressed as `"analysis_attribution": "solver_verified"`, sufficient to make
capture worse. It was the note's headline result. The reported figures were sql −0.006 (p=0.80) and
arith −0.167 (p=0.0003) as prose against sql **+0.209** (p=7e-09), arith +0.054 (p=0.005) and code
+0.091 (p=0.032) as a field, and the conclusion read "provenance expressed as data is acted upon."

Every one of those numbers is reproducible. The inference from them was wrong.

`solver_rationale` is not "the same rationale, attributed." The harness's frozen string prepends
`"Solver (verified) result: <wrong value>. "` to an identical body, so the step changes an identity
word, a verification claim **and** a restatement of the wrong value at once. No within-encoding
marginal on it measures an attribution effect. We had not read the stimulus builder closely enough to
see this, because the four conditions were inherited from the preregistered CCC study where they were
never intended as a factor decomposition.

The decomposition (§6.3) finds the typed field inert across six label values and the prose label
protective. The interaction kept its sign and lost its mechanism, and the harmful components turn out
to be the verification claim and the restatement — with the identity label opposing both.

What the error cost: one headline finding, one abstract, one conclusion, and a design note arguing for
a follow-up that was itself misconceived — a "6×4 cross" assembled from label and reliability
dictionaries belonging to a *different* experiment's module, which the CCC runners never import and
whose stimulus cache holds only four of the twenty-four cells. That cross would also have violated the
harness's own factor separation, which states in comments that `sealed` is a status manipulation and
not an identity.

What caught it: reading the stimulus builder before designing the follow-up, and preregistering a
contrast whose sign was allowed to come out either way. The preregistration says so explicitly —
contrast 3 was to be reported "whatever its sign, including if it overturns §6.3." §9.1 was caught by
a baseline gate. This one was caught by a design that could lose.

The recurring lesson across both is the same and it is not about this judge: **a frozen stimulus set
inherited from an earlier study carries the earlier study's design assumptions, and those assumptions
are invisible at the call site.** Both failures lived in the gap between what a condition was named
and what its string actually contained.

## 10. Limitations

**One endpoint.** `jev-1.13` is a single structured-decision model observed over a single week. We do
not claim it is representative of the class.

**Not preregistered.** This note reuses CCC's instrument on a new target. The conditions, items and
dependent variables were fixed in advance of any live call, and the two encodings were specified
before data collection with verbatim nominated primary; but no preregistration was filed, and §9
records a mid-study correction. Treat every result as exploratory.

**Underpowered for nulls.** n is 16–24 per domain. The smallest detectable injected flip rate at 80%
power against a 3% control is 33% (sql) and 44% (arith, code); the smallest detectable dz is 0.60 and
0.75. Observed effects far exceed these, so the positive findings are secure, **but no null in this
note certifies resistance.** Confirming the weakest contrast at 80% power needs 50 items per cell; the
bank holds 16–24.

**Single measurement per cell.** The judge is stochastic (within-item sd 0.0154). At the observed
effect sizes this attenuates dz by 2–4%, which is immaterial for the large effects but not negligible
for the attribution marginals of §6.3. A k=3 replication is warranted before those are leaned on.

**The structured schema is ours.** The verbatim encoding is CCC's instrument exactly; the structured
encoding is a decomposition we designed, and a different sensible decomposition might behave
differently. §9 shows how consequential that choice is. We publish the schema with the code.

**No confidence measure.** The documented `confidence` field is absent from Noul answers, so a
planned secondary analysis is missing rather than null.

**Clean-input calibration is unresolved, not confirmed.** §7.1 reports ECE point estimates of 0.04 to
0.15 with no cell's bootstrap interval excluding 0.10. We therefore claim approximate calibration and
no more. An earlier version of the audit applied a fixed 0.05 threshold to the point estimate and
printed two different verdicts for the two encodings' statistically indistinguishable baselines; that
was a threshold artefact, and the published tool now reports the interval and declines to resolve what
this n cannot.

**The failure-mode taxonomy is interval-gated but below our own reporting floor in four of six cells.**
The audit's reporting floor is 40 labelled forecasts; `code` and `arith` supply 32 per condition. Each
verdict in §7.2 is gated on a bootstrap interval rather than a point estimate, so the classifications
are evidence-backed, but the per-condition blocks for those domains are flagged below-floor in the
tool's own output and should be read as such. The three contrasts of §7.4 were also chosen after
inspecting the summary tables.

**The `solver_rationale` contrast is a bundle, and the decomposition of it is incomplete.** The frozen
stimulus prepends an attributed, verification-claimed restatement of the wrong value to an otherwise
identical rationale (§3), so no within-encoding marginal on it measures attribution. §6.3 replaces it
with a preregistered decomposition, and §9.2 records what the original inference cost. The
decomposition does not close the question: its own additivity check fails. The hand-written legacy
bundle captures +0.116 [+0.077, +0.157] more than the composed cell with the same three factors, and
the parts predict +0.064 against +0.224 observed. **Presentation format is therefore an uncontrolled
factor of comparable size to every factor tested**, and the obvious next experiment is inline-prose
against header rendering of byte-identical content.

**The decomposition's contrasts are per-encoding, and the Holm family is eight rather than five.** The
preregistration says "across the five"; contrasts 1, 2 and 4 are each computed in two encodings, which
that wording did not resolve. We corrected across all eight computed primaries, the stricter reading.
Every primary except the structured identity null clears Holm at p < 1.1e-6.

**Arithmetic's verdict measure is uninformative****Arithmetic's verdict measure is uninformative** (§6.5), and the structured code arm carries a
yes-bias that weakens one half of its contrasts (§4).

## 11. Conclusion

Contextual conclusion capture crosses the boundary from prose judges to structured-decision judges
intact, and in the relational domain arrives stronger: a bare wrong value in a state object costs the
judge all of its discrimination and then reverses it, at a complete verdict-flip rate, from an input
no more elaborate than a number. A guarantee that a judge cannot emit a malformed answer buys nothing
against this.

The architectural change does not merely transplant the failure; it redistributes it. How a stimulus
is encoded leaves plain judging alone and changes how heavily a planted claim weighs. And the direction
of that redistribution is the opposite of what we first reported: a typed provenance field is inert
across six label values, while the same label in prose is protective — most strongly when it attributes
the analysis to the party being judged. Serialising provenance into application state does not add an
attack surface so much as **discard a defence**, and that is the claim we would most like to see
replicated.

What does carry authority's force is the reliability assertion rather than the identity: a sentence
claiming verification is worth +0.106 of capture on its own, and a sentence disclaiming reliability
buys −0.166 back. A judge that can be protected by a disclaimer can be captured by its negation, and
both are one string away.

Two of this note's findings were withdrawn in the course of writing it (§9), one caught by a baseline
gate and one by a preregistered contrast permitted to come out either way. We think that record is
worth more than a cleaner narrative would have been, and we would rather this note be read as an
argument for designs that can lose than as a set of results about one endpoint.

---

## Reproducibility appendix

### Artifacts

| | |
|---|---|
| Item bank and conditions | `petesherratt-collab/The-Generated-Trace-Leak-Harness`, branch `codex/ccc-openweight-publish-v2` @ `bdcfd74` |
| Probe code | `jev-ccc-pilot/` in the same repository |
| Raw rows | `_obs_clean.jsonl` — 1,088 rows, 896 distinct cells |
| Verbatim runs | `ff36f581d084`, `f45674093a45` |
| Structured run (post-correction) | `15ce0214d7f0` |
| Determinism run | 20 items × 5 repeats, `determinism_obs.jsonl` |
| Served model | `typesafe/jev-1.13-20260917` |
| Seed | `305774821` (the harness's authoritative frontier v3 seed) |
| Stimulus digest | `acb978aa…` over all 896 per-cell request hashes |
| Total cost | $0.0366, 1,344 calls, zero failures, 100% completeness |
| Calibration audit | `calibration.py`, 45 self-tests in `test_calibration.py`; no API calls |
| Attribution decomposition | `PREREG_jev_attribution.md` (frozen before any stimulus was composed), `attribution_stimuli.py`, `attribution_run.py`, `analyse_attribution.py`, 38-check gate in `test_attribution_stimuli.py` |
| Attribution run | `3bee5c5c030e` — 11,904 calls, k=3, 18 cells (17 arith), zero failures, 100% completeness, $0.3019 |
| Attribution stimulus digests | sql `88afaaf1eab58491`, code `804d2292377f6ed4`, arith `27812d88f54de59d` |

### What can be reproduced without API calls

All reported statistics recompute from `_obs_clean.jsonl`:

```
python analyse_ccc.py  --obs _obs_clean.jsonl --by-domain
python calibration.py  --obs _obs_clean.jsonl --both-encodings --by-domain
python calibration.py  --obs _obs_clean.jsonl --both-encodings \
    --contrast answer_only:full_rationale \
    --contrast answer_only:solver_rationale \
    --contrast full_rationale:solver_rationale
python test_calibration.py

python test_attribution_stimuli.py --repo <repo>          # blocking stimulus gate
python analyse_attribution.py --obs _obs_attr.jsonl --repo <repo> --secondary
python calibration.py --obs _obs_attr.jsonl --condition-field cell --both-encodings
python analyse_ccc.py --obs _obs_clean.jsonl --encoding structured --by-domain
python check_platform.py --obs _obs_clean.jsonl
```

Every row carries the run id, domain, item, condition, candidate type, encoding, served model, cost,
attempt count, the raw answer object, the recording platform, and a SHA-256 of the exact request
(`state_sha`). No figure in this note is aggregated in flight; all are recomputed from rows.

### Platform independence

Stimulus construction was verified identical across operating systems rather than assumed. Windows
and Linux produce the same stimulus digest (`acb978aa…`) over all 896 cells. Separately, a clone
checked out under Windows git defaults (`core.autocrlf=true`, `core.eol=crlf`) produces
byte-identical stimuli to an LF clone: all 280 tracked files match, and the digest is unchanged. The
repository's `.gitattributes` `eol=lf` is load-bearing for the recorded artifact hashes, but the
stimuli are immune regardless, because Python normalises line endings when parsing source and JSON
structural newlines never enter parsed values. Observation files are written LF-only on every
platform so that a run is byte-reproducible across operating systems.

### Fresh endpoint replication

```
python smoke.py --env-file <path to .env>            # 1 call, verifies the API contract
python determinism.py --items items_demo.jsonl --n 20 --repeats 5
python ccc_jev_run.py --repo <harness>/experiments --obs _obs_new.jsonl
python analyse_ccc.py --obs _obs_new.jsonl --by-domain
```

The full grid is 896 calls at roughly 1.6 pence. `smoke.py` checks every field this note's analysis
depends on and names the required code change if the API contract has moved; it was written before
first contact and caught the absent `confidence` field on the first live call.

### Note on the broken arm

The withdrawn structured-SQL rows are retained in `_obs_ccc.jsonl` (run `ff36f581d084`,
`encoding=structured`) and are **excluded** from `_obs_clean.jsonl`. `merge_obs.py` constructs the
clean file and refuses to average cells across code versions; reproducing the withdrawn result
requires selecting those rows deliberately.
