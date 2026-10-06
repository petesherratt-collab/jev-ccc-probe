# Jev structured-decision judge probe

Does **contextual conclusion capture** survive a change of judge *architecture*?

CCC established that an LLM judge's ability to tell a correct answer from an incorrect
one degrades when a conflicting conclusion is present in its context, that a bare
neutrally-labelled wrong answer is **sufficient** to induce it, and that an
authoritative source label is **not necessary**. All of that evidence came from judges
that read a prompt and emit text.

This repository probes `typesafe/jev-1.13` — a "System One" structured-decision model
that takes a typed `state` object and returns a calibrated probability rather than
prose, and which its vendor describes as unable to hallucinate by construction.

**Headline results.** Capture crosses the architectural boundary intact and, in the
relational domain, arrives stronger. The failure *mode* depends jointly on domain and
on how the stimulus is encoded. And a preregistered decomposition found that a typed
provenance field is **inert** while the same label in prose is **protective** — so
serialising provenance into application state removes a defence rather than adding an
attack surface. A judge told an analysis *may contain errors* defers less; told it *has
been verified*, it collapses to chance.

The write-up is `jev_structured_judge_probe.md`. Two of its earlier findings were
withdrawn in the course of the work — one caught by a baseline gate, one by a
preregistered contrast that was permitted to come out either way — and both are
reported in §9 rather than removed.

---

## Why this is a separate repository

This is a different research line from the Generated-Trace Leak Harness, and the
programme's rule is to keep lines analytically distinct so each gets its own framing.
The probe nonetheless needs the harness's frozen CCC item banks and prompt builders to
construct its stimuli.

Those modules are **vendored byte-identical into `experiments/` and hash-pinned** in
`VENDOR_PINS.json`, rather than submoduled. That follows the harness's own practice: it
vendors and pins its substrate block (`_SUBSTRATE_PIN`, byte-identical v0.6 through
v1.5.1) for exactly this reason. A pin makes this repository self-contained and its
stimuli reproducible without the other repo existing; a submodule makes it a moving
reference.

The pin is a **self-consistency detector, not an authenticity proof** — the same caveat
the harness applies to its own file pin. It catches a vendored module being edited and
the stimuli silently changing. It does not prove the vendored copy matches upstream,
which would need an external anchor.

Vendored from `petesherratt-collab/The-Generated-Trace-Leak-Harness`, branch
`claude/amazing-faraday-gvs9fy`, commit `9f7437ae`.

---

## Requirements

Python 3.10+. The runners and the stimulus gate are **stdlib only** — they need no
third-party packages and no network beyond the API call itself. The *analysis* scripts
(`analyse_ccc.py`, `analyse_attribution.py`, `calibration.py`) need `numpy` and
`scipy`:

```bash
pip install numpy scipy
```

That split is deliberate: evidence can be produced on a machine with nothing installed,
and analysed somewhere else.

## Quick start

```bash
python verify_vendor.py                              # pins must match first
python test_calibration.py                           # 46 self-tests, no API calls
python test_attribution_stimuli.py --repo .          # 38 stimulus invariants
```

All reported statistics recompute from the committed observation files, with no API
calls and no credentials:

```bash
python analyse_ccc.py        --obs _obs_clean.jsonl --by-domain
python calibration.py        --obs _obs_clean.jsonl --both-encodings --by-domain
python analyse_attribution.py --obs _obs_attr.jsonl --repo . --secondary
python calibration.py        --obs _obs_attr.jsonl --condition-field cell --both-encodings
```

To make live calls you need `OPENROUTER_API_KEY` and about 35p:

```bash
python smoke.py --env-file path/to/.env          # one call, checks the endpoint contract
python determinism.py --env-file path/to/.env    # is the judge deterministic?
python ccc_jev_run.py --repo . --env-file path/to/.env --obs _obs_new.jsonl
python attribution_run.py --repo . --env-file path/to/.env --repeats 3 --obs _obs_attr_new.jsonl
```

`--repo .` works because the vendored modules sit in `experiments/`, which is where the
tooling's path helper looks.

Windows users: `Run-JevCcc.ps1` drives the main grid end to end and pins UTF-8.

---

## Layout

| path | what it is |
|---|---|
| `jev_structured_judge_probe.md` | the write-up |
| `PREREG_jev_attribution.md` | preregistration for the attribution decomposition, frozen before any stimulus was composed |
| `jevlib.py` | client, LF-pinned observation writer, name-derived RNG streams, offline simulator |
| `smoke.py`, `determinism.py` | endpoint contract check; determinism characterisation |
| `ccc_jev_run.py`, `analyse_ccc.py` | the main 896-cell grid and its two dependent variables |
| `calibration.py`, `test_calibration.py` | calibration audit — Brier, Murphy decomposition, AUC, flip margin, tie-respecting ECE |
| `attribution_stimuli.py`, `test_attribution_stimuli.py` | the composed stimulus set and its 38-check blocking gate |
| `attribution_run.py`, `analyse_attribution.py` | the preregistered decomposition and its five contrasts |
| `merge_obs.py`, `check_platform.py` | evidence assembly that refuses to mix code versions; cross-platform stimulus digests |
| `experiments/` | vendored harness modules, hash-pinned |
| `experiments/results/` | the **frozen** arith stimulus caches. Without these, `run_provenance_injection` regenerates the injected texts from a model — needing a key and yielding different stimuli |
| `VENDOR_PINS.json`, `verify_vendor.py` | the pins and their checker, covering modules and frozen data alike |
| `_obs_*.jsonl`, `determinism_obs.jsonl` | raw streamed observations; every reported number recomputes from these |

`analyse.py`, `pilot.py` and `make_demo_items.py` are the standalone demo path, usable
without the harness modules.

---

## Evidence

| | |
|---|---|
| Served model | `typesafe/jev-1.13-20260917` |
| Main grid | 896 cells, 1,344 calls, $0.037, zero failures, 100% completeness |
| Attribution run | `3bee5c5c030e` — 11,904 calls, k=3, $0.3019, zero failures, 100% completeness |
| Main stimulus digest | `acb978aa…` over all 896 per-cell request hashes, identical on Windows and Linux |
| Attribution digests | sql `88afaaf1eab58491`, code `804d2292377f6ed4`, arith `27812d88f54de59d` |
| Determinism | verdicts 100% stable over 5 repeats; within-item sd 0.0154 |
| Seed | `305774821` (the harness's authoritative frontier v3 seed) |

Total measurement cost across both experiments: about 34p.

---

## Guards

Each was added after a real defect and each is exercised against deliberately broken
input, because this study has twice reported something it had to withdraw:

- the runner refuses to start (exit 4) if any domain-by-encoding cell collapses its
  conditions to identical requests — the check that caught a bug which would have
  reported a flat null as "resists capture";
- it refuses to start (exit 3) if a domain fails to load, rather than writing a partial
  evidence file that looks complete;
- it refuses to start (exit 5) if the stimulus gate does not pass;
- the stimulus gate asserts body invariance, one-factor-per-contrast, no cell collapse,
  harness-constant reuse, and byte agreement with the harness's own builder;
- the analysis refuses to print a share-of-baseline ratio when baseline discrimination
  is below 0.10;
- `merge_obs.py` refuses to average cells originating from different code versions;
- observation files are LF-pinned, so the same run produces the same bytes on Windows
  and Linux;
- every calibration verdict is gated on a bootstrap interval rather than a point
  estimate.

## Licence

MIT, matching the harness repository.
