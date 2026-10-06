#!/usr/bin/env python3
"""
Stimulus gate for the attribution decomposition. BLOCKING: the runner refuses to
make a call unless this exits 0.

It asserts the five invariants of PREREG_jev_attribution.md §1 against the real
repository stimuli. Every one of them is a bug this study has already had, or a
close relative:

  1. Body invariance        -- the arith bug: a factor that silently did not vary.
  2. One factor per contrast-- the solver_rationale bundle: three changes at once.
  3. No collapse            -- the exit-4 preflight, extended to 18 cells.
  4. Reused constants       -- re-typing COMMON_INTRO would silently fork the prose.
  5. Builder agreement      -- the arith bug again: never assume what a builder emits.
                               Restatement-absent cells must equal the harness's own
                               build_injection_block byte-for-byte.

    python test_attribution_stimuli.py --repo /path/to/The-Generated-Trace-Leak-Harness
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n          {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


def digest(obj) -> str:
    return hashlib.blake2b(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode(),
        digest_size=8).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--domain", action="append", default=None)
    args = ap.parse_args()

    import attribution_stimuli as A
    A.add_repo_to_path(args.repo)
    import provenance_injection_harness as H

    domains = args.domain or ["sql", "code", "arith"]
    print("attribution stimulus gate\n")
    print(f"  cells: {len(A.CELLS)}   per domain: {A.EXPECTED_CELLS}")

    # ---- 4. constants come from the harness, not re-typed ------------------
    src = open("attribution_stimuli.py", encoding="utf-8").read()
    check("4. COMMON_INTRO not re-typed as a literal",
          "additional analysis of the problem" not in src,
          "the intro string appears literally in this module; import it instead")
    check("4. RELIABILITY_CLAIMS not re-typed",
          "may contain errors" not in src)
    check("4. identity label strings not re-typed",
          "### Solver Analysis" not in src)

    all_rows: dict[str, list[dict]] = {}
    for dom in domains:
        rows = list(A.LOADERS[dom]()())
        all_rows[dom] = rows
        exp = A.EXPECTED_CELLS[dom] * 2 * len(A.CANDIDATES)
        check(f"[{dom}] row count = cells x encodings x candidates x items",
              len(rows) % exp == 0 and len(rows) > 0,
              f"got {len(rows)}, not a multiple of {exp}")

        # ---- 1. body invariance --------------------------------------------
        bodies = defaultdict(set)
        for r in rows:
            bodies[r["item_id"]].add(r["body"])
        bad = {k: len(v) for k, v in bodies.items() if len(v) != 1}
        check(f"[{dom}] 1. body byte-identical across all cells of an item",
              not bad, f"items with >1 body: {list(bad)[:4]}")

        # the body must actually appear in every injected block, unchanged
        missing = [(r["item_id"], r["cell"]) for r in rows
                   if r["cell"] != "no_injection" and r["body"] not in r["notes"]]
        check(f"[{dom}] 1. body present verbatim in every injected block",
              not missing, f"{len(missing)} blocks altered the body, e.g. {missing[:3]}")

        # ---- 2. one factor per primary contrast ----------------------------
        # restatement effect: id:neutral vs id:neutral+restate
        # identity effect:    id:neutral vs id:solver
        by = {}
        for r in rows:
            by[(r["item_id"], r["cell"], r["encoding"], r["candidate_type"])] = r
        item0 = rows[0]["item_id"]

        def blk(cell, enc="verbatim"):
            """The prose actually sent, for the given encoding.

            This must come from different row fields per encoding: `block` carries
            the label line (verbatim), `notes` does not (structured, where the label
            is a typed field). The first version of this helper returned `notes`
            unconditionally, which made the verbatim identity assertions compare a
            string with itself -- one vacuous PASS and one spurious FAIL.
            """
            r = by[(item0, cell, enc, "correct")]
            return r["block"] if enc == "verbatim" else r["notes"]

        H_intro = H.COMMON_INTRO
        n_plain, n_rest = blk("id:neutral", "structured"), \
            blk("id:neutral+restate", "structured")
        check(f"[{dom}] 2. restatement contrast differs only by one inserted line",
              n_rest.endswith(n_plain.split(H_intro, 1)[1])
              and len(n_rest) > len(n_plain),
              "restatement cell is not the plain cell plus one line")

        # identity contrast: in STRUCTURED the prose must be identical (label is
        # a field), in VERBATIM it must differ only in the label line.
        s_neu = blk("id:neutral", "structured")
        s_sol = blk("id:solver", "structured")
        check(f"[{dom}] 2. identity contrast leaves structured prose identical",
              s_neu == s_sol,
              "structured prose differs between neutral and solver; the label "
              "has leaked into analysis_notes")
        v_neu = blk("id:neutral", "verbatim")
        v_sol = blk("id:solver", "verbatim")
        check(f"[{dom}] 2. identity contrast differs in verbatim prose",
              v_neu != v_sol)
        check(f"[{dom}] 2. verbatim identity difference is the label line only",
              v_neu.split("\n", 1)[1] == v_sol.split("\n", 1)[1],
              "something besides the first line differs")

        # attribution field present only in structured
        f_struct = A.build_state(by[(item0, "id:solver", "structured", "correct")], dom)
        f_verb = A.build_state(by[(item0, "id:solver", "verbatim", "correct")], dom)
        check(f"[{dom}] 2. analysis_attribution set in structured only",
              "analysis_attribution" in f_struct and "analysis_attribution" not in f_verb)
        check(f"[{dom}] 2. structured baseline carries no attribution field",
              "analysis_attribution" not in
              A.build_state(by[(item0, "no_injection", "structured", "correct")], dom))

        # ---- 3. no two cells collapse to the same request ------------------
        seen = defaultdict(dict)
        for r in rows:
            st = A.build_state(r, dom)
            st = {k: v for k, v in st.items() if not k.startswith("_")}
            seen[(r["item_id"], r["encoding"], r["candidate_type"])][r["cell"]] = digest(st)
        collapsed = []
        for key, bycell in seen.items():
            if len(set(bycell.values())) < len(bycell):
                inv = defaultdict(list)
                for c, d in bycell.items():
                    inv[d].append(c)
                collapsed.append((key, [v for v in inv.values() if len(v) > 1]))
        check(f"[{dom}] 3. no two cells produce the same request",
              not collapsed,
              f"{len(collapsed)} collapsed groups, e.g. {collapsed[:2]}")

        # ---- 5. agreement with the harness's own builder -------------------
        # Restatement-absent, label-included composition must EQUAL
        # build_injection_block. Checked on arith, where Item objects exist.
        if dom == "arith":
            import run_provenance_injection as rp
            from provenance_injection_harness import Condition, build_injection_block
            items = rp.build_items(None, rp.CONFIRMATORY_RAW, rp.CONFIRMATORY_CACHE)
            it = items[0]
            mism = []
            for cell in A.cells_for("arith"):
                if cell.identity is None or cell.restate:
                    continue
                ours = A.compose(cell, it.injected["full_wrong_rationale"],
                                 "UNUSED", include_label=True)
                theirs = build_injection_block(
                    Condition(cell.identity, "full_wrong_rationale", cell.claim), it)
                if ours != theirs:
                    mism.append(cell.name)
            check("[arith] 5. composition == harness build_injection_block",
                  not mism, f"mismatched cells: {mism}")

        # the legacy bridge must reproduce the main study's string exactly
        if "legacy:solver_bundle" in {c.name for c in A.cells_for(dom)}:
            mod = {"sql": "run_ccc_sql", "code": "run_ccc_codedomain"}[dom]
            a = __import__(mod)
            stim = a.build_stimuli()
            want = stim[item0]["text"]["solver/full_wrong_rationale"]
            got = blk("legacy:solver_bundle", "verbatim")
            check(f"[{dom}] 5. legacy bridge reproduces the frozen string",
                  got == want, f"got {got[:70]!r}\n          want {want[:70]!r}")
            check(f"[{dom}] 5. legacy bridge differs from its composed analogue",
                  got != blk("comp:solver+verified+restate", "verbatim"),
                  "bridge and composed cell are identical; contrast 5 is vacuous")

    # ---- cross-domain: stimulus digest, for the run record -----------------
    for dom, rows in all_rows.items():
        d = digest(sorted(
            (r["item_id"], r["cell"], r["encoding"], r["candidate_type"],
             digest({k: v for k, v in A.build_state(r, dom).items()
                     if not k.startswith("_")}))
            for r in rows))
        print(f"  stimulus digest [{dom}]: {d}")

    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED -- no call may be made: {', '.join(FAILS)}")
        return 1
    print("all stimulus invariants hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
