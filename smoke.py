#!/usr/bin/env python3
"""
First contact. One call, maximum information, no secrets in the output.

Everything in this toolkit so far has only ever spoken to the simulator. The
client was written from OpenRouter's published Jev docs, NOT from a live
response, so the API contract is the one genuine unknown. This isolates it:
a hardcoded trivial arithmetic item, no dependency on the harness repo, no
item bank, no analysis. If this passes, the contract is right and any later
failure is in the loaders. If it fails, it says exactly which field moved.

Costs roughly 0.002 pence. Prints the full raw response so nothing is taken
on trust, and never prints the API key.

    export OPENROUTER_API_KEY=...          # bash
    $env:OPENROUTER_API_KEY = '...'        # PowerShell
    python smoke.py

    python smoke.py --probe-endpoints      # try both documented paths
    python smoke.py --repeats 3            # cheap determinism peek
    python smoke.py --raw-out raw.json     # save the response verbatim

The output is safe to paste into a chat: it contains the response shape, the
probabilities, and the costs, but no credential and nothing from your repo.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

# Both paths the docs mention. The first is what jevlib uses.
ENDPOINTS = [
    "https://openrouter.ai/api/alpha/decisions",
    "https://openrouter.ai/api/v1/systemone",
]

MODEL = "typesafe/jev-1.13"
QID = "correct"

# Deliberately trivial and self-contained: a judge that cannot get this right
# has a problem unrelated to CCC, and nothing here touches the item bank.
STATE = {
    "question": "What is 17 + 4?",
    "candidate_answer": "21",
}
QUESTIONS = {
    QID: {
        "type": "noul",
        "instructions": ("You are grading a candidate answer to the question. "
                         "Decide whether the candidate answer is correct."),
        "criteria": {
            "true": "The candidate answer is correct.",
            "false": "The candidate answer is incorrect.",
        },
    }
}

OK = "  [ok]   "
BAD = "  [FAIL] "
WARN = "  [warn] "


def call(endpoint: str, key: str, timeout: float = 60.0):
    body = json.dumps({"model": MODEL, "state": STATE,
                       "questions": QUESTIONS}).encode()
    req = urllib.request.Request(
        endpoint, data=body, method="POST",
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
            return r.status, raw, (time.perf_counter() - t0) * 1000, None
    except urllib.error.HTTPError as e:
        try:
            raw = e.read().decode()
        except Exception:
            raw = ""
        return e.code, raw, (time.perf_counter() - t0) * 1000, None
    except Exception as e:
        return None, "", (time.perf_counter() - t0) * 1000, f"{type(e).__name__}: {e}"


def explain_status(code: int, raw: str) -> None:
    """Translate the HTTP code into what to actually do about it."""
    hint = {
        400: "Malformed request. The state/questions schema has changed -- "
             "compare the error body against jevlib.noul_question().",
        401: "Auth rejected. Check the key is an OpenRouter key and is current.",
        402: "Payment/credit required on the OpenRouter account.",
        403: "Forbidden. The account may not have access to Jev yet, or an "
             "egress policy is blocking openrouter.ai.",
        404: "Wrong path. Try --probe-endpoints; if the other path works, "
             "change ENDPOINT in jevlib.py to match.",
        422: "Schema rejected. The question type or criteria shape has moved.",
        429: "Rate limited. jevlib retries this with backoff; the smoke test "
             "does not.",
    }.get(code)
    if hint:
        print(f"{BAD}HTTP {code}: {hint}")
    elif code and code >= 500:
        print(f"{BAD}HTTP {code}: server side. jevlib retries these; retry later.")


def check_contract(payload: dict) -> list[str]:
    """Verify every assumption jevlib and the analysis scripts rely on."""
    problems = []

    answers = payload.get("answers")
    if not isinstance(answers, dict):
        problems.append("no 'answers' object -- jevlib.JevResult.answer() "
                        "reads payload['answers'][qid]")
        print(f"{BAD}answers        : missing")
        return problems
    print(f"{OK}answers        : present ({len(answers)} key(s): "
          f"{', '.join(answers)})")

    if QID not in answers:
        problems.append(f"the answer is not keyed by the question id "
                        f"{QID!r} -- got {list(answers)}. jevlib assumes the "
                        f"response echoes the request's question keys.")
        print(f"{BAD}answers['{QID}'] : missing")
        return problems
    a = answers[QID]
    print(f"{OK}answers['{QID}'] : present, keys = {', '.join(sorted(a))}")

    noul = a.get("noul")
    if noul is None:
        problems.append("no 'noul' field. This is the PRIMARY dependent "
                        "variable -- analyse_ccc.py reads row['noul']. If it "
                        "has been renamed, change the extraction in "
                        "jevlib.JevResult and ccc_jev_run.call().")
        print(f"{BAD}noul           : missing  <-- primary DV")
    elif not isinstance(noul, (int, float)):
        problems.append(f"noul is {type(noul).__name__}, expected a number")
        print(f"{BAD}noul           : {noul!r} (not numeric)")
    elif not 0.0 <= float(noul) <= 1.0:
        problems.append(f"noul = {noul}, outside [0,1]. The verdict rule "
                        f"(noul > 0.5) assumes a probability.")
        print(f"{BAD}noul           : {noul} (outside [0,1])")
    else:
        verdict = "correct" if float(noul) > 0.5 else "incorrect"
        print(f"{OK}noul           : {noul}  -> verdict '{verdict}'")
        if float(noul) <= 0.5:
            problems.append(f"noul = {noul} means Jev judged '17 + 4 = 21' "
                            f"INCORRECT. The answer is right, so either the "
                            f"criteria polarity is inverted (check that 'true' "
                            f"means correct) or the field is not a probability "
                            f"of correctness.")

    conf = a.get("confidence")
    if conf is None:
        print(f"{WARN}confidence     : missing (secondary only; not fatal)")
    else:
        print(f"{OK}confidence     : {conf}")

    usage = payload.get("usage") or {}
    cost = usage.get("cost")
    if cost is None:
        print(f"{WARN}usage.cost     : missing -- cost tracking will read 0")
    else:
        print(f"{OK}usage.cost     : ${cost}")
    if usage.get("input_tokens") is not None:
        print(f"{OK}input_tokens   : {usage['input_tokens']}")

    served = payload.get("model")
    if served:
        print(f"{OK}model served   : {served}")
        if MODEL.split("/")[-1].split("-")[0] not in served:
            problems.append(f"requested {MODEL} but got {served}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default=None,
                    help="override the endpoint (default: the documented "
                         "decisions path)")
    ap.add_argument("--probe-endpoints", action="store_true",
                    help="try every documented path and report which works")
    ap.add_argument("--repeats", type=int, default=1,
                    help="repeat the identical call; >1 gives a cheap "
                         "determinism peek")
    ap.add_argument("--raw-out", default=None,
                    help="write the raw response JSON to this file")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--env-file", default=None,
                    help="read OPENROUTER_API_KEY from a KEY=VALUE file "
                         "(.env or .env.txt) instead of the environment")
    args = ap.parse_args()

    key = os.environ.get("OPENROUTER_API_KEY")

    if not key and args.env_file:
        # Minimal .env reader: KEY=VALUE per line, optional `export `, optional
        # surrounding quotes, # comments ignored. Deliberately does not shell out,
        # so nothing in the file can execute.
        try:
            with open(args.env_file, encoding="utf-8-sig") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    name, _, val = line.partition("=")
                    name = name.replace("export ", "").strip()
                    if name == "OPENROUTER_API_KEY":
                        key = val.strip().strip('"').strip("'")
                        print(f"  key source : {args.env_file}")
                        break
        except OSError as e:
            print(f"cannot read {args.env_file}: {e}", file=sys.stderr)
            return 2
        if not key:
            print(f"{args.env_file} does not define OPENROUTER_API_KEY",
                  file=sys.stderr)
            return 2

    if not key:
        print("OPENROUTER_API_KEY is not set.\n", file=sys.stderr)
        print("  bash       : export OPENROUTER_API_KEY=sk-or-...",
              file=sys.stderr)
        print("  PowerShell : $env:OPENROUTER_API_KEY = 'sk-or-...'",
              file=sys.stderr)
        print("  or         : python smoke.py --env-file path\\to\\.env.txt",
              file=sys.stderr)
        return 2

    print("=" * 70)
    print("JEV SMOKE TEST -- first contact")
    print("=" * 70)
    print(f"  model    : {MODEL}")
    print(f"  key      : present, {len(key)} chars (not printed)")
    print(f"  item     : {STATE['question']} -> {STATE['candidate_answer']} "
          f"(correct)")
    print(f"  expecting: noul > 0.5")

    targets = ENDPOINTS if args.probe_endpoints else [args.endpoint or ENDPOINTS[0]]

    payload = None
    endpoint_used = None
    for ep in targets:
        print(f"\n--- POST {ep}")
        code, raw, ms, err = call(ep, key, args.timeout)
        if err:
            print(f"{BAD}transport: {err}")
            continue
        print(f"  HTTP {code}  {ms:.0f}ms  {len(raw)} bytes")
        if code != 200:
            explain_status(code, raw)
            if raw:
                print("  response body:")
                print("    " + raw[:600].replace("\n", "\n    "))
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"{BAD}response is not JSON: {e}")
            print("    " + raw[:400])
            continue
        endpoint_used = ep
        break

    if payload is None:
        print("\n" + "-" * 70)
        print("RESULT: no usable response. Nothing above is a code bug in the")
        print("toolkit until a 200 comes back -- fix access first.")
        return 1

    print(f"\n=== RAW RESPONSE (verbatim) ===")
    print(json.dumps(payload, indent=2, sort_keys=True))

    if args.raw_out:
        with open(args.raw_out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        print(f"\n  saved to {args.raw_out}")

    print(f"\n=== CONTRACT CHECK ===")
    problems = check_contract(payload)

    # ---- cheap determinism peek ---------------------------------------- #
    if args.repeats > 1:
        print(f"\n=== DETERMINISM PEEK ({args.repeats} identical calls) ===")
        vals = []
        first = (payload.get("answers", {}).get(QID, {}) or {}).get("noul")
        if first is not None:
            vals.append(float(first))
        for i in range(args.repeats - 1):
            code, raw, ms, err = call(endpoint_used, key, args.timeout)
            if err or code != 200:
                print(f"  call {i + 2}: failed ({err or f'HTTP {code}'})")
                continue
            try:
                v = json.loads(raw)["answers"][QID]["noul"]
            except (KeyError, json.JSONDecodeError, TypeError):
                print(f"  call {i + 2}: unexpected shape")
                continue
            vals.append(float(v))
        print(f"  noul values : {vals}")
        if len(vals) > 1:
            spread = max(vals) - min(vals)
            print(f"  spread      : {spread:.6f}")
            if spread == 0:
                print("  -> identical on this item. Suggestive of determinism, "
                      "but ONE item is not a determinism check.")
            else:
                print("  -> varies. Run determinism.py properly before sizing "
                      "anything.")
        print("\n  Either way, run: python determinism.py --items "
              "items_demo.jsonl --n 20 --repeats 5")

    print("\n" + "-" * 70)
    if problems:
        print(f"RESULT: {len(problems)} contract problem(s). Fix before the run:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("RESULT: the API contract matches what the toolkit assumes.")
    print(f"Endpoint in use: {endpoint_used}")
    print("\nNext:")
    print("  python determinism.py --items items_demo.jsonl --n 20 --repeats 5")
    print("  python ccc_jev_run.py --repo <...>/experiments --domain sql "
          "--encoding verbatim --obs _obs_ccc.jsonl")
    print("  python analyse_ccc.py --obs _obs_ccc.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
