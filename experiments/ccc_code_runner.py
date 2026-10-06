"""Deterministic, sandboxed test runner for the CCC code-domain replication.

This is the AUTHORITATIVE grader used at run time to assign the mechanical gold
label to a candidate implementation. It is a frozen artifact: its SHA-256 is
recorded in `PREREG_ccc_codedomain.md`, and the gold labels every experiment
uses are produced only through `grade_sandboxed`.

Isolation / determinism guarantees (see the preregistration for the contract):
  - **Fixed interpreter:** requires CPython `PYTHON_REQUIRED` (3.11); aborts otherwise.
  - **Fresh subprocess per candidate**, launched in isolated mode (`-I -S`), so no
    ambient environment, user site-packages, or parent state leaks in.
  - **Deterministic hashing:** child runs with `PYTHONHASHSEED=0`.
  - **CPU + memory + process + fd limits** applied via `resource.setrlimit` in the
    child *before* the candidate code is exec'd. These are **POSIX-only**; on native
    Windows (no `resource` module, no `preexec_fn`) they are skipped and the wall-clock
    timeout is the runtime bound. The network/write blocks, fixed hash seed, and
    fail-closed grading are cross-platform, so gold labels are identical on every OS.
    (Run on POSIX for the full limit set; the frozen items never approach the limits,
    so the labels do not depend on them.)
  - **Wall-clock timeout** enforced by the parent (SIGKILL backstop) in case a soft
    CPU limit is evaded (e.g. sleeping).
  - **No network:** `socket` is neutralised in the child before the candidate runs.
  - **No candidate filesystem writes:** RLIMIT_FSIZE is set to 0 in the child.
  - **Fail-closed:** a crash, timeout, limit kill, or unparseable child result counts
    as a FAILED test — never as gold-correct.

A candidate is gold-correct iff it passes every frozen unit test; gold-wrong iff it
fails at least one. No model is ever consulted for the gold.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

# --- Frozen sandbox parameters (part of the preregistered contract) -------------
# CPython (major, minor) versions whose gold labels were verified byte-identical
# (gold signature 634302aa... across 3.10-3.13). Frozen accepted range for a run:
ACCEPTED_PYTHON = ((3, 11), (3, 12), (3, 13))
CPU_SECONDS = 2                # RLIMIT_CPU soft (hard = soft + 1) in the child
WALL_TIMEOUT_S = 5             # parent-side wall-clock backstop (SIGKILL)
ADDRESS_SPACE_BYTES = 512 * 1024 * 1024   # RLIMIT_AS (512 MiB)
MAX_OPEN_FILES = 64            # RLIMIT_NOFILE
MAX_PROCESSES = 64             # RLIMIT_NPROC (per-uid; modest, blocks fork bombs)

# The child harness. Run via `python3 -I -S -c <this>`; reads {"source","tests"}
# as JSON on stdin, applies limits, neutralises the network, then grades.
_CHILD = r'''
import sys, json, socket, builtins
try:
    import resource            # POSIX only; absent on native Windows
except Exception:
    resource = None
# 1) read the job BEFORE tightening limits (json import must succeed)
job = json.loads(sys.stdin.read())
source, tests = job["source"], job["tests"]
# 2) neutralise the network
def _no_net(*a, **k):
    raise RuntimeError("network access is disabled in the sandbox")
socket.socket = _no_net
socket.create_connection = _no_net
socket.create_server = _no_net
# 2b) block filesystem writes (write-mode open / os.open create/write flags)
def _no_write(*a, **k):
    raise RuntimeError("filesystem writes are disabled in the sandbox")
_real_open = builtins.open
def _guard_open(file, mode="r", *a, **k):
    if any(c in str(mode) for c in ("w", "a", "x", "+")):
        _no_write()
    return _real_open(file, mode, *a, **k)
builtins.open = _guard_open
import os as _os
_real_osopen = _os.open
def _guard_osopen(path, flags, *a, **k):
    if flags & (_os.O_WRONLY | _os.O_RDWR | _os.O_CREAT | _os.O_APPEND):
        _no_write()
    return _real_osopen(path, flags, *a, **k)
_os.open = _guard_osopen
# 3) resource limits (soft, hard) -- POSIX only; on Windows the wall-clock
#    timeout in the parent is the runtime bound, and net/write blocks + the
#    fixed hash seed still apply, so grading stays deterministic and contained.
if resource is not None:
    CPU = {cpu}
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (CPU, CPU + 1))
    except (ValueError, OSError):
        pass
    AS = {addr}
    try:
        resource.setrlimit(resource.RLIMIT_AS, (AS, AS))
    except (ValueError, OSError):
        pass
    try:
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))   # no candidate file writes
    except (ValueError, OSError):
        pass
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, ({nofile}, {nofile}))
    except (ValueError, OSError):
        pass
# 4) exec the candidate, then run each frozen test
ns = {{}}
passed = 0
detail = []
try:
    exec(source, ns)
    solve = ns["solve"]
except Exception as e:                       # compile/def error: all tests fail
    print(json.dumps({{"passed": 0, "total": len(tests),
                       "error": "load:" + repr(e)}}))
    sys.exit(0)
for args, expected in tests:
    try:
        got = solve(*args)
        ok = (got == expected)
    except Exception as e:                    # runtime crash: this test fails
        ok = False
    passed += 1 if ok else 0
    detail.append(bool(ok))
print(json.dumps({{"passed": passed, "total": len(tests), "detail": detail}}))
'''.format(cpu=CPU_SECONDS, addr=ADDRESS_SPACE_BYTES, nofile=MAX_OPEN_FILES)


def _require_python() -> None:
    if sys.version_info[:2] not in ACCEPTED_PYTHON:
        accepted = ", ".join(f"{a}.{b}" for a, b in ACCEPTED_PYTHON)
        raise RuntimeError(
            f"CCC runner requires CPython in {{{accepted}}}; found "
            f"{sys.version_info[0]}.{sys.version_info[1]}. Gold was verified only on "
            "those interpreters.")


def _preexec():  # pragma: no cover - runs only in the forked child before exec
    import resource as _r
    try:
        _r.setrlimit(_r.RLIMIT_NPROC, (MAX_PROCESSES, MAX_PROCESSES))
    except (ValueError, OSError):
        pass


def grade_sandboxed(source: str, tests, wall_timeout_s: int = WALL_TIMEOUT_S):
    """Return (passed, total, info) for a candidate `solve` source against `tests`.

    Runs in a fresh isolated subprocess with fixed hash seed, CPU/memory/process/fd
    limits, no network, and no filesystem writes. Any crash, timeout, or limit kill
    is fail-closed: it lowers `passed`, never raising it. `tests` is a sequence of
    (args_tuple, expected)."""
    _require_python()
    payload = json.dumps({
        "source": source,
        "tests": [[list(args), expected] for args, expected in tests],
    })
    env = {"PYTHONHASHSEED": "0", "PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    # preexec_fn (RLIMIT_NPROC) is POSIX-only; omit it off POSIX.
    preexec = _preexec if os.name == "posix" else None
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-S", "-c", _CHILD],
            input=payload, capture_output=True, text=True,
            timeout=wall_timeout_s, env=env, preexec_fn=preexec, check=False,
        )
    except subprocess.TimeoutExpired:
        return 0, len(tests), {"error": "wall_timeout"}
    out = (proc.stdout or "").strip().splitlines()
    if not out:
        return 0, len(tests), {"error": "no_output", "stderr": (proc.stderr or "")[-200:]}
    try:
        res = json.loads(out[-1])
    except json.JSONDecodeError:
        return 0, len(tests), {"error": "bad_output", "raw": out[-1][:200]}
    return int(res.get("passed", 0)), int(res.get("total", len(tests))), res


def gold_label(item, wall_timeout_s: int = WALL_TIMEOUT_S) -> dict:
    """Compute the frozen mechanical gold for one item: run its reference and buggy
    variants through the sandbox and return pass/fail summaries."""
    rp, rt, _ = grade_sandboxed(item["reference"], item["tests"], wall_timeout_s)
    bp, bt, _ = grade_sandboxed(item["buggy"], item["tests"], wall_timeout_s)
    return {
        "name": item["name"],
        "reference_correct": rp == rt,   # must be True
        "buggy_is_decoy": bp < bt,       # must be True
        "reference_pass": (rp, rt),
        "buggy_pass": (bp, bt),
    }


def self_verify(repeats: int = 2) -> None:
    """Assert the frozen invariant through the SANDBOX (not the in-process grader),
    and check determinism by grading each item `repeats` times and requiring identical
    results. Raises AssertionError on any violation."""
    from ccc_code_items import ITEMS
    _require_python()
    first = None
    for r in range(repeats):
        snapshot = []
        for it in ITEMS:
            g = gold_label(it)
            assert g["reference_correct"], f"{it['name']}: reference not gold-correct {g['reference_pass']}"
            assert g["buggy_is_decoy"], f"{it['name']}: buggy variant is not a decoy {g['buggy_pass']}"
            snapshot.append((g["name"], g["reference_pass"], g["buggy_pass"]))
        if first is None:
            first = snapshot
        else:
            assert snapshot == first, "non-deterministic grading across repeats"
    print(f"sandbox self_verify OK: {len(ITEMS)} items, {repeats} deterministic repeats; "
          f"references gold-correct, buggy variants are decoys")


def sha256_of(path: str) -> str:
    import hashlib
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    self_verify()
    print("ccc_code_items.py  sha256:", sha256_of(os.path.join(here, "ccc_code_items.py")))
    print("ccc_code_runner.py sha256:", sha256_of(os.path.join(here, "ccc_code_runner.py")))
    print("interpreter:", sys.version.split()[0])
