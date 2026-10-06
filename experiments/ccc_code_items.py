"""Frozen code items for the Contextual Conclusion Capture code-domain replication.

Each item is a small, self-contained function-implementation task with:
  - `spec`      : natural-language specification shown to the judge;
  - `signature` : the function signature the candidate must implement;
  - `reference` : a correct implementation (passes every unit test  -> gold-correct);
  - `buggy`     : a plausible single-bug implementation (fails >= 1 test -> gold-wrong);
  - `bug_desc`  : one-line description of the injected fault;
  - `tests`     : the unit-test gold as (args_tuple, expected_output) pairs.

The gold label of a candidate is MECHANICAL: run the candidate source against
`tests`; "correct" == passes all, "wrong" == fails >= 1. No model is asked to
decide the gold. This module is FROZEN: its SHA-256 is recorded in the
preregistration (`PREREG_ccc_codedomain.md`) and in every run's metadata.

Run `python3 ccc_code_items.py` to (a) self-verify the correct/buggy invariants
and (b) print the file's SHA-256.
"""

from __future__ import annotations

ITEMS = [
    {
        "name": "run_length_encode",
        "spec": "Return the run-length encoding of a string: each maximal run of a "
                "character c of length n becomes c followed by the decimal n. "
                "Example: 'aaabb' -> 'a3b2'. The empty string maps to ''.",
        "signature": "def solve(s: str) -> str:",
        "reference": (
            "def solve(s):\n"
            "    if not s:\n"
            "        return ''\n"
            "    out = []\n"
            "    prev = s[0]; c = 1\n"
            "    for ch in s[1:]:\n"
            "        if ch == prev:\n"
            "            c += 1\n"
            "        else:\n"
            "            out.append(prev + str(c)); prev = ch; c = 1\n"
            "    out.append(prev + str(c))\n"
            "    return ''.join(out)\n"
        ),
        "buggy": (
            "def solve(s):\n"
            "    if not s:\n"
            "        return ''\n"
            "    out = []\n"
            "    prev = s[0]; c = 1\n"
            "    for ch in s[1:]:\n"
            "        if ch == prev:\n"
            "            c += 1\n"
            "        else:\n"
            "            out.append(prev + str(c)); prev = ch; c = 1\n"
            "    return ''.join(out)\n"  # BUG: never emits the final run
        ),
        "bug_desc": "drops the final run (missing terminal append).",
        "tests": [(("aaabb",), "a3b2"), (("abc",), "a1b1c1"), (("",), ""), (("zzzz",), "z4")],
    },
    {
        "name": "is_balanced",
        "spec": "Return True iff the parentheses in the string are balanced and "
                "properly nested (only the characters '(' and ')' appear). "
                "'(())' -> True, ')(' -> False, '(()' -> False.",
        "signature": "def solve(s: str) -> bool:",
        "reference": (
            "def solve(s):\n"
            "    bal = 0\n"
            "    for ch in s:\n"
            "        if ch == '(':\n"
            "            bal += 1\n"
            "        elif ch == ')':\n"
            "            bal -= 1\n"
            "            if bal < 0:\n"
            "                return False\n"
            "    return bal == 0\n"
        ),
        "buggy": (
            "def solve(s):\n"
            "    return s.count('(') == s.count(')')\n"  # BUG: ignores ordering
        ),
        "bug_desc": "checks only equal counts, ignoring nesting order.",
        "tests": [(("(())",), True), ((")(",), False), (("(()",), False), (("()()",), True)],
    },
    {
        "name": "roman_to_int",
        "spec": "Convert an uppercase Roman numeral to its integer value, honouring "
                "subtractive pairs (IV=4, IX=9, XL=40, ...). 'MCMXCIV' -> 1994.",
        "signature": "def solve(s: str) -> int:",
        "reference": (
            "def solve(s):\n"
            "    vals = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}\n"
            "    total = 0\n"
            "    for i, ch in enumerate(s):\n"
            "        v = vals[ch]\n"
            "        if i + 1 < len(s) and vals[s[i+1]] > v:\n"
            "            total -= v\n"
            "        else:\n"
            "            total += v\n"
            "    return total\n"
        ),
        "buggy": (
            "def solve(s):\n"
            "    vals = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}\n"
            "    return sum(vals[ch] for ch in s)\n"  # BUG: no subtractive rule
        ),
        "bug_desc": "adds every symbol, ignoring the subtractive rule.",
        "tests": [(("III",), 3), (("IV",), 4), (("IX",), 9), (("LVIII",), 58), (("MCMXCIV",), 1994)],
    },
    {
        "name": "missing_number",
        "spec": "Given a list containing all-but-one of the integers 0..n exactly "
                "once (length n), return the single missing integer. [0,1,3] -> 2.",
        "signature": "def solve(nums: list) -> int:",
        "reference": (
            "def solve(nums):\n"
            "    n = len(nums)\n"
            "    return n * (n + 1) // 2 - sum(nums)\n"
        ),
        "buggy": (
            "def solve(nums):\n"
            "    n = len(nums)\n"
            "    return n * (n - 1) // 2 - sum(nums)\n"  # BUG: wrong triangular number
        ),
        "bug_desc": "uses n(n-1)/2 instead of n(n+1)/2.",
        "tests": [(([0, 1, 3],), 2), (([3, 0, 1],), 2), (([0],), 1), (([1],), 0)],
    },
    {
        "name": "caesar_decode",
        "spec": "Decode a Caesar cipher with left shift 3 over lowercase letters "
                "(wrapping a->x), leaving non-letters unchanged. 'abc' -> 'xyz'.",
        "signature": "def solve(s: str) -> str:",
        "reference": (
            "def solve(s):\n"
            "    return ''.join(\n"
            "        chr((ord(c) - 97 - 3) % 26 + 97) if c.isalpha() else c\n"
            "        for c in s)\n"
        ),
        "buggy": (
            "def solve(s):\n"
            "    return ''.join(\n"
            "        chr(ord(c) - 3) if c.isalpha() else c\n"  # BUG: no wraparound
            "        for c in s)\n"
        ),
        "bug_desc": "subtracts without modular wraparound.",
        "tests": [(("dro",), "aol"), (("abc",), "xyz"), (("d e",), "a b"), (("aaa",), "xxx")],
    },
    {
        "name": "flatten_sum",
        "spec": "Return the sum of every integer in an arbitrarily nested list of "
                "integers. [1,[2,[3,4]]] -> 10.",
        "signature": "def solve(x: list) -> int:",
        "reference": (
            "def solve(x):\n"
            "    if isinstance(x, list):\n"
            "        return sum(solve(e) for e in x)\n"
            "    return x\n"
        ),
        "buggy": (
            "def solve(x):\n"
            "    total = 0\n"
            "    for e in x:\n"
            "        if isinstance(e, list):\n"
            "            for f in e:\n"
            "                if not isinstance(f, list):\n"
            "                    total += f\n"  # BUG: ignores lists deeper than one level
            "        else:\n"
            "            total += e\n"
            "    return total\n"
        ),
        "bug_desc": "only descends one level; deeper integers are dropped.",
        "tests": [(([1, [2, 3], 4],), 10), (([1, [2, [3, 4]]],), 10), (([[1], [2], [3]],), 6)],
    },
    {
        "name": "count_words",
        "spec": "Return the number of whitespace-separated words, treating any run of "
                "whitespace as a single separator. '  a   b ' -> 2. '' -> 0.",
        "signature": "def solve(s: str) -> int:",
        "reference": (
            "def solve(s):\n"
            "    return len(s.split())\n"
        ),
        "buggy": (
            "def solve(s):\n"
            "    return len(s.split(' '))\n"  # BUG: empty tokens from repeated spaces
        ),
        "bug_desc": "splits on single spaces, counting empty tokens.",
        "tests": [(("a b c",), 3), (("  a   b ",), 2), (("",), 0), (("hello",), 1)],
    },
    {
        "name": "gcd",
        "spec": "Return the greatest common divisor of two positive integers. "
                "gcd(48,36) -> 12.",
        "signature": "def solve(a: int, b: int) -> int:",
        "reference": (
            "def solve(a, b):\n"
            "    while b:\n"
            "        a, b = b, a % b\n"
            "    return a\n"
        ),
        "buggy": (
            "def solve(a, b):\n"
            "    return b if a % b == 0 else a % b\n"  # BUG: single Euclid step only
        ),
        "bug_desc": "performs one modulo step instead of iterating.",
        "tests": [(((12, 8)), 4), (((48, 36)), 12), (((17, 5)), 1), (((100, 10)), 10)],
    },
    {
        "name": "is_palindrome_alnum",
        "spec": "Return True iff the string is a palindrome when considering only "
                "alphanumeric characters, case-insensitively. 'Racecar' -> True.",
        "signature": "def solve(s: str) -> bool:",
        "reference": (
            "def solve(s):\n"
            "    t = [c.lower() for c in s if c.isalnum()]\n"
            "    return t == t[::-1]\n"
        ),
        "buggy": (
            "def solve(s):\n"
            "    t = [c for c in s if c.isalnum()]\n"  # BUG: case-sensitive
            "    return t == t[::-1]\n"
        ),
        "bug_desc": "does not fold case before comparing.",
        "tests": [(("Racecar",), True), (("ab",), False), (("12321",), True), (("Aa",), True)],
    },
    {
        "name": "rpn_eval",
        "spec": "Evaluate a reverse-Polish expression given as a token list over the "
                "operators +, -, * and integer literals. ['5','2','-'] -> 3.",
        "signature": "def solve(tokens: list) -> int:",
        "reference": (
            "def solve(tokens):\n"
            "    st = []\n"
            "    for t in tokens:\n"
            "        if t in ('+', '-', '*'):\n"
            "            b = st.pop(); a = st.pop()\n"
            "            st.append(a + b if t == '+' else a - b if t == '-' else a * b)\n"
            "        else:\n"
            "            st.append(int(t))\n"
            "    return st[0]\n"
        ),
        "buggy": (
            "def solve(tokens):\n"
            "    st = []\n"
            "    for t in tokens:\n"
            "        if t in ('+', '-', '*'):\n"
            "            b = st.pop(); a = st.pop()\n"
            "            st.append(a + b if t == '+' else b - a if t == '-' else a * b)\n"  # BUG: subtraction order swapped
            "        else:\n"
            "            st.append(int(t))\n"
            "    return st[0]\n"
        ),
        "bug_desc": "computes b - a for subtraction (operand order swapped).",
        "tests": [((["3", "4", "+"],), 7), ((["5", "2", "-"],), 3), ((["2", "3", "4", "*", "+"],), 14)],
    },
    {
        "name": "title_case",
        "spec": "Capitalise the first letter of each whitespace-separated word and "
                "lowercase the rest. 'hello WORLD' -> 'Hello World'.",
        "signature": "def solve(s: str) -> str:",
        "reference": (
            "def solve(s):\n"
            "    return ' '.join(w[:1].upper() + w[1:].lower() for w in s.split())\n"
        ),
        "buggy": (
            "def solve(s):\n"
            "    return ' '.join(w[:1].upper() + w[1:] for w in s.split())\n"  # BUG: rest not lowercased
        ),
        "bug_desc": "capitalises the first letter but leaves the rest untouched.",
        "tests": [(("hello WORLD",), "Hello World"), (("aBc",), "Abc"), (("foo",), "Foo")],
    },
    {
        "name": "binary_to_int",
        "spec": "Interpret a non-empty string of '0'/'1' as an unsigned binary number "
                "(most significant bit first) and return its integer value. '110' -> 6.",
        "signature": "def solve(s: str) -> int:",
        "reference": (
            "def solve(s):\n"
            "    return int(s, 2)\n"
        ),
        "buggy": (
            "def solve(s):\n"
            "    return sum(int(b) * (2 ** i) for i, b in enumerate(s))\n"  # BUG: LSB-first weighting
        ),
        "bug_desc": "weights the first character as the least-significant bit.",
        "tests": [(("101",), 5), (("110",), 6), (("1",), 1), (("0",), 0), (("1000",), 8)],
    },
    {
        "name": "second_largest",
        "spec": "Return the second-largest DISTINCT value in a list of integers "
                "(the list has at least two distinct values). [5,5,4] -> 4.",
        "signature": "def solve(nums: list) -> int:",
        "reference": (
            "def solve(nums):\n"
            "    s = sorted(set(nums))\n"
            "    return s[-2]\n"
        ),
        "buggy": (
            "def solve(nums):\n"
            "    return sorted(nums)[-2]\n"  # BUG: does not deduplicate the maximum
        ),
        "bug_desc": "returns the second element without removing duplicate maxima.",
        "tests": [(([3, 1, 2],), 2), (([5, 5, 4],), 4), (([1, 2, 2, 3],), 2)],
    },
    {
        "name": "char_frequency_max",
        "spec": "Return the character that occurs most often; break ties by returning "
                "the lexicographically smallest tied character. 'bbaa' -> 'a'.",
        "signature": "def solve(s: str) -> str:",
        "reference": (
            "def solve(s):\n"
            "    from collections import Counter\n"
            "    c = Counter(s)\n"
            "    m = max(c.values())\n"
            "    return min(ch for ch, n in c.items() if n == m)\n"
        ),
        "buggy": (
            "def solve(s):\n"
            "    from collections import Counter\n"
            "    return Counter(s).most_common(1)[0][0]\n"  # BUG: tie-break follows insertion order
        ),
        "bug_desc": "ignores the lexicographic tie-break rule.",
        "tests": [(("aabb",), "a"), (("bbaa",), "a"), (("abc",), "a")],
    },
    {
        "name": "digital_root",
        "spec": "Repeatedly sum the decimal digits of a non-negative integer until a "
                "single digit remains, and return it. 99 -> 9, 942 -> 6.",
        "signature": "def solve(n: int) -> int:",
        "reference": (
            "def solve(n):\n"
            "    while n >= 10:\n"
            "        n = sum(int(d) for d in str(n))\n"
            "    return n\n"
        ),
        "buggy": (
            "def solve(n):\n"
            "    return sum(int(d) for d in str(n))\n"  # BUG: single digit-sum pass
        ),
        "bug_desc": "sums digits once instead of iterating to a single digit.",
        "tests": [(((16,)), 7), (((99,)), 9), (((5,)), 5), (((942,)), 6)],
    },
    {
        "name": "remove_adjacent_dups",
        "spec": "Collapse each run of consecutive equal characters to a single "
                "character, preserving order. 'aabcca' -> 'abca'.",
        "signature": "def solve(s: str) -> str:",
        "reference": (
            "def solve(s):\n"
            "    out = []\n"
            "    for ch in s:\n"
            "        if not out or out[-1] != ch:\n"
            "            out.append(ch)\n"
            "    return ''.join(out)\n"
        ),
        "buggy": (
            "def solve(s):\n"
            "    out = []\n"
            "    for ch in s:\n"
            "        if ch not in out:\n"  # BUG: removes non-adjacent duplicates too
            "            out.append(ch)\n"
            "    return ''.join(out)\n"
        ),
        "bug_desc": "deduplicates globally rather than only adjacent runs.",
        "tests": [(("aabcca",), "abca"), (("abba",), "aba"), (("aaa",), "a")],
    },
]


def _run(source: str, args) -> object:
    """Execute a candidate `solve` source on positional `args`; return output or
    an ('__error__', repr) sentinel so a crash counts as a failed test."""
    ns: dict = {}
    try:
        exec(source, ns)
        return ns["solve"](*args)
    except Exception as e:  # a crash is a failed test, never gold-correct
        return ("__error__", repr(e))


def grade(source: str, tests) -> tuple[int, int]:
    """Return (passed, total) for a candidate source against the frozen tests."""
    passed = 0
    for args, expected in tests:
        if _run(source, args) == expected:
            passed += 1
    return passed, len(tests)


def self_verify() -> None:
    """Assert the frozen invariant: every reference passes all tests; every buggy
    variant fails at least one. Raises AssertionError on any violation."""
    names = [it["name"] for it in ITEMS]
    assert len(names) == len(set(names)), "duplicate item name"
    for it in ITEMS:
        rp, rt = grade(it["reference"], it["tests"])
        bp, bt = grade(it["buggy"], it["tests"])
        assert rp == rt, f"{it['name']}: reference failed {rt - rp}/{rt} tests"
        assert bp < bt, f"{it['name']}: buggy variant passed all tests (not a decoy)"
    print(f"self_verify OK: {len(ITEMS)} items; references pass all, buggy variants fail >= 1")


def items_sha256() -> str:
    """SHA-256 of this source file (the frozen item definitions)."""
    import hashlib
    with open(__file__, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


if __name__ == "__main__":
    self_verify()
    print("ccc_code_items.py sha256:", items_sha256())
