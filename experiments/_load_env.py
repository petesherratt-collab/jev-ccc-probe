"""Minimal, dependency-free .env loader for the experiment adapters.

Loads KEY=VALUE lines from a .env file into os.environ so a run can pick up
OPENROUTER_API_KEY (and similar) from a file instead of an exported shell
variable. Deliberately tiny: no python-dotenv dependency, no interpolation.

Resolution order for the .env path (first that exists wins):
  1. explicit `path` argument;
  2. the `OPENROUTER_ENV_FILE` environment variable (use this to point at a
     .env that lives outside the repo, e.g. a Downloads folder);
  3. `<repo-root>/.env`;
  4. `./.env` (current working directory).

Safety:
  - never overrides a variable already set in the environment (an explicit
    export or CI secret always wins over the file);
  - never prints or logs values — returns only the list of KEY NAMES loaded,
    so callers can log "loaded OPENROUTER_API_KEY from .env" without leaking it;
  - ignores blank lines and `#` comments; strips optional surrounding quotes.
"""

from __future__ import annotations

import os


def _candidate_paths(path: str | None):
    if path:
        yield path
    env_file = os.environ.get("OPENROUTER_ENV_FILE")
    if env_file:
        yield env_file
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yield os.path.join(repo_root, ".env")
    yield os.path.join(os.getcwd(), ".env")


def load_env(path: str | None = None, *, override: bool = False) -> list[str]:
    """Load KEY=VALUE pairs from the first existing .env candidate into os.environ.

    Returns the list of key names that were set (never the values). If no .env
    is found, returns [] silently — callers fall back to the ambient env var.
    """
    target = next((p for p in _candidate_paths(path) if p and os.path.isfile(p)), None)
    if target is None:
        return []
    loaded: list[str] = []
    with open(target, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key.lower().startswith("export "):
                key = key[len("export "):].strip()
            value = value.strip().strip('"').strip("'")
            if not key:
                continue
            if override or key not in os.environ:
                os.environ[key] = value
                loaded.append(key)
    return loaded
