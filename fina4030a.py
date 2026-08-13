"""
fina4030a.py — shared model client for FINA4030A, Augmented Intelligence in Finance.

Every lab notebook fetches this file at run time. It exists so that the course has
exactly ONE place where a provider is named. If the University confirms institutional
access mid-term, change PROVIDER here and every notebook follows on next run.

It also keeps an automatic transcript of every model call, which is what the
Reproducibility and Verification Appendix required by the course outline is built
from. You are not asked to keep that record by hand.

Standard library only — nothing to install, works in Colab and in the browser
fallback runtime.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

__version__ = "0.3"

# ---------------------------------------------------------------------------
# Configuration.  Change PROVIDER here, nowhere else.
# ---------------------------------------------------------------------------

PROVIDER = "github"          # "github" | "cuhk_portal" | "cached" | "echo"
MODEL    = "openai/gpt-4o-mini"

_PROVIDERS = {
    # GitHub Models. Free with any GitHub account; token needs the models:read
    # permission. Two base URLs have existed; we probe the current one first and
    # fall back to the legacy Azure host.
    "github": {
        "urls": [
            "https://models.github.ai/inference/chat/completions",
            "https://models.inference.ai.azure.com/chat/completions",
        ],
        "auth": "bearer",
        "env": "GITHUB_TOKEN",
    },
    # CUHK API Portal (Azure API Management in front of Azure Foundry).
    # Fill base_url and the subscription key from the portal before use.
    "cuhk_portal": {
        "urls": [""],
        "auth": "apim",
        "env": "CUHK_APIM_KEY",
    },
}

_state = {
    "provider": PROVIDER,
    "model": MODEL,
    "token": None,
    "base_url": None,
    "cache": None,       # list of recorded/supplied responses for offline use
    "cache_pos": 0,
    "verified": False,
}


class ModelError(RuntimeError):
    """Raised when a model call fails in a way the student should see plainly."""


# ---------------------------------------------------------------------------
# Transcript — the audit trail
# ---------------------------------------------------------------------------

@dataclass
class Call:
    n: int
    when: str
    provider: str
    model: str
    temperature: float | None
    seed: int | None
    system: str | None
    prompt: str
    response: str
    seconds: float
    usage: dict = field(default_factory=dict)
    error: str | None = None


TRANSCRIPT: list[Call] = []


def reset_transcript() -> None:
    TRANSCRIPT.clear()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def configure(provider: str | None = None,
              model: str | None = None,
              token: str | None = None,
              base_url: str | None = None,
              cache_path: str | None = None) -> None:
    """Set the provider for this session. Called once at the top of a lab."""
    if provider:
        _state["provider"] = provider
    if model:
        _state["model"] = model
    if base_url:
        _state["base_url"] = base_url
    if token:
        _state["token"] = token
    if cache_path:
        with open(cache_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        _state["cache"] = payload["responses"] if isinstance(payload, dict) else payload
        _state["cache_pos"] = 0
    _state["verified"] = False


def _get_token() -> str:
    """Find a token without ever printing it."""
    if _state["token"]:
        return _state["token"]

    spec = _PROVIDERS.get(_state["provider"])
    if not spec:
        return ""
    env_name = spec["env"]

    if os.environ.get(env_name):
        _state["token"] = os.environ[env_name]
        return _state["token"]

    # Colab secrets panel (the key icon in the left sidebar)
    try:
        from google.colab import userdata  # type: ignore
        val = userdata.get(env_name)
        if val:
            _state["token"] = val
            return val
    except Exception:
        pass

    # Last resort: masked prompt. Never echoed, never written to the notebook.
    try:
        import getpass
        val = getpass.getpass(f"Paste your {env_name} (input is hidden): ").strip()
        _state["token"] = val
        return val
    except Exception:
        raise ModelError(
            f"No {env_name} found. Add it to the Colab secrets panel (key icon, "
            f"left sidebar) with the name {env_name}, then run this cell again."
        )


def _post(url: str, body: dict, headers: dict, timeout: int = 90) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# The one function every lab uses
# ---------------------------------------------------------------------------

def complete(prompt: str,
             system: str | None = None,
             temperature: float | None = 0.0,
             seed: int | None = None,
             max_tokens: int = 900,
             retries: int = 3) -> str:
    """Send one prompt, return the text. Records the call in TRANSCRIPT."""
    provider = _state["provider"]
    t0 = time.time()

    if provider == "echo":
        text = f"[echo] {prompt[:200]}"
        _record(prompt, system, temperature, seed, text, time.time() - t0, {})
        return text

    if provider == "cached":
        cache = _state["cache"] or []
        if not cache:
            raise ModelError("Provider is 'cached' but no cache file was loaded. "
                             "Pass cache_path= to configure().")
        text = cache[_state["cache_pos"] % len(cache)]
        _state["cache_pos"] += 1
        _record(prompt, system, temperature, seed, text, time.time() - t0,
                {"cached": True})
        return text

    spec = _PROVIDERS.get(provider)
    if spec is None:
        raise ModelError(f"Unknown provider {provider!r}.")

    token = _get_token()
    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
    body = {"model": _state["model"], "messages": messages,
            "max_tokens": max_tokens}
    if temperature is not None:
        body["temperature"] = temperature
    if seed is not None:
        body["seed"] = seed

    if spec["auth"] == "bearer":
        headers = {"Authorization": f"Bearer {token}",
                   "Content-Type": "application/json",
                   "Accept": "application/vnd.github+json"}
    else:  # Azure API Management
        headers = {"Ocp-Apim-Subscription-Key": token,
                   "Content-Type": "application/json"}

    urls = [_state["base_url"]] if _state["base_url"] else list(spec["urls"])
    urls = [u for u in urls if u]
    if not urls:
        raise ModelError(f"No base URL configured for provider {provider!r}. "
                         "Pass base_url= to configure().")

    last = None
    for attempt in range(retries):
        for url in urls:
            try:
                out = _post(url, body, headers)
                text = out["choices"][0]["message"]["content"]
                _state["base_url"] = url          # remember what worked
                secs = time.time() - t0
                _record(prompt, system, temperature, seed, text, secs,
                        out.get("usage", {}))
                return text
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:300]
                last = f"HTTP {e.code}: {detail}"
                if e.code == 401:
                    raise ModelError(
                        "Rejected: 401 Unauthorized. The token is missing, wrong, or "
                        "lacks the models:read permission. Regenerate it on GitHub "
                        "under Settings > Developer settings > Personal access tokens."
                    ) from None
                if e.code == 429:
                    wait = 5 * (attempt + 1)
                    print(f"  rate limited, waiting {wait}s "
                          f"(attempt {attempt + 1} of {retries})")
                    time.sleep(wait)
                    break                            # retry outer loop
                if e.code == 404:
                    continue                         # try the other base URL
            except urllib.error.URLError as e:
                last = f"network: {e.reason}"
                continue
            except (KeyError, IndexError, json.JSONDecodeError) as e:
                last = f"unexpected response shape: {type(e).__name__}"
                continue

    secs = time.time() - t0
    _record(prompt, system, temperature, seed, "", secs, {}, error=last)
    raise ModelError(
        f"Model call failed after {retries} attempts. Last error — {last}\n"
        "If this is a rate limit, wait a minute. If it is a network error, check "
        "your VPN. If it persists, switch to the cached responses supplied with "
        "the lab so you can still complete it: fina4030a.configure("
        "provider='cached', cache_path='...')"
    )


def _record(prompt, system, temperature, seed, response, secs, usage, error=None):
    TRANSCRIPT.append(Call(
        n=len(TRANSCRIPT) + 1, when=_now(), provider=_state["provider"],
        model=_state["model"], temperature=temperature, seed=seed,
        system=system, prompt=prompt, response=response,
        seconds=round(secs, 2), usage=usage or {}, error=error,
    ))


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def verify(quiet: bool = False) -> bool:
    """Check the configured provider actually answers. Safe to call anytime."""
    lines = ["Model client check", "-" * 46,
             f"  client version   {__version__}",
             f"  provider         {_state['provider']}",
             f"  model            {_state['model']}"]
    ok = True
    if _state["provider"] in ("echo", "cached"):
        n = len(_state["cache"] or [])
        lines.append(f"  cached responses {n}")
        ok = _state["provider"] == "echo" or n > 0
        lines.append("  [ OK ]  offline mode ready" if ok
                     else "  [FAIL]  no cache loaded")
    else:
        try:
            t0 = time.time()
            out = complete("Reply with exactly the word: ready",
                           temperature=0.0, max_tokens=5)
            lines.append(f"  endpoint         {_state['base_url']}")
            lines.append(f"  [ OK ]  answered in {time.time() - t0:.1f}s "
                         f"— {out.strip()[:30]!r}")
        except ModelError as e:
            ok = False
            lines.append(f"  [FAIL]  {str(e).splitlines()[0]}")
    _state["verified"] = ok
    if not quiet:
        print("\n".join(lines))
    return ok


# ---------------------------------------------------------------------------
# Reproducibility and Verification Appendix
# ---------------------------------------------------------------------------

def appendix(student: str = "", verification: str = "",
             residual_risk: str = "", reproducibility: str = "") -> str:
    """Render the appendix the course outline requires, from the transcript."""
    if not TRANSCRIPT:
        return "No model calls were made in this notebook."

    errs = sum(1 for c in TRANSCRIPT if c.error)
    total_s = sum(c.seconds for c in TRANSCRIPT)
    toks = sum(c.usage.get("total_tokens", 0) for c in TRANSCRIPT)
    temps = sorted({c.temperature for c in TRANSCRIPT})

    md = [
        "## Reproducibility and Verification Appendix",
        "",
        f"**Student.** {student or '_not stated_'}",
        "",
        "**1. Configuration.**",
        "",
        f"- Provider: `{_state['provider']}`",
        f"- Model: `{_state['model']}`",
        f"- Endpoint: `{_state['base_url'] or 'n/a'}`",
        f"- Client: `fina4030a.py` v{__version__}",
        f"- Temperature(s) used: {', '.join(str(t) for t in temps)}",
        "",
        "**2. The chain.**",
        "",
        f"{len(TRANSCRIPT)} model call(s), {total_s:.1f}s total"
        + (f", {toks} tokens" if toks else "")
        + (f", {errs} error(s)" if errs else "") + ".",
        "",
        "| # | when (UTC) | temp | seed | secs | chars out | error |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in TRANSCRIPT:
        md.append(f"| {c.n} | {c.when} | {c.temperature} | {c.seed} | "
                  f"{c.seconds} | {len(c.response)} | {c.error or '—'} |")

    md += [
        "",
        "Full prompts and responses are in the accompanying `transcript.json`.",
        "",
        "**3. Verification performed.**",
        "",
        verification or "_to be completed by the student_",
        "",
        "**4. Residual risk.**",
        "",
        residual_risk or "_to be completed by the student_",
        "",
        "**5. Reproducibility.**",
        "",
        reproducibility or "_to be completed by the student_",
    ]
    return "\n".join(md)


def save_transcript(path: str = "transcript.json") -> str:
    payload = {
        "course": "FINA4030A",
        "client_version": __version__,
        "provider": _state["provider"],
        "model": _state["model"],
        "endpoint": _state["base_url"],
        "written": _now(),
        "calls": [asdict(c) for c in TRANSCRIPT],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)
    return path


# ---------------------------------------------------------------------------
# Small helpers the labs share
# ---------------------------------------------------------------------------

def extract_answer(text: str, tag: str = "ANSWER") -> float | None:
    """Pull a number from a line like 'ANSWER: 1234.5'. None if absent/unparseable.

    A missing or malformed tag is itself a finding — the model was asked for a
    format and did not comply. Do not silently drop those cases.
    """
    m = re.search(rf"{tag}\s*[:=]\s*\$?\s*(-?[\d,]+(?:\.\d+)?)", text, re.I)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None
