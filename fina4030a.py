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

__version__ = "1.1"


# ---------------------------------------------------------------------------
# Configuration.  Change PROVIDER here, nowhere else.
# ---------------------------------------------------------------------------

PROVIDER = "cuhk_portal"     # "cuhk_portal" | "groq" | "openrouter" | "cached" | "echo"
MODEL    = "gpt-5.4-mini"

# GitHub Models was retired on 30 July 2026. Kept here only so that an old
# notebook produces an explanation rather than a mystifying 404 or 410.
_RETIRED = {
    "github": ("GitHub Models was fully retired on 30 July 2026. Switch to "
               "provider='groq' (free, no card) or provider='openrouter'. "
               "See the setup cell at the top of the lab."),
}

_PROVIDERS = {
    # Groq. Free tier, no card. 30 requests/min and 1,000/day on the 70B model
    # — roughly a hundred runs of a ten-call lab per day.
    "groq": {
        "urls": ["https://api.groq.com/openai/v1/chat/completions"],
        "catalog": "https://api.groq.com/openai/v1/models",
        "auth": "bearer",
        "env": "GROQ_API_KEY",
        "signup": "https://console.groq.com  (sign in, then API Keys)",
        "min_interval": 2.0,
    },
    # OpenRouter. Routes to many providers; the ':free' models cost nothing.
    # Lower daily cap than Groq, but reachable almost everywhere.
    "openrouter": {
        "urls": ["https://openrouter.ai/api/v1/chat/completions"],
        "catalog": "https://openrouter.ai/api/v1/models",
        "auth": "bearer",
        "env": "OPENROUTER_API_KEY",
        "signup": "https://openrouter.ai/keys",
        "min_interval": 3.0,
    },
    # CUHK API Portal — Azure API Management in front of Azure AI Foundry.
    # Endpoint confirmed by CUHK eLearning, August 2026. This is the OpenAI v1
    # surface, so no api-version query parameter is required.
    #   model        gpt-5.4-mini      tool calling  yes     vision  yes
    #   max input    32,000 tokens     max output    4,096 tokens
    # Starter subscriptions are rate limited on tokens as well as on calls, so
    # keep max_tokens modest and pause between calls in a classroom.
    "cuhk_portal": {
        "urls": ["https://cuhk-apip.azure-api.net/foundry-eus2/openai/v1/chat/completions"],
        "catalog": "https://cuhk-apip.azure-api.net/foundry-eus2/openai/v1/models",
        "auth": "apim",
        "env": "CUHK_APIM_KEY",
        "signup": "https://cuhk-apip.developer.azure-api.net",
        # Starter plan: 5 calls per minute, 100 per week, weekly renewal.
        # 12.5s between calls keeps us inside the per-minute limit without
        # relying on any lab remembering to pause.
        "min_interval": 12.5,
        "weekly_quota": 100,
        # Confirmed live 13 Aug 2026: gpt-5.4-mini accepts temperature but wants
        # max_completion_tokens, not max_tokens. Starting on the right shape
        # avoids one wasted 400 per session — which counts against the quota.
        "param_style": 1,
    },
}

_last_call_at = [0.0]      # module-level so the throttle survives cell re-runs

_state = {
    "param_style": None,
    "param_label": None,   # which request shape this model accepts
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
    notes: str = ""            # e.g. temperature not accepted by this model


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
    spec = _PROVIDERS.get(_state["provider"], {})
    if _state["param_style"] is None:
        _state["param_style"] = spec.get("param_style")
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
             max_tokens: int = 600,
             retries: int = 3,
             history: list | None = None) -> str:
    """Send one prompt, return the text. Records the call in TRANSCRIPT.

    history — optional list of prior turns, each {"role": "user"|"assistant",
    "content": "..."}, sent ahead of `prompt` so the model sees the earlier
    exchange. Use it when the question only makes sense as a follow-up.
    Omit it and every call is independent, which is the default and is what
    Labs 1 and 2 rely on.
    """
    provider = _state["provider"]
    if history:
        if not isinstance(history, list) or not all(
                isinstance(m, dict) and m.get("role") in ("user", "assistant")
                and isinstance(m.get("content"), str) for m in history):
            raise ModelError(
                "history must be a list of {'role': 'user' or 'assistant', "
                "'content': '...'} dictionaries, oldest turn first. You passed "
                f"{type(history).__name__}.")
    if provider in _RETIRED:
        raise ModelError(_RETIRED[provider])
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
               list(history or []) + \
               [{"role": "user", "content": prompt}]

    # Model families disagree about request shape. Newer reasoning-style models
    # reject max_tokens (wanting max_completion_tokens) and reject any
    # temperature other than the default. Rather than hardcode a guess, try the
    # shapes in order and remember which one this deployment accepts.
    def _variants():
        base = {"model": _state["model"], "messages": messages}
        t = {} if temperature is None else {"temperature": temperature}
        sd = {} if seed is None else {"seed": seed}
        return [
            ("max_tokens + temperature",         {**base, **t, **sd, "max_tokens": max_tokens}),
            ("max_completion_tokens + temperature",
                                                 {**base, **t, **sd, "max_completion_tokens": max_tokens}),
            ("max_completion_tokens, no temperature",
                                                 {**base, **sd, "max_completion_tokens": max_tokens}),
            ("no length limit, no temperature",  {**base}),
        ]

    variants = _variants()
    order = list(range(len(variants)))
    if _state["param_style"] is not None:          # start with what worked before
        i = _state["param_style"]
        order = [i] + [j for j in order if j != i]

    if spec["auth"] == "bearer":
        headers = {"Authorization": f"Bearer {token}",
                   "Content-Type": "application/json",
                   "Accept": "application/json"}
    else:  # Azure API Management in front of Foundry
        headers = {"api-key": token,
                   "Ocp-Apim-Subscription-Key": token,
                   "Content-Type": "application/json"}

    urls = [_state["base_url"]] if _state["base_url"] else list(spec["urls"])
    urls = [u for u in urls if u]
    if not urls:
        raise ModelError(f"No base URL configured for provider {provider!r}. "
                         "Pass base_url= to configure().")

    gap = spec.get("min_interval", 0)
    if gap:
        wait = gap - (time.time() - _last_call_at[0])
        if wait > 0:
            time.sleep(wait)
    _last_call_at[0] = time.time()

    last, tried = None, []
    for attempt in range(retries):
        for url in urls:
            for vi in order:
                label, body = variants[vi]
                try:
                    out = _post(url, body, headers)
                    text = out["choices"][0]["message"]["content"]
                    _state["base_url"] = url
                    _state["param_style"] = vi
                    _state["param_label"] = label
                    note = "" if vi == 0 else f"request adapted: {label}"
                    if history:
                        note = (f"follow-up call carrying {len(history)} prior turn(s)"
                                + ("; " + note if note else ""))
                    if "temperature" not in body and temperature is not None:
                        note += ("; temperature was NOT accepted by this model, so "
                                 "the request could not ask for determinism")
                    _record(prompt, system,
                            body.get("temperature"), seed, text,
                            time.time() - t0, out.get("usage", {}), notes=note)
                    return text
                except urllib.error.HTTPError as e:
                    detail = e.read().decode("utf-8", "replace")[:500]
                    last = f"HTTP {e.code}: {detail}"
                    if e.code == 400:
                        continue            # wrong request shape — try the next
                    if e.code == 401:
                        raise ModelError(
                            f"401 Unauthorized. The {spec['env']} key is missing, "
                            f"wrong, or not yet approved. Get or check it at "
                            f"{spec.get('signup','')}") from None
                    if e.code == 429:
                        wait = 15 * (attempt + 1)
                        print(f"  rate limited, waiting {wait}s "
                              f"(attempt {attempt + 1} of {retries})")
                        time.sleep(wait)
                        break
                    if e.code in (403, 410):
                        raise ModelError(f"HTTP {e.code} — refused or withdrawn. "
                                         f"{detail}") from None
                    if e.code == 404:
                        tried.append(url)
                        break
                except urllib.error.URLError as e:
                    last = f"network: {e.reason}"
                    break
                except (KeyError, IndexError, json.JSONDecodeError) as e:
                    last = f"unexpected response shape: {type(e).__name__}"
                    break

    secs = time.time() - t0
    _record(prompt, system, temperature, seed, "", secs, {}, error=last)
    hint = ""
    if tried:
        hint = ("\n\nEvery candidate endpoint returned 404:\n  "
                + "\n  ".join(dict.fromkeys(tried))
                + "\n\nA 404 here usually means the model id is wrong rather than the "
                  "URL. Run  fina4030a.list_models(show=20)  to see what your token "
                  "can reach, then fina4030a.configure(model='...').")
    raise ModelError(
        f"Model call failed after {retries} attempts. Last error — {last}{hint}\n"
        "If this is a rate limit, wait a minute. If it is a network error, check "
        "your VPN. If it persists, switch to the cached responses supplied with "
        "the lab so you can still complete it: fina4030a.configure("
        "provider='cached', cache_path='...')"
    )


def _record(prompt, system, temperature, seed, response, secs, usage,
            error=None, notes=""):
    TRANSCRIPT.append(Call(
        n=len(TRANSCRIPT) + 1, when=_now(), provider=_state["provider"],
        model=_state["model"], temperature=temperature, seed=seed,
        system=system, prompt=prompt, response=response,
        seconds=round(secs, 2), usage=usage or {}, error=error, notes=notes,
    ))


def _get_json(url: str, headers: dict, timeout: int = 30):
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def list_models(show: int = 0) -> list:
    """Return the model ids this token can actually reach. Use when a call 404s."""
    spec = _PROVIDERS.get(_state["provider"])
    if not spec or not spec.get("catalog"):
        raise ModelError(f"No catalog endpoint for provider {_state['provider']!r}.")
    tok = _get_token()
    if spec["auth"] == "apim":
        headers = {"api-key": tok, "Ocp-Apim-Subscription-Key": tok,
                   "Accept": "application/json"}
    else:
        headers = {"Authorization": f"Bearer {tok}", "Accept": "application/json"}
    try:
        data = _get_json(spec["catalog"], headers)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:200]
        if e.code == 401:
            raise ModelError(f"401 from the catalog. The {spec['env']} key is missing "
                             f"or wrong. Get one at {spec.get('signup','')}") from None
        if e.code in (403, 410):
            raise ModelError(f"HTTP {e.code} — this service is refusing or has been "
                             f"withdrawn. {body}") from None
        raise ModelError(f"Catalog request failed: HTTP {e.code} {body}") from None
    if isinstance(data, dict):
        items = data.get("data") or data.get("models") or []
    else:
        items = data
    ids = sorted({(m.get("id") or m.get("name") or "") for m in items
                  if isinstance(m, dict)} - {""})
    if show:
        print(f"{len(ids)} model(s) available to this token. Showing "
              f"{min(show, len(ids))}:")
        for i in ids[:show]:
            print("   ", i)
    return ids




def probe(verbose: bool = True) -> dict:
    """Try every provider for which a key is already available. Report what works.

    Does not prompt for keys — only uses ones already in the environment or in
    Colab's secrets panel, so it is safe to run without being asked anything.
    """
    import urllib.error as _ue
    results = {}
    saved = (_state["provider"], _state["token"], _state["base_url"])
    for name, spec in _PROVIDERS.items():
        key = os.environ.get(spec["env"])
        if not key:
            try:
                from google.colab import userdata  # type: ignore
                key = userdata.get(spec["env"])
            except Exception:
                key = None
        if not key:
            results[name] = "no key"
            continue
        if not [u for u in spec["urls"] if u]:
            results[name] = "no base URL configured"
            continue
        _state.update(provider=name, token=key, base_url=None)
        try:
            complete("Reply with the word ok", temperature=0.0, max_tokens=5,
                     retries=1)
            results[name] = "WORKS"
        except Exception as e:
            results[name] = str(e).splitlines()[0][:80]
    _state["provider"], _state["token"], _state["base_url"] = saved
    if verbose:
        print("Provider probe")
        print("-" * 52)
        for k, v in results.items():
            mark = "[ OK ]" if v == "WORKS" else "[    ]" if v == "no key" else "[FAIL]"
            print(f"  {mark}  {k:<14} {v}")
        working = [k for k, v in results.items() if v == "WORKS"]
        print()
        print(f"  Use: fina4030a.configure(provider='{working[0]}')" if working
              else "  No provider is currently usable. See the setup cell.")
    return results


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def verify(quiet: bool = False) -> bool:
    """Check the configured provider answers, and say precisely what failed."""
    L = ["Model client check", "-" * 52,
         f"  client version   {__version__}",
         f"  provider         {_state['provider']}",
         f"  model            {_state['model']}"]
    ok = True

    if _state["provider"] in ("echo", "cached"):
        n = len(_state["cache"] or [])
        L.append(f"  cached responses {n}")
        ok = _state["provider"] == "echo" or n > 0
        L.append("  [ OK ]  offline mode ready" if ok else "  [FAIL]  no cache loaded")
        _state["verified"] = ok
        if not quiet:
            print("\n".join(L))
        return ok

    spec = _PROVIDERS.get(_state["provider"], {})
    L.append("")

    # step 1 — catalog. Isolates token and host from model id and path.
    ids = []
    if spec.get("catalog"):
        L.append("  step 1  can the token reach the service?")
        try:
            ids = list_models()
            L.append(f"          [ OK ]  {len(ids)} model(s) visible")
        except ModelError as e:
            ok = False
            body = " ".join(str(e).split())          # collapse newlines
            L.append(f"          [FAIL]  {body[:400]}")
            L.append("")
            L.append("  Stop here. Nothing downstream can work until this does.")
            L.append(f"  Colab secret must be named {spec.get('env','?')} with")
            L.append("  notebook access toggled on. Get a key at:")
            L.append(f"    {spec.get('signup', '')}")
            L.append("  Or run  fina4030a.probe()  to see which providers do work.")
            _state["verified"] = False
            if not quiet:
                print("\n".join(L))
            return False

        # step 2 — is the configured model actually one of them?
        L.append(f"  step 2  is {_state['model']!r} in the catalog?")
        if _state["model"] in ids:
            L.append("          [ OK ]")
        else:
            ok = False
            suffix = _state["model"].split("/")[-1].lower()
            near = [i for i in ids if suffix in i.lower()][:5]
            L.append("          [FAIL]  not available to this token")
            if near:
                L.append("          closest matches:")
                L += [f"            {i}" for i in near]
            else:
                L.append("          a few that are available:")
                L += [f"            {i}" for i in ids[:8]]
            L.append("")
            L.append("          Fix with: fina4030a.configure(model='<one of the above>')")

    # step 3 — an actual completion
    if ok:
        L.append("  step 3  does a real call succeed?")
        try:
            t0 = time.time()
            out = complete("Reply with exactly the word: ready",
                           temperature=0.0, max_tokens=5)
            L.append(f"          [ OK ]  {time.time() - t0:.1f}s via {_state['base_url']}")
            L.append(f"          replied {out.strip()[:30]!r}")
            lbl = _state.get("param_label")
            if lbl:
                L.append(f"          request shape accepted: {lbl}")
                if "no temperature" in lbl:
                    L.append("          NOTE  this model refuses the temperature "
                             "parameter, so you")
                    L.append("                cannot ask it for deterministic output "
                             "at all.")
        except ModelError as e:
            ok = False
            body = " ".join(str(e).split())          # collapse newlines
            L.append(f"          [FAIL]  {body[:400]}")

    q = spec.get("weekly_quota")
    if q:
        L.append("")
        L.append(f"  note     this plan allows {q} calls per week and "
                 f"{int(60 / spec.get('min_interval', 60))} per minute.")
        L.append(f"           {len(TRANSCRIPT)} call(s) made in this session. Calls are")
        L.append("           paced automatically; you do not need to add pauses.")

    _state["verified"] = ok
    if not quiet:
        print("\n".join(L))
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
    ]
    notes = sorted({c.notes for c in TRANSCRIPT if c.notes})
    if notes:
        md += ["- **Request adaptation:** " + "; ".join(notes)]
    md += [
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
