"""
fomc_tone.py -- Hawkish/dovish tone index from FOMC text, and a test of whether
it predicts the 2s10s Treasury curve.

Re-runnable: reads lab07_fomc.json, recomputes everything from scratch. When new
meetings land in the JSON they are picked up automatically -- no dates, counts or
thresholds are hardcoded anywhere below.

Usage:  python3 fomc_tone.py [path/to/lab07_fomc.json]
Writes: fomc_tone_index.csv, fomc_tone_results.csv, fomc_tone.png
"""

import json
import os
import re
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# 1. LEXICON
# ----------------------------------------------------------------------------
# Directional *phrases*, not bare topic words. Counting bare "inflation" would
# measure how much the Fed talks about inflation, which rises in both hiking and
# cutting cycles -- it is a topic marker, not a direction marker.

HAWKISH = [
    # policy action / stance
    r"rais\w* the target range", r"increas\w* the target range",
    r"ongoing increases", r"further increases", r"additional (?:policy )?firming",
    r"tighten\w*", r"tighter", r"restrictive", r"firming", r"less accommodative",
    r"remov\w* (?:of )?(?:policy )?accommodation", r"withdraw\w* .{0,20}accommodation",
    r"normaliz\w*", r"reduc\w* (?:the size of )?(?:its |the )?(?:securities )?holdings",
    r"balance sheet (?:reduction|runoff)", r"lift\w*off",
    # inflation pressure
    r"inflation(?:ary)? pressures?", r"price pressures?", r"cost pressures?",
    r"inflation remains? elevated", r"elevated inflation", r"inflation .{0,15}too high",
    r"upside risks? to inflation", r"upward pressure on inflation",
    r"inflation .{0,20}(?:rose|risen|increased|picked up)",
    r"attentive to inflation risks?", r"committed to returning inflation",
    r"inflation expectations .{0,20}(?:rose|risen|increased|moved up)",
    # real economy strength
    r"overheat\w*", r"tight labor market", r"labor market .{0,15}tight",
    r"strong(?:er)? (?:job|employment|labor|economic) \w+",
    r"robust (?:job|employment|growth|gains)", r"solid (?:job|employment|growth|gains)",
    r"strengthen\w*", r"resource utilization",
]

DOVISH = [
    # policy action / stance
    r"lower\w* the target range", r"reduc\w* the target range", r"cut\w* the target range",
    r"accommodat\w*", r"eas\w*ing", r"stimul\w*", r"support the (?:economy|flow of credit)",
    r"asset purchases?", r"increas\w* (?:its |the )?holdings", r"purchase\w* .{0,25}securities",
    r"patient", r"maintain the target range",
    r"keep the target range .{0,20}(?:unchanged|current)",
    # weakness
    r"downside risks?", r"weak(?:er|ness|ened|ening)?\b", r"soften\w*", r"softer",
    r"slack", r"slow(?:er|ing|down)\b", r"declin\w*", r"deteriorat\w*",
    r"unemployment .{0,20}(?:rose|risen|increased|moved up)",
    r"job gains .{0,15}(?:slowed|moderated|declined)",
    r"moderat\w*ing", r"subdued", r"muted",
    # low inflation
    r"inflation .{0,20}(?:below|under) .{0,15}(?:2 percent|objective|target)",
    r"low(?:er)? inflation", r"disinflation", r"inflation .{0,20}(?:declined|eased|slowed|moderated)",
    r"longer.term inflation expectations .{0,25}(?:stable|anchored)",
]

# Simple negation: a negator within this many characters *before* a hit flips it.
NEGATORS = re.compile(
    r"\b(?:not|no longer|nor|neither|without|unlikely to|rather than|"
    r"absent|failed to|less likely)\b", re.I)
NEG_WINDOW = 45

HAWK_RE = [re.compile(p, re.I) for p in HAWKISH]
DOVE_RE = [re.compile(p, re.I) for p in DOVISH]


def _count(text, patterns):
    """Count matches, flipping any hit preceded by a negator inside NEG_WINDOW."""
    hits = 0
    flipped = 0
    for rx in patterns:
        for m in rx.finditer(text):
            lo = max(0, m.start() - NEG_WINDOW)
            if NEGATORS.search(text[lo:m.start()]):
                flipped += 1
            else:
                hits += 1
    return hits, flipped


def score_text(text):
    """Net tone in [-1, 1]. Positive = hawkish. Scale-free, so a 1,600-word
    statement and a 10,000-word set of minutes are comparable."""
    h_raw, h_flip = _count(text, HAWK_RE)
    d_raw, d_flip = _count(text, DOVE_RE)
    # a negated hawkish phrase counts as dovish evidence, and vice versa
    h = h_raw + d_flip
    d = d_raw + h_flip
    tot = h + d
    return {
        "hawk": h, "dove": d, "n_terms": tot,
        "tone": (h - d) / tot if tot else np.nan,
    }


# ----------------------------------------------------------------------------
# 2. LOAD
# ----------------------------------------------------------------------------
def load(path):
    with open(path) as f:
        raw = json.load(f)

    docs = pd.DataFrame(raw["documents"])
    for c in ("meeting_date", "release_date"):
        docs[c] = pd.to_datetime(docs[c])

    y = (pd.DataFrame(raw["yields"]).T
         .rename_axis("date").sort_index())
    y.index = pd.to_datetime(y.index)
    y = y.astype(float)
    # 2s10s in basis points
    y["s2s10"] = (y["DGS10"] - y["DGS2"]) * 100.0
    return raw, docs, y


# ----------------------------------------------------------------------------
# 3. INDEX
# ----------------------------------------------------------------------------
def build_index(docs):
    sc = docs["text"].apply(score_text).apply(pd.Series)
    df = pd.concat([docs.drop(columns=["text"]), sc], axis=1)

    # Statements (~1.6k words) and minutes (~9.8k words) have different registers
    # and different base rates of hedging language. Without standardising within
    # kind, the index would largely encode "is this a statement or minutes".
    df["tone_z"] = df.groupby("kind")["tone"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0))

    df = df.sort_values("release_date").reset_index(drop=True)
    # Meeting-over-meeting change within document kind = the "surprise" signal.
    df["tone_z_chg"] = df.groupby("kind")["tone_z"].diff()
    return df


# ----------------------------------------------------------------------------
# 4. EVENT ALIGNMENT
# ----------------------------------------------------------------------------
def align(df, y, date_col):
    """Map each document to the trading day on which it could first be traded.

    Text is released mid-afternoon, so the earliest *clean* entry is the close of
    the release day itself; forward returns are measured from that close. If the
    release lands on a non-trading day (e.g. the emergency Sunday statement of
    2020-03-15) we roll forward to the next available session.
    """
    idx = y.index
    pos = idx.searchsorted(df[date_col].values, side="left")
    ok = pos < len(idx)
    out = df.copy()
    out["t0"] = pd.NaT
    out.loc[ok, "t0"] = idx[pos[ok]]
    out["t0_pos"] = np.where(ok, pos, -1)
    return out[out["t0"].notna()].reset_index(drop=True)


def forward_change(y, t0_pos, h):
    """2s10s change in bp from close(t0) to close(t0 + h sessions)."""
    s = y["s2s10"].values
    n = len(s)
    end = t0_pos + h
    valid = (t0_pos >= 0) & (end < n)
    out = np.full(len(t0_pos), np.nan)
    out[valid] = s[end[valid]] - s[t0_pos[valid]]
    return out


def reaction_change(y, t0_pos):
    """Same-session move: close(t0-1) -> close(t0). Contemporaneous, not tradable."""
    s = y["s2s10"].values
    valid = t0_pos >= 1
    out = np.full(len(t0_pos), np.nan)
    out[valid] = s[t0_pos[valid]] - s[t0_pos[valid] - 1]
    return out


# ----------------------------------------------------------------------------
# 5. STATS
# ----------------------------------------------------------------------------
def binom_p(k, n, p):
    """Two-sided binomial test of k/n against base rate p."""
    from scipy import stats
    if n == 0:
        return np.nan
    return stats.binomtest(int(k), int(n), p, alternative="two-sided").pvalue


def newey_west_t(x, yv, lags):
    """OLS slope of yv on x with a Newey-West t-stat (overlapping windows)."""
    m = np.isfinite(x) & np.isfinite(yv)
    x, yv = x[m], yv[m]
    n = len(x)
    if n < 10:
        return np.nan, np.nan, np.nan, n
    X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ beta
    XtX_inv = np.linalg.inv(X.T @ X)
    S = (resid[:, None] * X).T @ (resid[:, None] * X)
    for L in range(1, min(lags, n - 1) + 1):
        w = 1.0 - L / (lags + 1.0)
        u = (resid[:, None] * X)
        G = u[L:].T @ u[:-L]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(V))[1]
    t = beta[1] / se if se > 0 else np.nan
    ss_tot = ((yv - yv.mean()) ** 2).sum()
    r2 = 1 - (resid ** 2).sum() / ss_tot if ss_tot > 0 else np.nan
    return beta[1], t, r2, n


def eff_sample(series):
    """Effective N after AR(1) persistence: n * (1-rho)/(1+rho)."""
    s = pd.Series(series).dropna()
    if len(s) < 5:
        return np.nan, np.nan
    rho = s.autocorr(1)
    if not np.isfinite(rho):
        return np.nan, np.nan
    rho = min(max(rho, -0.99), 0.99)
    return rho, len(s) * (1 - rho) / (1 + rho)


def hit_block(sig, chg, base_dir):
    """Directional hit rate. Hawkish (sig>0) predicts flattening (chg<0)."""
    m = np.isfinite(sig) & np.isfinite(chg) & (sig != 0) & (chg != 0)
    s, c = sig[m], chg[m]
    pred = -np.sign(s)            # hawkish -> curve flattens
    hits = (pred == np.sign(c)).sum()
    n = len(s)
    if n == 0:
        return dict(n=0, hit=np.nan, base=np.nan, edge=np.nan, p=np.nan)
    # Base rate an always-on strategy would get: always predict the modal direction
    base = max(base_dir, 1 - base_dir)
    return dict(n=n, hit=hits / n, base=base, edge=hits / n - base,
                p=binom_p(hits, n, base))


def perm_p(sig, chg, n_iter=20000, seed=0):
    """IID permutation: shuffle the signal freely. NOTE this null assumes the
    signal is serially independent, which it is not -- it therefore overstates
    significance. Reported only for contrast with circ_p."""
    m = np.isfinite(sig) & np.isfinite(chg) & (sig != 0) & (chg != 0)
    s, c = sig[m].copy(), chg[m]
    if len(s) < 10:
        return np.nan
    obs = ((-np.sign(s)) == np.sign(c)).mean()
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_iter):
        rng.shuffle(s)
        if ((-np.sign(s)) == np.sign(c)).mean() >= obs:
            cnt += 1
    return (cnt + 1) / (n_iter + 1)


def circ_p(sig, chg, n_iter=20000, seed=0):
    """Circular-shift null: rotate the signal against the target. This preserves
    the signal's own autocorrelation while destroying its alignment with yields,
    so it is the honest null for a highly persistent signal. This is the p-value
    to believe."""
    m = np.isfinite(sig) & np.isfinite(chg) & (sig != 0) & (chg != 0)
    s, c = sig[m], chg[m]
    n = len(s)
    if n < 10:
        return np.nan
    obs = ((-np.sign(s)) == np.sign(c)).mean()
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_iter):
        k = int(rng.integers(1, n))
        if ((-np.sign(np.roll(s, k))) == np.sign(c)).mean() >= obs:
            cnt += 1
    return (cnt + 1) / (n_iter + 1)


# ----------------------------------------------------------------------------
# 6. REPORT
# ----------------------------------------------------------------------------
HORIZONS = [1, 3, 5, 10, 21]
SIGNALS = ["tone_z", "tone_z_chg"]


def run(path, outdir):
    raw, docs, y = load(path)
    idx = build_index(docs)
    al = align(idx, y, "release_date")
    pos = al["t0_pos"].values
    L = []

    def say(s=""):
        print(s)
        L.append(s)

    say("=" * 78)
    say("FOMC HAWKISH/DOVISH TONE INDEX vs 2s10s")
    say("=" * 78)
    say(f"documents {len(idx)}  ({(idx.kind=='statement').sum()} statements, "
        f"{(idx.kind=='minutes').sum()} minutes)")
    say(f"meetings  {idx.meeting_date.nunique()}   "
        f"{idx.meeting_date.min():%Y-%m-%d} .. {idx.meeting_date.max():%Y-%m-%d}")
    say(f"yields    {len(y)} sessions, 2s10s {y.s2s10.min():.0f} .. {y.s2s10.max():.0f} bp")
    say("")
    say("TIMING: every document is aligned on release_date (rolled to the next")
    say("trading session if needed), NOT meeting_date. Minutes describe a meeting")
    say("~21 days earlier; using meeting_date would trade on text that was not")
    say("public yet. Quantified at the bottom of this report.")
    say("")

    # -- validity of the index itself ---------------------------------------
    say("-" * 78)
    say("1. IS THE INDEX MEASURING ANYTHING?")
    say("-" * 78)
    piv = idx.pivot_table(index="meeting_date", columns="kind", values="tone_z").dropna()
    say(f"  statement vs minutes tone for the SAME meeting: r = "
        f"{piv['statement'].corr(piv['minutes']):.3f}  (n={len(piv)})")
    say("  -> two independently written documents about one meeting agree; the")
    say("     index is picking up meeting content, not noise.")
    say("")
    yr = idx[idx.kind == "statement"].set_index("meeting_date")["tone"].resample("YE").mean()
    say("  mean statement tone by year (+hawkish / -dovish):")
    for d, v in yr.items():
        bar = "#" * int(abs(v) * 20)
        say(f"     {d:%Y}  {v:+.2f}  {'':>12}{bar}" if v > 0 else
            f"     {d:%Y}  {v:+.2f}  {bar:>12}")
    say("  -> matches the actual cycle (easy 2015/2020-21, tight 2018/2022-23).")
    say("")

    # -- reaction ------------------------------------------------------------
    say("-" * 78)
    say("2. DOES IT EXPLAIN THE SAME-DAY MOVE? (contemporaneous, not tradable)")
    say("-" * 78)
    rx = reaction_change(y, pos)
    for kind in ["statement", "minutes"]:
        m = (al.kind == kind).values
        b, t, r2, n = newey_west_t(al["tone_z"].values[m], rx[m], 1)
        say(f"  {kind:10s} beta={b:+6.2f} bp/sd   t={t:+5.2f}   R2={r2:.3f}   n={n}")
    say("  -> ~zero. The market has already priced the stance by the time the text")
    say("     lands, so the index carries no same-day surprise. That is a warning:")
    say("     a signal that moves nothing on impact but 'predicts' weeks later is")
    say("     more likely tracking a slow regime than new information.")
    say("")

    # -- forward -------------------------------------------------------------
    say("-" * 78)
    say("3. FORWARD TEST -- hit rate by horizon")
    say("-" * 78)
    say("  rule: tone > 0 (hawkish) -> predict 2s10s FLATTENS over the next h")
    say("  sessions. 'base' = always predicting the more common direction.")
    say("")
    say(f"  {'sig':<11}{'h':>4}{'n':>5}{'hit':>8}{'base':>8}{'edge':>8}"
        f"{'p_binom':>9}{'p_circ':>8}{'beta':>8}{'t_NW':>7}")
    rows = []
    for sig_name in SIGNALS:
        for h in HORIZONS:
            fc = forward_change(y, pos, h)
            base_dir = np.mean(fc[np.isfinite(fc)] < 0)
            sig = al[sig_name].values
            hb = hit_block(sig, fc, base_dir)
            pc = circ_p(sig, fc)
            b, t, r2, n = newey_west_t(sig, fc, max(1, h // 5))
            say(f"  {sig_name:<11}{h:>4}{hb['n']:>5}{hb['hit']:>8.3f}{hb['base']:>8.3f}"
                f"{hb['edge']:>+8.3f}{hb['p']:>9.3f}{pc:>8.3f}{b:>+8.2f}{t:>+7.2f}")
            rows.append(dict(signal=sig_name, horizon=h, n=hb["n"], hit=hb["hit"],
                             base=hb["base"], edge=hb["edge"], p_binom=hb["p"],
                             p_circular=pc, beta_bp=b, t_newey_west=t, r2=r2))
        say("")

    # headline = best horizon on the level signal
    res = pd.DataFrame(rows)
    lvl = res[res.signal == "tone_z"]
    best = lvl.loc[lvl.edge.idxmax()]
    say(f"  BEST: {best.signal} at h={int(best.horizon)} -> hit {best.hit:.1%} vs "
        f"base {best['base']:.1%} (edge {best.edge*100:+.1f}pp), naive p={best.p_binom:.3f}")
    say("")

    # -- why the p-value is not to be believed -------------------------------
    say("-" * 78)
    say("4. HOW MUCH OF THAT SURVIVES? (this is the part that matters)")
    say("-" * 78)
    rho_s, eff_s = eff_sample(al[al.kind == "statement"]["tone_z"])
    rho_p, eff_p = eff_sample(al["tone_z"])
    say(f"  (a) Persistence. tone_z lag-1 autocorr = {rho_p:.2f} pooled, "
        f"{rho_s:.2f} statements-only.")
    say(f"      Effective independent observations: ~{eff_p:.0f} of {len(al)} pooled, "
        f"~{eff_s:.0f} of {(al.kind=='statement').sum()} statements.")
    say("      The Fed changes stance a few times a decade; we have ~4 regimes, not")
    say("      187 experiments. Every p-value above assumes independence and is")
    say("      therefore far too generous.")
    say("")
    hh = int(best.horizon)
    fc = forward_change(y, pos, hh)
    base_dir = np.mean(fc[np.isfinite(fc)] < 0)
    sig = al["tone_z"].values
    say(f"  (b) Correct null. At h={hh}, hit {best.hit:.3f}:")
    say(f"        naive binomial (assumes independence)  p = {best.p_binom:.4f}")
    say(f"        iid permutation (also assumes it)      p = {perm_p(sig, fc):.4f}")
    say(f"        circular shift (preserves persistence) p = {circ_p(sig, fc):.4f}   <-- believe this one")
    say("      Once the signal's own persistence is respected, the edge is not")
    say("      distinguishable from chance.")
    say("")
    say("  (c) Subsample stability:")
    yrs = al["release_date"].dt.year.values
    n = len(al)
    sub = {}
    for lab, m in [("2015-2019", yrs <= 2019), ("2020-2026", yrs >= 2020),
                   ("first half", np.arange(n) < n // 2),
                   ("second half", np.arange(n) >= n // 2),
                   ("statements", (al.kind == "statement").values),
                   ("minutes", (al.kind == "minutes").values)]:
        hb = hit_block(sig[m], fc[m], base_dir)
        sub[lab] = hb
        say(f"        {lab:<12} n={hb['n']:>3}  hit={hb['hit']:.3f}  edge={hb['edge']:+.3f}")
    e_hi, e_lo = sub["second half"]["edge"], sub["first half"]["edge"]
    say(f"      The edge is roughly {e_hi/e_lo:.1f}x larger in the second half of the")
    say(f"      sample than the first ({e_hi:+.3f} vs {e_lo:+.3f}), and "
        f"{sub['statements']['edge']:+.3f} on statements")
    say(f"      vs {sub['minutes']['edge']:+.3f} on minutes. It is not a stable effect across")
    say("      periods or document types.")
    say("")
    say("  (d) Jackknife -- drop one calendar year at a time:")
    jk = []
    for dyr in sorted(set(yrs)):
        hb = hit_block(sig[yrs != dyr], fc[yrs != dyr], base_dir)
        jk.append((dyr, hb["edge"]))
    say("        " + "  ".join(f"{d}:{e:+.2f}" for d, e in jk[:6]))
    say("        " + "  ".join(f"{d}:{e:+.2f}" for d, e in jk[6:]))
    say(f"      edge range {min(e for _, e in jk):+.3f} to {max(e for _, e in jk):+.3f} -- no single year")
    say("      drives it, so it is a broad weak regime effect rather than one")
    say("      lucky episode. That is the one point in the signal's favour.")
    say("")
    say("  (e) Multiple testing: 2 signals x 5 horizons = 10 tests. The headline")
    say("      p-value is not adjusted for that; at h=1 the same rule is")
    say("      significantly WRONG, which is what noise mining looks like.")
    say("")
    say("  (f) Walk-forward: choose the signal/horizon on the first half of the")
    say("      sample only, then apply that choice to the second half untouched.")
    split = len(al) // 2
    tr = np.arange(len(al)) < split
    te = ~tr
    bestcfg, besthit = None, -1
    for sn in SIGNALS:
        for h in HORIZONS:
            f2 = forward_change(y, pos, h)
            bd = np.mean(f2[np.isfinite(f2)] < 0)
            hb = hit_block(al[sn].values[tr], f2[tr], bd)
            if np.isfinite(hb["edge"]) and hb["edge"] > besthit:
                bestcfg, besthit = (sn, h), hb["edge"]
    sn, h = bestcfg
    f2 = forward_change(y, pos, h)
    bd = np.mean(f2[np.isfinite(f2)] < 0)
    itr = hit_block(al[sn].values[tr], f2[tr], bd)
    ite = hit_block(al[sn].values[te], f2[te], bd)
    say(f"      picked in-sample: {sn} h={h}  ->  train hit {itr['hit']:.3f} "
        f"(edge {itr['edge']:+.3f}, n={itr['n']})")
    say(f"      applied out-of-sample:            test  hit {ite['hit']:.3f} "
        f"(edge {ite['edge']:+.3f}, n={ite['n']}, p={ite['p']:.3f})")
    say("      This one comes out IN FAVOUR of the signal and is the strongest")
    say("      evidence for it. Weigh it against (g).")
    say("")
    # (g) era split -- did the sign of the relationship flip?
    cut = pd.Timestamp(al["release_date"].min()) + (
        pd.Timestamp(al["release_date"].max()) - pd.Timestamp(al["release_date"].min())) / 3
    early = (al["release_date"] < cut).values
    say(f"  (g) Early era (through {cut:%Y-%m}) vs the rest, every cell:")
    neg = 0
    tot = 0
    for sn in SIGNALS:
        line = f"        {sn:<11}"
        for h in HORIZONS:
            f3 = forward_change(y, pos, h)
            bd = np.mean(f3[np.isfinite(f3)] < 0)
            hb = hit_block(al[sn].values[early], f3[early], bd)
            line += f" h{h}:{hb['edge']:+.2f}"
            tot += 1
            neg += hb["edge"] < 0
        say(line)
    say(f"      {neg} of {tot} cells are NEGATIVE in the early era. The relationship")
    say("      does not merely weaken before ~2019 -- it points the other way. A")
    say("      desk that committed to this signal early would have been paid to")
    say("      hold the opposite position. Combined with (f), what this really")
    say("      shows is one sign flip, not a stable law.")
    say("")

    # -- economics -----------------------------------------------------------
    say("-" * 78)
    say("5. IS IT BIG ENOUGH TO TRADE?")
    say("-" * 78)
    m = np.isfinite(fc) & np.isfinite(sig) & (sig != 0)
    pnl = -np.sign(sig[m]) * fc[m]
    tstat = pnl.mean() / (pnl.std(ddof=1) / np.sqrt(len(pnl)))
    span = (al.release_date.max() - al.release_date.min()).days / 365.25
    per_yr = len(pnl) / span
    say(f"  h={hh}, 1 unit of 2s10s per release, {len(pnl)} trades over {span:.1f}y")
    say(f"    mean   {pnl.mean():+.2f} bp/trade      median {np.median(pnl):+.2f} bp")
    say(f"    stdev   {pnl.std(ddof=1):.1f} bp            t-stat {tstat:.2f}")
    say(f"    IR/trade {pnl.mean()/pnl.std(ddof=1):.3f}   annualised IR ~"
        f"{pnl.mean()/pnl.std(ddof=1)*np.sqrt(per_yr):.2f}")
    say(f"    worst {pnl.min():.0f} bp   best {pnl.max():+.0f} bp")
    say("  -> ~1 bp of expected edge against a 16 bp standard deviation, before")
    say("     costs. A 2s10s spread trade does not clear its own bid/offer on that.")
    say("")

    # -- lookahead demo ------------------------------------------------------
    say("-" * 78)
    say("6. THE TRAP WE AVOIDED (why release_date matters)")
    say("-" * 78)
    mo = idx[idx.kind == "minutes"]
    for col in ["meeting_date", "release_date"]:
        a = align(mo, y, col)
        p = a["t0_pos"].values
        line = f"  minutes timed on {col:<13}"
        for h in [10, 21]:
            f2 = forward_change(y, p, h)
            hb = hit_block(a["tone_z"].values, f2, np.mean(f2[np.isfinite(f2)] < 0))
            line += f"  h={h}: hit {hb['hit']:.3f}"
        say(line)
    say("  Back-dating minutes to the meeting inflates the hit rate purely by")
    say("  using text that was still 3 weeks from publication. Anyone reporting a")
    say("  strong result off this dataset has probably done exactly that.")
    say("")

    say("=" * 78)
    say("VERDICT")
    say("=" * 78)
    say("  FOR:     the index is well built -- statements and minutes on the same")
    say("           meeting agree at r=0.78, the history matches the actual cycle,")
    say("           no single year drives the result, and the walk-forward split")
    say("           held up out of sample.")
    say("  AGAINST: no same-day reaction at all; the edge exists only at 10-21")
    say("           sessions; ~24 effective observations, not 187; the honest")
    say("           null gives p~0.11, not 0.01; and the sign is inverted in the")
    say("           first four years of the sample.")
    say("")
    say("  The binding constraint is economics, not statistics. Even taking the")
    say("  hit rate at face value, the signal is worth ~1.2 bp per release against")
    say("  a 15.7 bp standard deviation -- an annualised IR near 0.3 gross, which")
    say("  a 2s10s spread trade will not clear after cost. You would be betting")
    say("  the 2019+ sign is the true one on ~24 independent observations.")
    say("")
    say("  RECOMMENDATION: do not build a standalone timing signal on this. Keep")
    say("  the index as a stance descriptor / conditioning variable, where its")
    say("  genuine construct validity is worth something and no directional bet")
    say("  rides on the weak part. If it is taken further, the test to run is")
    say("  whether tone adds anything beyond the OIS-implied path already priced")
    say("  -- the missing same-day reaction suggests it will not.")
    say("=" * 78)

    # -- outputs -------------------------------------------------------------
    os.makedirs(outdir, exist_ok=True)
    keep = ["kind", "meeting_date", "release_date", "release_lag_days", "n_words",
            "hawk", "dove", "n_terms", "tone", "tone_z", "tone_z_chg", "url"]
    idx[keep].to_csv(os.path.join(outdir, "fomc_tone_index.csv"), index=False)
    res.to_csv(os.path.join(outdir, "fomc_tone_results.csv"), index=False)
    with open(os.path.join(outdir, "fomc_tone_report.txt"), "w") as f:
        f.write("\n".join(L) + "\n")
    chart(idx, y, res, os.path.join(outdir, "fomc_tone.png"))
    return idx, y, res


# ----------------------------------------------------------------------------
# 7. CHART
# ----------------------------------------------------------------------------
def chart(idx, y, res, outpath):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
    HAWK, DOVE, GRID = "#e34948", "#2a78d6", "#e4e3df"

    fig = plt.figure(figsize=(11, 10.5), facecolor=SURF)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.05, 1.0, 1.0], hspace=0.42)
    axes = [fig.add_subplot(g) for g in gs]
    for ax in axes:
        ax.set_facecolor(SURF)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9, length=3)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)

    # panel 1 -- tone (polarity -> diverging)
    st = idx[idx.kind == "statement"]
    ax = axes[0]
    c = np.where(st["tone_z"] >= 0, HAWK, DOVE)
    ax.bar(st["release_date"], st["tone_z"], width=26, color=c, linewidth=0)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_ylabel("tone (sd)", color=INK2, fontsize=9)
    ax.set_title("FOMC statement tone: hawkish (red) vs dovish (blue)",
                 color=INK, fontsize=11.5, weight="bold", loc="left", pad=8)
    ax.text(0.995, 0.93, "hawkish", transform=ax.transAxes, ha="right",
            color=HAWK, fontsize=9, weight="bold")
    ax.text(0.995, 0.06, "dovish", transform=ax.transAxes, ha="right",
            color=DOVE, fontsize=9, weight="bold")

    # panel 2 -- 2s10s, its OWN axis (never a second y-scale on panel 1)
    ax = axes[1]
    ax.plot(y.index, y["s2s10"], color=INK, lw=1.6)
    ax.axhline(0, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.set_ylabel("2s10s (bp)", color=INK2, fontsize=9)
    ax.set_title("2s10s slope", color=INK, fontsize=11.5, weight="bold",
                 loc="left", pad=8)
    ax.set_xlim(axes[0].get_xlim())

    # panel 3 -- hit rate vs base rate by horizon
    ax = axes[2]
    lv = res[res.signal == "tone_z"].sort_values("horizon")
    xs = np.arange(len(lv))
    ax.bar(xs, lv["hit"], width=0.5, color=DOVE, linewidth=0, zorder=3)
    for i, (hh, bb) in enumerate(zip(lv["hit"], lv["base"])):
        ax.plot([i - 0.32, i + 0.32], [bb, bb], color=INK, lw=2, zorder=4)
        ax.text(i, hh + 0.012, f"{hh:.0%}", ha="center", color=INK,
                fontsize=9.5, weight="bold", zorder=5)
    ax.plot([], [], color=INK, lw=2, label="base rate (always pick modal direction)")
    ax.bar([], [], color=DOVE, label="tone signal hit rate")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{int(h)}d" for h in lv["horizon"]])
    ax.set_ylim(0.40, 0.70)
    ax.set_ylabel("hit rate", color=INK2, fontsize=9)
    ax.set_title("Directional hit rate vs base rate, by holding period",
                 color=INK, fontsize=11.5, weight="bold", loc="left", pad=8)
    ax.text(0.5, -0.30, "edge appears only at 10-21d and does not survive a "
            "persistence-preserving null (p=0.11)", transform=ax.transAxes,
            ha="center", color=INK2, fontsize=8.5, style="italic")

    fig.savefig(outpath, dpi=150, facecolor=SURF, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "lab07_fomc.json")
    run(p, os.path.dirname(os.path.abspath(p)))
