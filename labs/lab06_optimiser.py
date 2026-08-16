#!/usr/bin/env python3
"""
Mean-variance optimiser for the Ken French 49 industry portfolios.

Produces, for a configurable estimation window:
  * the efficient frontier
  * the minimum-variance portfolio
  * the tangency (max-Sharpe) portfolio
  * an equal-weighted (1/N) reference portfolio
with weights, annualised return and annualised volatility for each.

Re-run after a data refresh with:  python3 mv_optimiser.py
Everything tunable lives in CONFIG below. Dependencies: numpy, scipy,
pandas, matplotlib only (Ledoit-Wolf shrinkage is implemented here rather
than imported, so no sklearn is required).

-------------------------------------------------------------------------
READ THIS BEFORE USING THE 36-MONTH NUMBERS
-------------------------------------------------------------------------
There are 49 assets. A 36-month window gives 36 observations. When the
number of assets N is greater than or equal to the number of months T, the
sample covariance matrix is SINGULAR (rank <= T-1 = 35 < 49). It cannot be
inverted, so the textbook mean-variance solution is not merely imprecise,
it is undefined.

Concretely, at N=49 / T=36 there are 14 directions in portfolio space with
EXACTLY ZERO estimated variance. With short-selling allowed you can
therefore construct a portfolio that appears to be riskless and has an
arbitrarily large expected return: the frontier becomes a vertical line and
the tangency portfolio has an infinite Sharpe ratio. Those are artifacts of
having too little data, not investment opportunities.

This script does two things about that:
  1. It defaults to LONG_ONLY = True and COV_ESTIMATOR = 'ledoit_wolf'.
     The no-shorting constraint and the shrinkage together keep the problem
     well-posed and the answer finite even when the raw covariance is
     singular.
  2. It always prints a diagnostics block, and refuses to use the
     closed-form (shorting-allowed) solution when the covariance matrix is
     rank-deficient.

Shrinkage repairs the *covariance*. Nothing repairs the *means*: over 36
months the standard error of each industry's mean return is about the same
size as the entire cross-sectional spread of those means, so the tangency
portfolio -- which is driven entirely by the ranking of the means -- is
close to noise at this window length. Set WINDOW_MONTHS = 120 for estimates
that are actually identified. The sensitivity table at the bottom of the
run shows how much the answer moves with the window.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "DATA_FILE": os.path.join(HERE, "lab06_returns.json"),
    "OUTDIR": HERE,

    # Estimation window, in months. 36 = "last three years" as requested.
    # See the header note: 36 < 49 assets, so this window cannot identify
    # the covariance matrix. 120 is the shortest window here that is
    # comfortably well-conditioned.
    "WINDOW_MONTHS": 36,

    # No short selling. Strongly recommended to leave True whenever
    # WINDOW_MONTHS <= number of assets, otherwise the frontier is unbounded.
    "LONG_ONLY": True,

    # 'ledoit_wolf' (shrinkage to a scaled identity) or 'sample'.
    "COV_ESTIMATOR": "ledoit_wolf",

    # Windows compared in the sensitivity table and the right-hand chart panel.
    "SENSITIVITY_WINDOWS": [36, 60, 120, 240],

    # Number of points traced along the frontier.
    "N_FRONTIER": 60,

    # Periods per year for annualisation.
    "PERIODS_PER_YEAR": 12,

    # Rolling out-of-sample backtest: re-estimate every month on the trailing
    # window, hold one month, measure what was actually realised. This is the
    # only honest test of whether the optimiser adds value, and it is what
    # justifies the window recommendation. Adds ~1-2 min to a run.
    "RUN_BACKTEST": True,
    "BACKTEST_WINDOWS": [36, 120],
}

# Palette (validated categorical slots 1-3, plus ink tokens).
C_FRONTIER = "#2a78d6"   # slot 1 blue
C_TANGENCY = "#eb6834"   # slot 2 orange
C_MINVAR = "#1baf7a"     # slot 3 aqua
C_W60, C_W120, C_W240 = "#eb6834", "#1baf7a", "#eda100"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8981"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"


# --------------------------------------------------------------------------
# Data loading and validation
# --------------------------------------------------------------------------
def load_data(path):
    """Load the returns file and return decimals, not percent."""
    with open(path) as fh:
        d = json.load(fh)

    dates = list(d["dates"])
    industries = list(d["industries"])
    R = np.asarray(d["returns_pct"], dtype=float)

    if R.shape != (len(dates), len(industries)):
        raise ValueError(
            f"returns_pct is {R.shape}, expected {(len(dates), len(industries))}"
        )

    # Ken French codes missing observations as -99.99 / -999. They are left
    # in the file deliberately; treat them as missing rather than as a -99%
    # return, which would wreck both the mean and the covariance.
    sentinels = set(d.get("missing_code", [-99.99, -999.0]))
    mask = np.isin(R, list(sentinels))
    if mask.any():
        R = R.copy()
        R[mask] = np.nan

    rf_map = d.get("risk_free_pct", {})
    rf = np.array([rf_map.get(dt, np.nan) for dt in dates], dtype=float)

    return {
        "dates": dates,
        "industries": industries,
        "R": R / 100.0,       # decimal monthly returns
        "rf": rf / 100.0,     # decimal monthly risk-free
        "meta": {k: d.get(k) for k in ("dataset", "source", "frequency", "units")},
        "n_missing": int(mask.sum()),
    }


def take_window(data, months):
    """Most recent `months` observations. Drops assets with any missing data."""
    R = data["R"][-months:]
    rf = data["rf"][-months:]
    dates = data["dates"][-months:]
    names = list(data["industries"])

    ok = ~np.isnan(R).any(axis=0)
    if not ok.all():
        dropped = [n for n, k in zip(names, ok) if not k]
        print(f"  ! dropped {len(dropped)} industries with missing data "
              f"in this window: {', '.join(dropped)}")
        R = R[:, ok]
        names = [n for n, k in zip(names, ok) if k]

    return {"R": R, "rf": rf, "dates": dates, "names": names}


def diagnostics(win, label=""):
    """Report whether this window can identify a covariance matrix at all."""
    R = win["R"]
    T, N = R.shape
    S = np.cov(R, rowvar=False, ddof=1)
    ev = np.linalg.eigvalsh(S)
    rank = int(np.linalg.matrix_rank(S))
    cond = float(np.linalg.cond(S))
    n_zero = int((ev <= max(ev.max(), 0) * 1e-12).sum())

    mu = R.mean(axis=0)
    se = R.std(axis=0, ddof=1) / np.sqrt(T)
    spread = mu.std(ddof=1)
    noise_signal = float(np.median(se) / spread) if spread > 0 else np.inf

    singular = rank < N

    print(f"  {label}T={T} months, N={N} assets"
          f"   rank(cov)={rank}   cond={cond:.3g}")
    if singular:
        print(f"  {label}** SINGULAR covariance: {n_zero} zero eigenvalues, "
              f"i.e. {N - rank} portfolio directions with exactly zero")
        print(f"  {label}   estimated risk. Sigma^-1 does not exist; the "
              f"unconstrained frontier is unbounded.")
    print(f"  {label}mean-return noise/signal = {noise_signal:.2f} "
          f"(median s.e. of a mean / cross-sectional spread of the means)")
    if noise_signal > 0.5:
        print(f"  {label}** Expected returns are not identified at this window "
              f"length; treat the tangency portfolio as indicative only.")

    return {"T": T, "N": N, "rank": rank, "cond": cond,
            "singular": singular, "noise_signal": noise_signal}


# --------------------------------------------------------------------------
# Covariance estimators
# --------------------------------------------------------------------------
def cov_sample(R):
    return np.cov(R, rowvar=False, ddof=1), 0.0


def cov_ledoit_wolf(R):
    """
    Ledoit-Wolf (2004) shrinkage toward a scaled identity.

    Sigma* = k*mu*I + (1-k)*S, with k chosen to minimise expected squared
    error. Guarantees a well-conditioned, invertible estimate even when
    T < N. Returns (Sigma, shrinkage_intensity).
    """
    T, N = R.shape
    X = R - R.mean(axis=0)
    S = X.T @ X / T                       # MLE covariance
    mu = np.trace(S) / N
    F = mu * np.eye(N)                    # shrinkage target

    d2 = np.sum((S - F) ** 2) / N         # ||S - F||^2 / N

    # b^2 = (1/(T^2 N)) * sum_t ||x_t x_t' - S||^2, expanded so the sum over
    # observations is vectorised:
    #   ||x x' - S||^2 = (x'x)^2 - 2 x'Sx + ||S||^2
    sq = np.einsum("ti,ti->t", X, X)              # x_t' x_t
    quad = np.einsum("ti,ij,tj->t", X, S, X)      # x_t' S x_t
    b2 = (np.sum(sq ** 2) - 2.0 * np.sum(quad) + T * np.sum(S ** 2))
    b2 /= (T ** 2) * N
    b2 = min(b2, d2)                      # 0 <= b2 <= d2

    k = b2 / d2 if d2 > 0 else 1.0        # shrinkage intensity in [0, 1]
    return k * F + (1.0 - k) * S, float(k)


ESTIMATORS = {"sample": cov_sample, "ledoit_wolf": cov_ledoit_wolf}


# --------------------------------------------------------------------------
# Portfolio maths
# --------------------------------------------------------------------------
def port_stats(w, mu, S, rf_m, ppy=12):
    """Annualised return, annualised vol and Sharpe for weights w."""
    m = float(w @ mu)
    v = float(np.sqrt(max(w @ S @ w, 0.0)))
    ann_r = m * ppy
    ann_v = v * np.sqrt(ppy)
    ann_rf = rf_m * ppy
    sharpe = (ann_r - ann_rf) / ann_v if ann_v > 0 else np.nan
    return {"ann_return": ann_r, "ann_vol": ann_v, "sharpe": sharpe}


def _solve(obj, N, long_only, extra_constraints=(), x0=None):
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    cons.extend(extra_constraints)
    bounds = [(0.0, 1.0)] * N if long_only else [(-2.0, 2.0)] * N
    if x0 is None:
        x0 = np.full(N, 1.0 / N)
    res = minimize(obj, x0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 800, "ftol": 1e-12})
    return res.x, res.success


def min_variance(mu, S, long_only):
    N = len(mu)
    w, _ = _solve(lambda w: w @ S @ w, N, long_only)
    return w


def frontier(mu, S, long_only, n_points):
    """Trace the minimum-variance frontier over a grid of target returns."""
    N = len(mu)
    w_mv = min_variance(mu, S, long_only)
    r_mv = float(w_mv @ mu)
    r_hi = float(mu.max()) if long_only else float(mu.max()) * 1.5
    targets = np.linspace(r_mv, r_hi, n_points)

    ws, rows = [], []
    x0 = w_mv.copy()
    for tgt in targets:
        cons = [{"type": "eq", "fun": (lambda w, t=tgt: w @ mu - t)}]
        w, ok = _solve(lambda w: w @ S @ w, N, long_only,
                       extra_constraints=cons, x0=x0)
        if not ok:
            continue
        x0 = w
        ws.append(w)
        rows.append((float(w @ mu), float(np.sqrt(max(w @ S @ w, 0.0)))))
    return np.array(ws), np.array(rows)


def tangency(mu, S, rf_m, long_only, fw=None):
    """
    Max-Sharpe portfolio.

    Shorting allowed AND covariance invertible -> closed form.
    Otherwise -> frontier scan for a starting point, then SLSQP refinement.
    The scan matters: the long-only max-Sharpe surface is not concave, so a
    naive single-start solve can stop at a local optimum.
    """
    N = len(mu)
    ex = mu - rf_m

    if not long_only:
        if np.linalg.matrix_rank(S) < N:
            raise np.linalg.LinAlgError(
                "covariance is singular: the shorting-allowed tangency "
                "portfolio is unbounded (infinite Sharpe). Use LONG_ONLY=True "
                "or a longer window."
            )
        z = np.linalg.solve(S, ex)
        return z / z.sum()

    def neg_sharpe(w):
        v = np.sqrt(max(w @ S @ w, 1e-18))
        return -(w @ ex) / v

    starts = []
    if fw is not None and len(fw):
        sr = np.array([-neg_sharpe(w) for w in fw])
        starts.append(fw[int(np.argmax(sr))])
    starts.append(np.full(N, 1.0 / N))
    pos = np.clip(ex, 0, None)
    if pos.sum() > 0:
        starts.append(pos / pos.sum())

    best, best_sr = None, -np.inf
    for x0 in starts:
        w, ok = _solve(neg_sharpe, N, True, x0=x0)
        if ok and -neg_sharpe(w) > best_sr:
            best, best_sr = w, -neg_sharpe(w)
    return best


# --------------------------------------------------------------------------
# Run one window end to end
# --------------------------------------------------------------------------
def run_window(data, months, long_only, estimator, n_frontier, ppy, verbose=True):
    win = take_window(data, months)
    R, names = win["R"], win["names"]
    rf_m = float(np.nanmean(win["rf"]))

    if verbose:
        print(f"\n  window: {win['dates'][0]} -> {win['dates'][-1]}  "
              f"({months} months)")
    diag = diagnostics(win, label="  ") if verbose else None

    mu = R.mean(axis=0)
    S, shrink = ESTIMATORS[estimator](R)
    if verbose and estimator == "ledoit_wolf":
        print(f"  Ledoit-Wolf shrinkage intensity = {shrink:.3f}   "
              f"cond after shrinkage = {np.linalg.cond(S):.4g}")

    fw, fpts = frontier(mu, S, long_only, n_frontier)
    w_mv = min_variance(mu, S, long_only)
    w_tan = tangency(mu, S, rf_m, long_only, fw=fw)
    w_eq = np.full(len(names), 1.0 / len(names))

    ports = {
        "min_variance": w_mv,
        "tangency": w_tan,
        "equal_weight_1overN": w_eq,
    }
    stats = {k: port_stats(w, mu, S, rf_m, ppy) for k, w in ports.items()}

    # Equal-weight reference is also worth reporting on realised numbers,
    # since it does not depend on the covariance estimate at all.
    eq_series = R @ w_eq
    stats["equal_weight_1overN"]["ann_vol_realised"] = float(
        eq_series.std(ddof=1) * np.sqrt(ppy))

    return {
        "months": months, "names": names, "dates": win["dates"],
        "mu": mu, "S": S, "rf_m": rf_m, "shrink": shrink,
        "frontier_w": fw, "frontier_pts": fpts,
        "ports": ports, "stats": stats, "diag": diag,
        "asset_pts": np.column_stack([R.std(axis=0, ddof=1) * np.sqrt(ppy),
                                      mu * ppy]),
    }


# --------------------------------------------------------------------------
# Rolling out-of-sample backtest
# --------------------------------------------------------------------------
def backtest(data, windows, long_only, estimator, ppy):
    """
    Walk-forward test. At each month t, estimate on the trailing `W` months,
    form the portfolios, hold for month t+1, record the realised return.
    No look-ahead. Compares against equal-weight, which uses no estimates
    at all and is therefore the honest hurdle.
    """
    R, rf = data["R"], data["rf"]
    T, N = R.shape
    out = []

    for W in windows:
        if W >= T:
            continue
        rows = []
        for t in range(W, T):
            Rw = R[t - W:t]
            if np.isnan(Rw).any() or np.isnan(R[t]).any():
                continue
            mu = Rw.mean(axis=0)
            S, _ = ESTIMATORS[estimator](Rw)
            rf_m = float(np.nanmean(rf[t - W:t]))
            w_mv = min_variance(mu, S, long_only)
            w_tan = tangency(mu, S, rf_m, long_only)
            rows.append((R[t] @ w_mv, R[t] @ w_tan, R[t].mean(), rf[t]))
        a = np.asarray(rows)
        if not len(a):
            continue
        rf_ann = float(np.nanmean(a[:, 3])) * ppy
        for i, nm in enumerate(("min_variance", "tangency",
                                "equal_weight_1overN")):
            x = a[:, i]
            ar = float(x.mean()) * ppy
            av = float(x.std(ddof=1)) * np.sqrt(ppy)
            out.append({
                "window_m": W,
                "portfolio": nm,
                "oos_months": len(a),
                "oos_ann_return_pct": 100 * ar,
                "oos_ann_vol_pct": 100 * av,
                "oos_sharpe": (ar - rf_ann) / av if av > 0 else np.nan,
            })
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
def write_tables(res, outdir, ppy):
    names = res["names"]
    w = pd.DataFrame({k: v for k, v in res["ports"].items()}, index=names)
    w.index.name = "industry"
    w = w.sort_values("tangency", ascending=False)
    wpath = os.path.join(outdir, "weights.csv")
    w.round(6).to_csv(wpath)

    rows = []
    for k, s in res["stats"].items():
        rows.append({
            "portfolio": k,
            "ann_return_pct": 100 * s["ann_return"],
            "ann_vol_pct": 100 * s["ann_vol"],
            "sharpe": s["sharpe"],
            "n_holdings": int((res["ports"][k] > 1e-4).sum()),
            "max_weight_pct": 100 * float(res["ports"][k].max()),
        })
    summ = pd.DataFrame(rows)
    spath = os.path.join(outdir, "summary.csv")
    summ.round(4).to_csv(spath, index=False)
    return w, summ, wpath, spath


def make_chart(res, sens, outdir, cfg):
    ppy = cfg["PERIODS_PER_YEAR"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6))
    fig.patch.set_facecolor(SURFACE)

    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8, linestyle="-")
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
            ax.spines[s].set_linewidth(0.8)
        ax.tick_params(colors=INK2, labelsize=9, length=0)

    # ---- Panel 1: the requested window --------------------------------
    ap = res["asset_pts"] * 100
    ax1.scatter(ap[:, 0], ap[:, 1], s=16, color=INK3, alpha=0.45,
                linewidths=0, zorder=2, label="Individual industries")

    f = res["frontier_pts"]
    ax1.plot(f[:, 1] * np.sqrt(ppy) * 100, f[:, 0] * ppy * 100,
             color=C_FRONTIER, linewidth=2, zorder=4, label="Efficient frontier")

    # Offsets are hand-placed so the three labels never collide with each
    # other; re-check them if the window or the data changes materially.
    pts = [("min_variance", "Minimum variance", C_MINVAR, (9, -30)),
           ("tangency", "Tangency", C_TANGENCY, (11, -4))]
    for key, lab, col, off in pts:
        s = res["stats"][key]
        x, y = 100 * s["ann_vol"], 100 * s["ann_return"]
        ax1.scatter([x], [y], s=95, color=col, zorder=6,
                    edgecolors=SURFACE, linewidths=2, label=lab)
        ax1.annotate(f"{lab}\n{y:.1f}% / {x:.1f}%", (x, y),
                     textcoords="offset points", xytext=off,
                     fontsize=9, color=INK, weight="medium", zorder=7)

    s = res["stats"]["equal_weight_1overN"]
    xe, ye = 100 * s["ann_vol"], 100 * s["ann_return"]
    ax1.scatter([xe], [ye], s=85, marker="D", color=INK, zorder=6,
                edgecolors=SURFACE, linewidths=2, label="Equal weight (1/N)")
    ax1.annotate(f"1/N (benchmark proxy)\n{ye:.1f}% / {xe:.1f}%", (xe, ye),
                 textcoords="offset points", xytext=(11, 9),
                 fontsize=9, color=INK, weight="medium", zorder=7)

    ax1.set_title(
        f"Efficient frontier — {res['months']}-month window "
        f"({res['dates'][0]}–{res['dates'][-1]})",
        fontsize=12, color=INK, loc="left", pad=12, weight="semibold")
    ax1.set_xlabel("Annualised volatility (%)", fontsize=10, color=INK2)
    ax1.set_ylabel("Annualised return (%)", fontsize=10, color=INK2)
    leg = ax1.legend(frameon=False, fontsize=9, loc="lower right")
    for t in leg.get_texts():
        t.set_color(INK2)

    # ---- Panel 2: how much the frontier moves with the window ----------
    cols = {36: C_FRONTIER, 60: C_W60, 120: C_W120, 240: C_W240}
    for months, r in sens.items():
        fp = r["frontier_pts"]
        x = fp[:, 1] * np.sqrt(ppy) * 100
        y = fp[:, 0] * ppy * 100
        ax2.plot(x, y, color=cols.get(months, INK3), linewidth=2,
                 label=f"{months} months", zorder=4)
        ax2.annotate(f"{months}m", (x[-1], y[-1]),
                     textcoords="offset points", xytext=(6, 0),
                     fontsize=9, color=cols.get(months, INK3), weight="medium")

    ax2.set_title("Same optimiser, different estimation windows",
                  fontsize=12, color=INK, loc="left", pad=12, weight="semibold")
    ax2.set_xlabel("Annualised volatility (%)", fontsize=10, color=INK2)
    ax2.set_ylabel("Annualised return (%)", fontsize=10, color=INK2)
    leg2 = ax2.legend(frameon=False, fontsize=9, loc="lower right")
    for t in leg2.get_texts():
        t.set_color(INK2)

    fig.text(0.5, 0.005,
             "49 industry portfolios, long-only, Ledoit-Wolf shrinkage. "
             "The 36-month frontier sits far above the others because with "
             "36 months and 49 assets the inputs are fitted noise, not signal.",
             ha="center", fontsize=8.5, color=INK2)

    fig.tight_layout(rect=[0, 0.045, 1, 1])
    path = os.path.join(outdir, "efficient_frontier.png")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
def main(cfg=CONFIG):
    print("=" * 74)
    print("MEAN-VARIANCE OPTIMISER — 49 INDUSTRY PORTFOLIOS")
    print("=" * 74)

    data = load_data(cfg["DATA_FILE"])
    print(f"loaded {data['R'].shape[0]} months x {data['R'].shape[1]} "
          f"industries, {data['dates'][0]} -> {data['dates'][-1]}")
    print(f"missing cells recoded to NaN: {data['n_missing']}")
    print(f"settings: window={cfg['WINDOW_MONTHS']}m  "
          f"long_only={cfg['LONG_ONLY']}  cov={cfg['COV_ESTIMATOR']}")

    print("\n" + "-" * 74)
    print("PRIMARY RUN")
    print("-" * 74)
    res = run_window(data, cfg["WINDOW_MONTHS"], cfg["LONG_ONLY"],
                     cfg["COV_ESTIMATOR"], cfg["N_FRONTIER"],
                     cfg["PERIODS_PER_YEAR"])

    ann_rf = res["rf_m"] * cfg["PERIODS_PER_YEAR"]
    print(f"  risk-free (mean over window, annualised) = {100*ann_rf:.2f}%")

    w, summ, wpath, spath = write_tables(res, cfg["OUTDIR"],
                                         cfg["PERIODS_PER_YEAR"])

    print("\n  RESULTS")
    print("  " + summ.to_string(index=False, float_format=lambda v: f"{v:.2f}"
                                ).replace("\n", "\n  "))

    print("\n  TOP 10 WEIGHTS (%)")
    top = (w * 100).round(2).head(10)
    print("  " + top.to_string().replace("\n", "\n  "))

    # ---- sensitivity ---------------------------------------------------
    print("\n" + "-" * 74)
    print("WINDOW SENSITIVITY — same optimiser, different estimation windows")
    print("-" * 74)
    sens, rows = {}, []
    for m in cfg["SENSITIVITY_WINDOWS"]:
        if m > data["R"].shape[0]:
            continue
        r = run_window(data, m, cfg["LONG_ONLY"], cfg["COV_ESTIMATOR"],
                       cfg["N_FRONTIER"], cfg["PERIODS_PER_YEAR"],
                       verbose=False)
        sens[m] = r
        wn = take_window(data, m)
        Sraw = np.cov(wn["R"], rowvar=False, ddof=1)
        for key in ("min_variance", "tangency"):
            st = r["stats"][key]
            rows.append({
                "window_m": m,
                "portfolio": key,
                "ann_return_pct": 100 * st["ann_return"],
                "ann_vol_pct": 100 * st["ann_vol"],
                "sharpe": st["sharpe"],
                "n_holdings": int((r["ports"][key] > 1e-4).sum()),
                "raw_cov_rank": int(np.linalg.matrix_rank(Sraw)),
                "raw_cov_cond": float(np.linalg.cond(Sraw)),
            })
    sens_df = pd.DataFrame(rows)
    sens_path = os.path.join(cfg["OUTDIR"], "window_sensitivity.csv")
    sens_df.round(4).to_csv(sens_path, index=False)
    print(sens_df.to_string(
        index=False,
        formatters={"ann_return_pct": "{:.2f}".format,
                    "ann_vol_pct": "{:.2f}".format,
                    "sharpe": "{:.3f}".format,
                    "raw_cov_cond": "{:.3g}".format}))

    # ---- out-of-sample backtest ---------------------------------------
    bt_path = None
    if cfg.get("RUN_BACKTEST"):
        print("\n" + "-" * 74)
        print("ROLLING OUT-OF-SAMPLE BACKTEST — walk-forward, no look-ahead")
        print("-" * 74)
        bt = backtest(data, cfg["BACKTEST_WINDOWS"], cfg["LONG_ONLY"],
                      cfg["COV_ESTIMATOR"], cfg["PERIODS_PER_YEAR"])
        bt_path = os.path.join(cfg["OUTDIR"], "backtest.csv")
        bt.round(4).to_csv(bt_path, index=False)
        print(bt.to_string(
            index=False,
            formatters={"oos_ann_return_pct": "{:.2f}".format,
                        "oos_ann_vol_pct": "{:.2f}".format,
                        "oos_sharpe": "{:.3f}".format}))
        for W in sorted(set(bt["window_m"])):
            sub = bt[bt["window_m"] == W].set_index("portfolio")["oos_sharpe"]
            eq = sub.get("equal_weight_1overN", np.nan)
            beat = [p for p in ("min_variance", "tangency") if sub.get(p, -9) > eq]
            verdict = (", ".join(beat) + " beat 1/N") if beat else \
                      "NEITHER optimised portfolio beat naive 1/N"
            print(f"  {W}-month estimation window -> {verdict}")

    cpath = make_chart(res, sens, cfg["OUTDIR"], cfg)

    print("\n" + "-" * 74)
    print("FILES WRITTEN")
    print("-" * 74)
    for p in (wpath, spath, sens_path, bt_path, cpath):
        if p:
            print(" ", p)

    if res["diag"] and res["diag"]["singular"]:
        print("\n" + "!" * 74)
        print("WARNING: at this window the raw covariance matrix is singular")
        print(f"(N={res['diag']['N']} assets vs T={res['diag']['T']} months). "
              f"Shrinkage and the long-only")
        print("constraint keep the optimiser numerically stable, but the "
              "underlying inputs")
        print("are not identified. Treat the levels as indicative; prefer "
              "WINDOW_MONTHS=120.")
        print("!" * 74)

    return res, sens


if __name__ == "__main__":
    main()
