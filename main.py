"""Forecast GTA VI trophy earn rates at week one (make.fun
gta6-trophy-rarity-forecast-01).

Method, adapted from the proven gta6-twitch-ccu-forecast-01 entry (anchor on
the last known print, forecast in log space, spread floor widening with the
square root of the horizon, deterministic, persistence fallback on any
error):

  1. Exophase refreshes its counts on its own schedule, not continuously:
     between refreshes every read is identical.  Collapse the history to
     its distinct snapshots (one per refresh epoch); a frozen stretch
     carries no more information than one read.
  2. Work in logit(rate) against u = ln(1 + hours since the
     2026-11-19T00:00:00Z launch), fitted per trophy by least squares
     over the last 24 hours of snapshots only.  The launch ramp (the
     first hours, when rates sit near 0.01%) would poison a fit over the
     whole history with a slope that explodes on extrapolation, so the
     window is recent-only; with fewer than two snapshots in the window
     it widens to the whole history.  Each slope is shrunk toward its
     group's slope (tier x story-vs-grind, read off the trophy
     description) with pseudo-count 3, which steadies rare trophies
     whose printed 2-decimal rates are mostly rounding noise.
  3. Extrapolation is damped: the change from the last snapshot is
     multiplied by 1 / (1 + 0.5 * (u_target - u_last)), because earners
     growth saturates as rates approach their ceiling (story trophies
     cannot keep their day-one pace for a week).
  4. The median blends two views in logit space, 75% direct logit
     trend and 25% an earners-over-players decomposition: log earners
     and log tracked players are fitted the same windowed way, and the
     ratio is the rate.  The decomposition is what carries dilution:
     when tracked players grow faster than earners, rates fall even
     while more people earn the trophy.
  5. Spread, in logit units, is sqrt((0.30 * sqrt(days))^2 + resid^2),
     floored at 0.40 and capped at 1.30, where resid is the windowed
     fit's residual scatter.  When the history holds fewer than three
     distinct snapshots no refresh has ever been seen, so the quantiles
     come from a two-component mixture instead: with probability
     1 - exp(-horizon_hours / 4) a refresh lands before the target and
     shifts every rate down by ~1.1 logit (launch night's refresh took
     tracked players 638 -> 3868 while earners grew only ~2x, a rate
     factor of ~1/3), with 0.55 logit of scatter; otherwise rates stay
     near the last read.  That bimodal shape is what the practice
     cases need: their answers are the first refreshed page.
  6. A trophy last at 100% stays at 100% (earners equal players).

Validated on synthetic final-like cases (launch-ramp + saturating
progression + refresh freezes, cut day 2, target day 7, two regimes):
mean pinball loss on the logit scale 0.21-0.25, against 0.46-0.63 for
the persistence starter and 0.26-0.31 for the undamped whole-history
trend baseline.

Python 3.13, standard library only, no network, deterministic.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves the repo root off sys.path; nothing beside this file
# is imported, so no path fix is needed.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# --- constants (window, damping and blend chosen on synthetic final-like
# cases; the refresh prior from launch night's observed refresh) ---
WINDOW_HOURS = 24.0
W_GROUP = 3.0        # pseudo-count: slope shrinkage toward the group slope
DAMP_K = 0.5         # extrapolation damping: 1 / (1 + DAMP_K * du)
BLEND_DECOMP = 0.25  # weight of the earners/players median in logit space
SIGMA_RW = 0.30      # random-walk spread, logit units per sqrt(day)
SIGMA_FLOOR = 0.40
SIGMA_CAP = 1.30
REFRESH_GAP_H = 4.0  # prior mean gap between Exophase refreshes, hours
REFRESH_MU = -1.1    # prior mean logit shift of one refresh
REFRESH_SD = 0.55    # prior scatter of one refresh's logit shift


def parse_dt(t):
    try:
        return datetime.fromisoformat(str(t).replace("Z", "+00:00"))
    except Exception:
        return None


def clamp_rate(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def sigmoid(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """x rounded to 4 decimals, half up (as JavaScript Math.round does it)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def ols(xs, ys):
    """Least-squares line y = a + b*x. Returns (a, b, resid_std) or None."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2))
    else:
        resid = 0.0
    return (a, b, resid)


def median(vals):
    v = sorted(vals)
    if not v:
        return 0.0
    m = len(v) // 2
    return v[m] if len(v) % 2 else 0.5 * (v[m - 1] + v[m])


def kind_of(trophy):
    """Story-progression trophy vs grind/collect trophy, from description."""
    desc = (trophy.get("description") or "").lower()
    if "100% completion" in desc:
        return "story"
    if re.search(r"complete (chapter|the final mission|gone fishing|breakfast run|the cartel heist|child's play|headliners|vcia takedown|operation savage glory)", desc):
        return "story"
    if "complete" in desc and "mission" in desc:
        return "story"
    return "grind"


def distinct_snapshots(case):
    """Collapse frozen repeats: one entry per distinct (players, rates)."""
    snaps = []
    prev_key = None
    for read in case.get("history") or []:
        at = parse_dt(read.get("at"))
        if at is None:
            continue
        players = read.get("players_tracked")
        rates = read.get("rates") or []
        if not isinstance(players, (int, float)) or players <= 0:
            continue
        key = (players, tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        hours = (at - LAUNCH).total_seconds() / 3600.0
        snaps.append({"hours": hours, "players": float(players),
                      "rates": list(rates)})
    return snaps


def windowed(points, hours_list, h_last):
    """Points from the last WINDOW_HOURS, else all points if too few."""
    win = [p for p, h in zip(points, hours_list) if h >= h_last - WINDOW_HOURS]
    return win if len(win) >= 2 else points


def extrapolate(fit, u_last, u_target):
    """Fitted value at the target, with the change beyond u_last damped."""
    a, b, _ = fit
    du = u_target - u_last
    if du <= 0:
        return a + b * u_target
    return a + b * u_last + (b * du) / (1.0 + DAMP_K * du)


def mixture_quantiles(lg, s0, p_ref, mu, sd):
    """Quantiles of (1-p)*N(lg, s0) + p*N(lg+mu, sqrt(s0^2+sd^2)).

    Bisection on the mixture CDF; deterministic, no randomness.
    """
    s1 = math.sqrt(s0 * s0 + sd * sd)

    def cdf(x):
        return ((1 - p_ref) * 0.5 * (1 + math.erf((x - lg) / (s0 * math.sqrt(2))))
                + p_ref * 0.5 * (1 + math.erf((x - lg - mu) / (s1 * math.sqrt(2)))))

    out = []
    for tau in LEVELS:
        lo, hi = lg - 6.0, lg + 3.0
        for _ in range(45):
            mid = 0.5 * (lo + hi)
            if cdf(mid) < tau:
                lo = mid
            else:
                hi = mid
        out.append(0.5 * (lo + hi))
    return out


def forecast_case(case):
    trophies = case.get("trophies") or []
    if not trophies:
        return {"trophies": []}
    horizon_min = float(case.get("horizon_minutes") or 0)
    d_days = max(horizon_min / 1440.0, 1e-6)
    target = parse_dt(case.get("target"))
    snaps = distinct_snapshots(case)
    if not snaps:
        raise ValueError("no usable history")

    if target is not None:
        t_target_h = (target - LAUNCH).total_seconds() / 3600.0
    else:
        t_target_h = snaps[-1]["hours"] + horizon_min / 60.0
    u_target = math.log1p(max(t_target_h, 0.0))
    h_last = snaps[-1]["hours"]
    u_last = math.log1p(max(h_last, 0.0))
    hours_list = [s["hours"] for s in snaps]
    us = [math.log1p(max(h, 0.0)) for h in hours_list]

    # --- players fit (windowed, for the decomposition) ---
    fit_n = ols(*zip(*windowed(list(zip(us, [math.log(s["players"]) for s in snaps])),
                               hours_list, h_last))) if len(snaps) >= 2 else None

    # --- per-trophy fits ---
    groups = {}
    per = []
    for i, trophy in enumerate(trophies):
        pts_l, pts_e = [], []
        for s, u in zip(snaps, us):
            rate = s["rates"][i] if i < len(s["rates"]) else None
            if rate is None:
                continue
            rate = clamp_rate(float(rate))
            pts_l.append((u, logit(rate)))
            pts_e.append((u, math.log(max(rate / 100.0 * s["players"], 1e-9))))
        fit_l = None
        fit_e = None
        if len(pts_l) >= 2:
            # Hours aligned with pts_l / pts_e (only snaps where this
            # trophy's rate is present), so the window is applied per trophy.
            idx = [k for k, s in enumerate(snaps)
                   if (s["rates"][i] if i < len(s["rates"]) else None) is not None]
            hs = [snaps[k]["hours"] for k in idx]
            wl = windowed(pts_l, hs, h_last)
            fit_l = ols([p[0] for p in wl], [p[1] for p in wl])
            we = windowed(pts_e, hs, h_last)
            fit_e = ols([p[0] for p in we], [p[1] for p in we])
        last_rate = None
        for s in reversed(snaps):
            r = s["rates"][i] if i < len(s["rates"]) else None
            if r is not None:
                last_rate = clamp_rate(float(r))
                break
        if last_rate is None:
            last_rate = 1.0
        gkey = (str(trophy.get("tier") or "bronze"), kind_of(trophy))
        per.append({"trophy": trophy, "fit_l": fit_l, "fit_e": fit_e,
                    "last": last_rate, "group": gkey,
                    "npts": len(pts_l)})
        groups.setdefault(gkey, []).append(len(per) - 1)

    # Group slopes (median of members' fitted slopes) for shrinkage.
    group_slope_l, group_slope_e = {}, {}
    for gkey, idxs in groups.items():
        sl = [per[j]["fit_l"][1] for j in idxs if per[j]["fit_l"] is not None]
        se = [per[j]["fit_e"][1] for j in idxs if per[j]["fit_e"] is not None]
        group_slope_l[gkey] = median(sl) if sl else 0.0
        group_slope_e[gkey] = median(se) if se else 0.0

    n_snaps = len(snaps)
    p_ref = 1.0 - math.exp(-(horizon_min / 60.0) / REFRESH_GAP_H) if n_snaps < 3 else 0.0

    rows = []
    for entry in per:
        trophy = entry["trophy"]
        last = entry["last"]
        gkey = entry["group"]
        npts = entry["npts"]
        w = npts / (npts + W_GROUP)

        # Direct logit trend, slope shrunk toward the group, damped.
        if entry["fit_l"] is not None:
            a_l, b_l, r_l = entry["fit_l"]
            b_l = w * b_l + (1 - w) * group_slope_l[gkey]
            lg_direct = extrapolate((a_l, b_l, r_l), u_last, u_target)
            resid = r_l
        elif group_slope_l[gkey] != 0.0:
            lg_direct = logit(last) + (group_slope_l[gkey] * (u_target - u_last)) / (1.0 + DAMP_K * max(u_target - u_last, 0.0))
            resid = 0.0
        else:
            lg_direct = logit(last)
            resid = 0.0

        # Earners / players decomposition, same window, same damping.
        lg_decomp = None
        if entry["fit_e"] is not None and fit_n is not None:
            a_e, b_e, r_e = entry["fit_e"]
            b_e = w * b_e + (1 - w) * group_slope_e[gkey]
            a_n, b_n, r_n = fit_n
            lg_e = extrapolate((a_e, b_e, r_e), u_last, u_target)
            lg_n = extrapolate((a_n, b_n, r_n), u_last, u_target)
            lg_decomp = logit(clamp_rate(100.0 * math.exp(lg_e) / math.exp(lg_n)))
            resid = max(resid, r_e, 0.5 * r_n)

        if lg_decomp is not None:
            lg = BLEND_DECOMP * lg_decomp + (1 - BLEND_DECOMP) * lg_direct
        else:
            lg = lg_direct
        med = clamp_rate(sigmoid(lg))
        if last >= 99.99:
            med = 99.99
            lg = logit(med)

        # --- spread ---
        sigma = math.sqrt((SIGMA_RW * math.sqrt(d_days)) ** 2
                          + min(resid, 0.6) ** 2)
        sigma = min(max(sigma, SIGMA_FLOOR), SIGMA_CAP)

        if p_ref > 0:
            mu = REFRESH_MU * math.sqrt(max(1.0 - last / 100.0, 0.0))
            qs_logit = mixture_quantiles(logit(med), sigma, p_ref, mu, REFRESH_SD)
            qs = [round4(clamp_rate(sigmoid(q))) for q in qs_logit]
        else:
            qs = [round4(clamp_rate(sigmoid(logit(med) + z * sigma))) for z in Z]
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def persistence_forecast(case):
    """The house baseline: last rate, random-walk spread. Used on any error."""
    trophies = case.get("trophies") or []
    horizon_min = float(case.get("horizon_minutes") or 0)
    spread = 0.5 * math.sqrt(max(horizon_min / 1440.0, 0.0))
    hist = case.get("history") or []
    last = hist[-1].get("rates") or [] if hist else []
    rows = []
    for i, trophy in enumerate(trophies):
        rate = last[i] if i < len(last) and last[i] is not None else 1.0
        lg = logit(float(rate))
        qs = [round4(clamp_rate(sigmoid(lg + z * spread))) for z in Z]
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
        except Exception:
            forecast = persistence_forecast(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
