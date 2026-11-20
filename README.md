# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/mytlyy0745/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/mytlyy0745/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for each of GTA VI's 55 PlayStation 5 trophies, the earn rate on Exophase at 2026-11-26T00:00:00Z — five quantiles (5%, 25%, 50%, 75%, 95%) of the share of tracked players holding the trophy a week after launch. It reads one case (`/input/<id>.in`: every Exophase read the house made before the checkpoint) and writes `/output/<id>.out`. Python 3.13, standard library only, no network, deterministic.

## Method

Adapted from the proven `gta6-twitch-ccu-forecast-01` entry (anchor on the last known print, forecast in log space, spread floor widening with the square root of the horizon, persistence fallback on any error):

1. **Refresh epochs, not reads.** Exophase refreshes its counts on its own schedule — between refreshes every read is identical. The history is collapsed to its distinct snapshots; a frozen stretch carries no more information than one read.
2. **Recent-window logit trend.** Per trophy, least squares of logit(rate) against ln(1 + hours since launch), fitted over the last 24 hours of snapshots only. The launch ramp (first hours, rates near 0.01%) would poison a whole-history fit with a slope that explodes on extrapolation. Slopes are shrunk toward their group's slope (tier × story-vs-grind, from the trophy description) with pseudo-count 3 — rare trophies' printed 2-decimal rates are mostly rounding noise.
3. **Damped extrapolation.** The change beyond the last snapshot is multiplied by 1 / (1 + 0.5 · Δu): earners growth saturates as rates approach their ceiling.
4. **Earners-over-players decomposition (25% weight).** Log earners and log tracked players are fitted the same windowed way; their ratio is the rate. This is what carries dilution: when tracked players grow faster than earners, rates fall even while more people earn the trophy.
5. **Honest spread.** Logit sigma = sqrt((0.30 · sqrt(days))² + resid²), floored at 0.40, capped at 1.30. When the history holds fewer than three distinct snapshots no refresh has ever been seen, so quantiles come from a two-component mixture: with probability 1 − exp(−horizon/4h) a refresh lands before the target and shifts rates down ~1.1 logit (launch night's refresh took tracked players 638 → 3,868 while earners grew only ~2×, a rate factor of ~1/3), otherwise rates stay near the last read.

## Assumptions (grounded in public trophy data)

- **Base rates from GTA V / RDR2 patterns:** story-progression trophies (chapter completions, final missions) saturate high and fast — most players who start the story finish the early chapters within days — while grind/collection trophies (250 stunts, gold medals everywhere, 100% completion, platinum) stay rare and grow slowly. The model's story-vs-grind grouping and tier shrinkage encode exactly that split.
- **Story gating in week one:** almost nobody reaches the platinum (all other trophies) or 100% completion in the first week; late-chapter trophies grow through the week as the player base works through the story, they do not jump on day one.
- **Launch scale:** GTA VI is the biggest console launch on record; Exophase's tracked-player count grows steeply for days as players link accounts, so dilution (new tracked players at zero progress) is the dominant force pulling early rates down at each refresh, partly offset by earners growth.
- **Refresh risk:** the target read shows the page as of Exophase's last refresh before it, so the forecast's spread must cover refresh timing, not just trend error.

## Validation

On synthetic final-like cases (launch ramp + saturating progression + refresh freezes, cut day 2, target day 7, two regimes: fast and slow progression): mean pinball loss on the logit scale **0.21–0.25**, against 0.46–0.63 for the persistence starter and 0.26–0.31 for the undamped whole-history trend baseline. Synthetic cases are a check of the machinery, not a promise — thousands of entries compete and only the lowest final losses place.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=89k0z65b&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
