# Labeling guide

How we hand-label complaint categories, and how the rules in `pipeline/labels.py` are developed and checked. The hand labels are the yardstick for the rules, so everyone labels the same way.

## Categories

A review can have several categories, or none. Label what the reviewer complains about, not what they mention.

| Category | Label it when the review complains about | Don't label it for |
|---|---|---|
| `crash` | Crashes, crash to desktop, freezes or hangs, blue screens, the game failing to launch or start | Low FPS alone (that's performance). If it both freezes and stutters, label both |
| `performance` | Low or dropping FPS, stutter, input lag, long loading, poor optimization, memory leaks, high CPU/GPU use | A hardware name on its own ("I have a 3080"). Network lag (that's match_server) |
| `match_server` | Matchmaking (including SBMM and unfair teams), queue times, disconnects, server quality, lag, ping, desync, rubberbanding, hit registration | Frame-rate problems. Cheaters alone are not a server complaint |
| `monetization` | Microtransactions, loot boxes, pay to win, battle pass, store prices, greed or cash grabs, the game not being worth its price | A neutral price mention ("got it on sale for $30") or "don't waste your money" with no monetization reason |

Not a category: bugs in general, cheaters, balance, toxicity, story, anti-cheat bans. A review about only these gets no label and counts as unclassified.

When it's unclear whether "lag" means frames or network, use the context (FPS, stutter, GPU → performance; ping, servers, rubberbanding → match_server). With no context, label `match_server`: these are online games and the rules do the same.

## Filling in the sample

`make label-sample` writes `data/samples/label_sample_<seed>.csv`. For a five-game sample:

```bash
make label-sample APPS="1172470 1517290 1811260 1085660 359550"
```

Each annotator works on their own copy, named `label_sample_226_<name>.csv`. The file holds review text, so share it outside Git (it lives under the ignored `data/` folder).

- Put `1` in each category column that applies. Leave the others empty (or `0`).
- Write anything unclear in `notes`, starting with `?` (for example `? "lag" with no context`). Still give your best label.
- Don't read the `period` column while labeling, and don't look up what the rules output.

## Workflow

1. **Split.** Rows 1-100, 101-200 and 201-300 of the `train` period go to one person each. Rows 1-30 of each block are also labeled by a second person (90 reviews, 30%).
2. **Agreement.** `python -m pipeline.label_eval agree FILE_A FILE_B` reports agreement and Cohen's kappa per category, and writes the disagreements to adjudicate.
3. **Adjudicate.** Go through the disagreements together and agree on one label. Add the hard cases to the examples below.
4. **Gold file.** Merge everything into `data/samples/label_gold_226.csv`, then `python -m pipeline.label_eval export data/samples/label_gold_226.csv` writes an ID-and-label-only copy to `labels/` to commit.
5. **Rules.** Edit `pipeline/labels.py` against the `train` reviews only (`python -m pipeline.label_eval score ... --period train`). Bump `LABEL_VERSION` each time the rules change.
6. **Freeze, then check.** When the rules are frozen, label the 150 `later` reviews and score them once with `--period later`. Don't change the rules after seeing those numbers. If you must, bump the version and say so in the report.

The `later` scores on the random sample are the population estimates for the report. If we add a keyword-enriched sample to get more positives for a rare category, its metrics are reported separately and not as population estimates.

## Examples

Agreed decisions on hard cases go here, with the review ID.

| Review | Decision | Why |
|---|---|---|
| "never crashed once but the servers are awful" | `match_server` | Negated crash |
| "not only does it crash, it runs at 20 fps" | `crash`, `performance` | "not only" isn't a negation |
| "I have a 4090 and it still stutters" | `performance` | The stutter is the complaint, not the GPU |
| "can't stop crashing" | `crash` | "can't stop" isn't a negation |
