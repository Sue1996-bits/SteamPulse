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

`make label-sample` writes `data/samples/label_sample_<seed>.csv` with 300 `train` and 150 `later` reviews. For a five-game sample:

```bash
make label-sample APPS="1172470 1517290 1811260 1085660 359550"
python -m pipeline.label_eval split data/samples/label_sample_226.csv --people yifan isabella yunseo
```

`split` writes one file per annotator, `label_sample_226_<name>.csv`, with 180 reviews: 100 `train` and 50 `later` reviews that are yours (`role` = `primary`), plus 30 `train` reviews from the next person's block (`role` = `second`) for agreement. Every sample review is primary for exactly one person. The `period` column is left out on purpose. `split` never overwrites existing files, since they may already hold labels: use `--out-dir` for a new split. The files hold review text, so share them outside Git (they live under the ignored `data/` folder).

- Fill **all four** category cells of every row: `1` if the category applies, `0` if it doesn't. A finished row is then distinguishable from one you only started.
- A row with some cells blank is **incomplete** and a row with none is **blank**. Neither is counted (and never as "no category"); the scripts list their review IDs, and `gold` refuses to build until every primary row is complete.
- Label every row in your file, `second` rows included.
- Write anything unclear in `notes`, starting with `?` (for example `? "lag" with no context`). Still give your best label.
- Don't look up what the rules output.

## Workflow

1. **Split.** `label_eval split`, as above. Each person labels their own file.
2. **Agreement.** `python -m pipeline.label_eval agree FILE_A FILE_B` for each pair of files reports agreement and Cohen's kappa per category on the shared reviews, and writes the disagreements to adjudicate.
3. **Adjudicate.** Go through the disagreements together and agree on one label. The person whose row is `primary` updates their file. Add the hard cases to the examples below.
4. **Gold file.** `python -m pipeline.label_eval gold data/samples/label_sample_226.csv data/samples/label_sample_226_*.csv` takes everyone's `primary` rows, restores `period` from the sample and writes `data/samples/label_gold_226.csv`. It stops if a primary row isn't complete. `--partial` builds a preview from what's done into its own file, `label_gold_226_partial.csv`, and never touches the official one. An existing output is only replaced with `--force` (for example, rebuilding after adjudication); the sample and the annotators' files are never overwritten. Then `python -m pipeline.label_eval export data/samples/label_gold_226.csv` writes an ID-and-label-only copy to `labels/` to commit.
5. **Rules.** Edit `pipeline/labels.py` against the `train` reviews only (`make label-eval GOLD=data/samples/label_gold_226.csv`). Bump `LABEL_VERSION` each time the rules change. `score` refuses a file whose `period` is missing or not `train`/`later`, so the `later` reviews can't slip into development by accident.
6. **Freeze, then check.** When the rules are frozen, score the 150 `later` reviews once (`make label-eval GOLD=... PERIOD=later`). Don't change the rules after seeing those numbers. If you must, bump the version and say so in the report.

The `later` scores on the random sample are the population estimates for the report. If we add a keyword-enriched sample to get more positives for a rare category, its metrics are reported separately and not as population estimates.

## Examples

Agreed decisions on hard cases go here, with the review ID.

| Review | Decision | Why |
|---|---|---|
| "never crashed once but the servers are awful" | `match_server` | Negated crash |
| "not only does it crash, it runs at 20 fps" | `crash`, `performance` | "not only" isn't a negation |
| "I have a 4090 and it still stutters" | `performance` | The stutter is the complaint, not the GPU |
| "can't stop crashing" | `crash` | "can't stop" isn't a negation |
