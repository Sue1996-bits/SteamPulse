# Related work and originality log

The guidelines require that our analysis is "not identical or similar to other publicly available analyses" of the same dataset. This log records what we searched, what overlaps and how SteamPulse differs. Add a row whenever you find something; don't delete rows.

Search status (2026-11-23): web search done for the dataset name and for complaint-category monitoring and forecasting on Steam. **Still open:** the Code tab of the Kaggle dataset page and Hugging Face "used by" links, which have to be checked by hand in a browser.

## Our claim, stated narrowly

SteamPulse forecasts the **share of each complaint category among a game's English negative reviews over the next seven days**. It flags a category as emerging with a guarded 7-day vs prior-28-day rule, and evaluates both:

1. whether a learned model beats persistence and rolling-28 baselines on identical, purged, chronological test rows, per category; and
2. whether categories flagged as emerging stay raised the following week, compared with rows that weren't flagged (`pipeline/alerts.py`).

Point-in-time safety (the strict unedited cohort, and cutoff-checked model versions) is part of the method. Using different tools or a seven-day horizon is not, on its own, a difference we claim.

## Log

| Date | Source | Data | Question and method | Overlap with us | Our difference |
|---|---|---|---|---|---|
| 2026-11-23 | Priambodo & Doewes, "Forecasting User Perception of Steam Game Reviews across Multiple Genres", JESSI 7(3), 2026. https://doi.org/10.59562/jessi.v7i3.13290 | English Steam reviews, 15 games, 12 months (own collection; corpus to confirm from the full text) | BERT sentiment aggregated into a weekly User Perception Score, BERTopic topics, short-horizon forecasting, topic attribution | Closest prior work: Steam reviews, about a year, forecasting, topics | They forecast one overall perception score per week; we forecast four fixed complaint-category shares among negatives, with volume guards, a purged split and persistence baselines, and test what follows an alert. Check their full text for dataset overlap |
| 2026-11-23 | Singh & O'Hagan, "Exploring Topic Modelling of User Reviews as a Monitoring Mechanism for Emergent Issues Within Social VR Communities", 2024. https://arxiv.org/abs/2406.03994 | Rec Room reviews from Steam, 2016-2024 | Topic modelling over time as a monitoring tool for emergent issues | Monitoring emergent issues from Steam reviews | One game and unsupervised topics, no forecast or evaluation against baselines. We don't claim issue monitoring itself as new |
| 2026-11-23 | Hu et al., "Analyzing Gamer Complaints in Reviews of Cross-Platform Video Games on Steam", IEEE CoG 2023. https://doi.org/10.1109/COG57401.2023.10333139 | Steam reviews of cross-platform games | Classifying reviews about platform-related issues (ML, about 80-90% accuracy) | Complaint classification on Steam reviews | Static classification of platform issues; no time series, alerts or forecasts |
| 2026-11-23 | Liu, Stanford CS191 project, Spring 2025. https://cs191.stanford.edu/projects/Spring2025/Willam___Liu_.pdf | 100,000+ Steam reviews | 10 issue categories (bugs, paywalls, ...) related to review score, Metacritic and revenue | Complaint taxonomy on Steam reviews | Cross-sectional link to success metrics; no daily shares, alerts or forecasting |
| 2026-11-23 | "Steam Review Miner — Reputation Drop Diagnostic" (Apify community tool). https://apify.com/muddy_snowshoe/steam-review-miner | Live Steam review API | Detects a statistically real drop in positive share, then classifies sampled negatives into complaint categories | Drop detection followed by complaint categories, which is close to our product idea | Commercial tool with no published evaluation. It flags drops in overall positivity; we model each category's share, forecast it, and measure forecast error and alert follow-through on held-out time. Cite it in the report as related practice |
| 2026-11-23 | GitHub `f0903/python-uge-4` (links to the kieranpoc dataset). https://awesome.ecosyste.ms/projects/github.com%2Ff0903%2Fpython-uge-4 | kieranpoc Steam reviews (our source) | Unknown; README only points at the dataset | Same dataset | To check: open the repository and record what it does |

## To do before the report

- [ ] Kaggle: open https://www.kaggle.com/datasets/kieranpoc/steam-reviews → Code, and log every notebook that does more than load the data (title, URL, method).
- [ ] Hugging Face: check https://huggingface.co/datasets/adde023/steam-reviews for linked models or Spaces.
- [ ] Read the full text of Priambodo & Doewes and record their data source, games and whether any overlap with ours.
- [ ] If close same-dataset work turns up, narrow our claim here and tell the instructor if the direction changes.
