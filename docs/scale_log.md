# Scale log

Each `make profile` run adds a row here automatically. For test reruns, use `make profile NOLOG=1` so the same run doesn't get counted twice.

The pipeline steps are logged only when asked: `make pipeline LOG=1` adds one row per step (clean, daily, and features and baselines once labels exist). Steps run in one Spark session, so the clean row includes Spark startup.

| Date | Stage | Source files | Relevant columns | Compressed bytes | Measured uncompressed bytes | Rows read | Unique rows | Filters | Partitions | Runtime | Output size | Machine / config | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-16 | profile | 2 part files | 8 of 25 | 51.4 MB | 87.4 MB | 384,128 | 384,128 | none | 201 files | 8.8s | 0.9 MB | local[*], driver 8g | benchmark (2/663 files); 476 games; all columns 0.10 GB uncompressed; snapshot d0ec12b |
| 2026-11-16 | profile | 663 part files | 8 of 25 | 19.88 GB | 29.98 GB | 113,883,717 | 113,883,717 | none | 201 files | 172.2s | 0.10 GB | local[*], driver 8g | full snapshot; 105,893 games; all columns 34.87 GB uncompressed; snapshot d0ec12b |
| 2026-11-21 | clean | 663 part files | 8 of 25 | 19.88 GB | 29.98 GB | 113,883,717 | 573,962 | 5 games, window, unedited, valid rows | 5 app_id, 45 files | 13.4s | 0.10 GB | local[8], driver 8g | includes Spark startup; quarantine in data/quarantine/; snapshot d0ec12b |
| 2026-11-21 | daily | reviews_clean | all | 0.10 GB | - | 573,962 | 1,825 | none | 5 app_id, 5 files | 1.6s | 20 KB | local[8], driver 8g | daily_negative; snapshot d0ec12b |
