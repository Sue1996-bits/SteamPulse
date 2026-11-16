# Scale log

Exclude irrelevant binaries and duplicate runs from scale totals.

| Date | Stage | Source files | Relevant columns | Compressed bytes | Measured uncompressed bytes | Rows read | Unique rows | Filters | Partitions | Runtime | Output size | Machine / config | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-16 | profile | 2 part files | 8 of 25 | 51.4 MB | 87.4 MB | 384,128 | 384,128 | none | 201 files | 8.8s | 0.9 MB | local[*], driver 8g | benchmark (2/663 files); 476 games; all columns 0.10 GB uncompressed; snapshot d0ec12b |
| 2026-11-16 | profile | 663 part files | 8 of 25 | 19.88 GB | 29.98 GB | 113,883,717 | 113,883,717 | none | 201 files | 172.2s | 0.10 GB | local[*], driver 8g | full snapshot; 105,893 games; all columns 34.87 GB uncompressed; snapshot d0ec12b |
