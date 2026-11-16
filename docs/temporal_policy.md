# Temporal policy

- **Day boundaries:** UTC (`spark.sql.session.timeZone=UTC`).
- **Strict cohort (`strict_unedited`):** only reviews with `timestamp_created == timestamp_updated`, applied to both category counts and denominators.
- **Exclusions per game:** TODO (share excluded as edited, and the resulting selection bias).
- **Coverage rule:** TODO (how suspected source gaps are detected; gaps are null via `coverage_status`, never zero).
- **Game selection:** judged on the training period only.
- **Fallback:** if the strict cohort is too sparse, a latest-version availability analysis using update time. This is not historical submission-volume reconstruction.
