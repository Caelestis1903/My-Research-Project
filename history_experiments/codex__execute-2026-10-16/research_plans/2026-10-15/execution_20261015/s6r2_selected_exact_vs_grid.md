# S6-R2 exact versus grid

R0 historical selected endpoints are stored in the per-dataset `s6r2_partA_{val,test}_metrics_*.csv`. This audit recomputes DH−F from those stored values (L2); it cannot independently bind them to absent original S6 NPZ/checkpoints.

R1–R4 validation epoch selection and their available exact-test lookups are reported only where the historical `test_available` flag is true. Separate A-grid files compare a different candidate set (9/19/29/39/49); same-direction grid results cannot fill missing exact selected epochs. The F/DH dense checkpoint run B1 is not done.

All four seeds and six cells, per-rule epoch and availability, are retained in `s6r2_paired_recomputed.csv` and the original source CSVs indexed in `artifact_inventory.json`. The original F/DH prediction-level CI, time-window analysis and common-epoch/timing reconstruction remain INCOMPLETE, not zero effects.
