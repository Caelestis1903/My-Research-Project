# C05 raw archive: classification complete; pack paused on capacity

The three known incomplete numeric telemetry streams were classified losslessly. Their original compressed bytes remain in same-directory `.incomplete_bytes` files and in the archived typed byte-provenance JSON. All three archived hex payloads decode to the original lengths and SHA-256 hashes. Verified complete prefixes contain **1, 15, and 1,385** JSONL records. Gzip EOF remains false; missing extent and PID 2982386 costs remain unknown/null.

The actual post-classification engineering validator and standalone archive precheck passed. The existing frozen pack CLI was invoked **exactly once**, foreground session **33899**, PID **3060800**, and exited **1** at `tables.py:476` after full archive write/content readback and before part/manifest publication.

- Preserved archive: `/tmp/phase0-3-c05-01a1e9a9/publication/RAW_RUNS.tar.gz`
- SHA-256: `fa08b88dd6d488b48f1810a4e900e680ba3a35e577d2118f6b68accf92e44250`
- Archive bytes: **677,004,657**
- Members: **4,026**; categories: `{"control": 1530, "design": 7, "raw": 2381, "receipt": 53, "source": 55}`
- Required workspace headroom (`3A + 256 MiB + 1 GiB`): **3,373,191,251 bytes**
- Measured workspace free after failure: **3,091,951,616 bytes**
- Current shortfall: **281,239,635 bytes**
- Parts/manifests published: **0 / none**

The archive includes both frozen final-evidence check receipts and the earlier exact review-text wrapper. The existing writer verified every archived member's name/order/type/length/content hash; this bounded readback additionally checked headers and the three archived compressed-byte roundtrips. No scientific count recomputation, export, witness recollection, retry, cleanup, source/guard change, model get/release, or remote action occurred. Test stayed sealed (`test_accessed=false`). The parent continues supervising the same RAM owner, PID 2926581/session 16855.

Current status is **PAUSED_CAPACITY_BLOCKER**, not successful part publication. See `RAW_ARCHIVE_COMPLETION.json` for exact measurements and aliases. This worker permanently returns.
