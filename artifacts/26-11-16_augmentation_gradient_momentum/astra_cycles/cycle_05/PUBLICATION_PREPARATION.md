# C05 publication preparation — NOT_FINAL

This is a bounded preparation handoff from native child `01a1ea56-f9b0-7410-919e-72b0aa904fcb` to parent `01a1dfb8-605b-7430-8462-5fd3da0b048f`. A fresh child must execute publication after the parent completes the required gates. This worker returns permanently. There are no remote mutations, local HEAD/index updates, raw packing, scientific analysis, owner IPC or Test actions in this preparation.

W is `/home/xzf/Code/phase0-3_tvg_rm_20261116`. The target is `Caelestis1903/My-Research-Project`, `main`, expected C04 head `cbccf5ae234e46bc3f976467de2b3327139f021b`, base tree `e874ff09381557117d9e357048e3b59a475bd366`. The physical numeric root R5 is `/tmp/phase0-3-c05-01a1e9a9/attempt1`; logical N5 is `artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_05/attempt1` relative to W. Later archive staging S5 is `/tmp/phase0-3-c05-01a1e9a9/publication`.

The source receipt identifies **55 files**, with SHA-256 `f716b4e3025621f20d10aae4547379b9adafa69f2824af769f4e01aaef1fe59d`. The prompt's `55f716...` joins the count and hash; it is not a 64-character hash. All 55 members are verified without importing research modules or changing source. Seven C05 files are published: `__init__.py`, `core.py`, `checks.py`, `protocol.py`, `runner.py`, `tables.py`, `README.md`.

The exact C04 `PUBLICATION_RECEIPT.json`, SHA-256 `319b921da73f3b5a157f45922bfdafb475673f77f987b4fc409525a958cc12cd`, **must be in the C05 commit**. C04's 89 raw parts, about 740 MB, are already published: retain their existing tree entries, never upload them again. Preserve the C04 lost-selected-RAM disclosure and its distinction between acceptance-time and current availability. The helper includes only the C04 receipt as a new C04 publication path.

## Offline helper and inventory

Run only with project Python, from W:

```bash
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python research_reports/cell_mass_bce_20261117/PREPARE_PUBLICATION.py \
  --expected-head cbccf5ae234e46bc3f976467de2b3327139f021b \
  --base-tree e874ff09381557117d9e357048e3b59a475bd366
```

Outputs are `/tmp/phase0-3-cycle05-publication-manifest.json`, `/tmp/phase0-3-cycle05-tree-plan.json`, plus a uniquely named temporary directory containing `original.index` and `prepared.index`. These are **NOT_FINAL** snapshots of current files. Rerun only after later files are ready; never treat today's tree SHA as the final tree. Keep the public preparation records static while preparing a snapshot to avoid self-referential hashes. The manifest explicitly says `publication_authorized=false`; it supplies no final-release authorization.

The helper copies the **existing index**, checks its tree equals the pinned C04 tree, and updates only actual allowlisted files in the copy. `git write-tree --missing-ok` retains thin/missing unrelated objects. Do not use `git read-tree HEAD`, recursive traversal of unrelated thin trees, or a network fetch to reconstruct the index. Git commands disable lazy fetching. Local blob/tree writes are expected; the real index and detached HEAD must remain byte-identical during preparation.

Public inventory consists of direct `.py/.md/.json/.csv/.png` files in the C05 report, planning and cycle artifact directories; exactly the seven C05 source files; all eight Markdown files under `research_context/experiments`; `AGENTS.md`, `buchong.md`, `EXP_AI_CONTEXT.md`, `MDMT_ALGORITHM_METRICS_LEDGER.md`, the series `00_EXECUTION_RUNBOOK.md`; and the C04 receipt. The C05 receipt is excluded until remote publication is verified. Each entry records SHA-256, Git blob SHA-1, bytes, original mode and whether a new upload is needed. Unchanged blobs already present in the base index need no upload.

The helper does not recursively publish R5 or N5. The future package contract is N5 `RAW_RUNS.tar.gz.part-NN` (8 MiB each, final part may be shorter), `RAW_RUNS_MANIFEST.json`, and `RAW_RUNS_INDEX.json`. Parts' paths in the manifest are basenames, in concatenation order; each has `path`, `bytes`, `sha256`. Manifest fields include `source_hash`, archive `bytes` and archive `sha256`; the index maps full archive membership and provenance. If the later packer uses a different index name, the fresh worker must explicitly reconcile that helper contract before preparing the final snapshot. Missing package files are recorded today, not invented.

Later pack R5 numeric/source/review evidence into the S5 archive and write the N5 parts/manifest/index after training and final evidence are ready. Include the distinct resource reviewer's R5 engineering signoff as archive evidence; do not edit it. Exclude `guardian.json` and `STORAGE_OWNER.json` at every archive depth, owner credentials, model/checkpoint/tensor files, and foreign files. Do not duplicate raw R5 files as individual Git blobs. Reused C04 records should reference their already-published archive and exact hashes rather than copying its raw parts into C05. Parent owns the policy for the final evidence package and claims.

## Full tree creation plan

For every changed path, the helper finds its parents and runs `git ls-tree -z` on each **immediate directory** in the prepared tree, deepest first, root last. Each plan item has `path` (root is `""`), expected `sha`, and complete `entries` with basename `path`, `mode`, `type`, `sha`. Every existing entry is included alongside additions. The helper independently hashes Git's binary tree serialization and requires equality. Ancestors use the exact child SHAs. Missing unrelated descendants need not be read to preserve their references.

C04's single `create_tree` request with a base tree plus 266 nested updates timed out / returned 422. Bottom-up full immediate-directory calls with **no base tree** worked in 18 calls. Use that method for C05; do not flatten paths or send only new entries in a no-base tree. Require every returned tree SHA to equal the plan and the last root SHA to equal the manifest's `expected_tree`.

## Connected API discovery and schemas

Discover current tool descriptions first; do not assume old schemas. No GitHub call was made by this preparation worker. In `functions.exec`:

```js
for (const suffix of ["create_blob", "create_tree", "create_commit", "update_ref", "fetch_commit", "search_branches"]) {
  text(ALL_TOOLS.find(t => t.name === "mcp__codex_apps__github_" + suffix));
}
```

The available declarations observed during preparation are:

| Tool suffix (prefix `mcp__codex_apps__github_`) | Arguments |
| --- | --- |
| `create_blob` | `{repository_full_name, content, encoding?: "utf-8" or "base64"}` |
| `create_tree` | `{repository_full_name, tree_elements: Array<object>, base_tree_sha?: string or null}` |
| `create_commit` | `{repository_full_name, tree_sha, parent_sha, message, additional_parent_shas?: string[] or null}` |
| `update_ref` | `{repository_full_name, branch_name, sha, expected_sha?: string or null, force?: boolean}` |
| `fetch_commit` | `{repo_full_name, commit_sha}` |
| `search_branches` | `{owner, repo_name, query, page_size?: number, cursor?: string or null}` |
| `fetch_file` | `{repository_full_name, path, ref?: string, encoding?: "utf-8" or "base64", start_line?: number, end_line?: number}` |
| `fetch_blob` | `{repository_full_name, blob_sha}` |

Results are MCP envelopes: inspect `structuredContent` and/or parse text `content` for each actual response. Reject `isError`, missing metadata or a mismatched SHA; do not confuse a tool receipt with GitHub success. Store compact verified SHA-to-path upload receipts for reuse. For `create_tree`, pass the full plan item's `entries` as `tree_elements`, omitting `base_tree_sha`. Supply no embedded file content to trees after blobs are uploaded.

## Binary transfer without oversized output

Use **768000 raw bytes per read**, divisible by 3, producing exactly **1024000 base64 characters** for a full chunk, below the 1 MiB tool-output cap. Do not read an entire 8 MiB part through one command. Set `max_output_tokens: 300000`; validate exit code and exact expected length for **every** chunk. The final chunk length is `4 * ceil(raw_bytes / 3)`. Keep every chunk offset divisible by 3, concatenate in order, then require `4 * ceil(file_bytes / 3)` characters. Never print base64 into model output or logs.

The following is a future worker orchestration pattern, **not executed here**. Select inventory rows with `upload_required=true`, deduplicate by `git_sha1`, and recheck their SHA-256/Git SHA locally before upload. `rows` below is a batch of at most two parts read from a verified local manifest, not untrusted shell input:

```js
const quote = s => "'" + s.replaceAll("'", "'\\''") + "'";
const tasks = rows.flatMap((row, file) =>
  Array.from({length: Math.ceil(row.bytes / 768000)}, (_, chunk) => ({row, file, chunk})));
const chunks = rows.map(() => []);
for (let start = 0; start < tasks.length; start += 16) {
  const batch = tasks.slice(start, start + 16);
  const results = await Promise.allSettled(batch.map(async t => {
    const r = await tools.exec_command({
      cmd: "set -o pipefail; dd if=" + quote(t.row.absolute_path) +
        " bs=768000 skip=" + t.chunk + " count=1 status=none | base64 -w0",
      shell: "/bin/bash", login: false, max_output_tokens: 300000,
      yield_time_ms: 1000
    });
    const raw = Math.min(768000, t.row.bytes - t.chunk * 768000);
    if (r.exit_code !== 0 || r.session_id || r.output.length !== 4 * Math.ceil(raw / 3))
      throw new Error("Chunk incomplete/truncated: " + t.file + ":" + t.chunk);
    return {file: t.file, chunk: t.chunk, data: r.output};
  }));
  const errors = [];
  for (let i = 0; i < results.length; i++) {
    const r = results[i];
    if (r.status === "rejected") errors.push(String(r.reason));
    else chunks[r.value.file][r.value.chunk] = r.value.data;
  }
  if (errors.length) throw new Error(errors.join("; "));
}
for (let i = 0; i < rows.length; i++) {
  const content = chunks[i].join("");
  if (content.length !== 4 * Math.ceil(rows[i].bytes / 3)) throw new Error("File length differs");
  store("c05-base64-" + rows[i].git_sha1, content);
}
text({prepared: rows.length}); // No base64 output.
```

If a read yields a process session, it is incomplete: finish that same foreground mechanical read with short `write_stdin` polls and exact combined-length checking before upload; do not discard or restart it blindly. The example deliberately rejects such an incomplete result. Normal small local reads complete immediately. No independent agent sessions or detached drivers are involved.

Upload via `create_blob({repository_full_name: "Caelestis1903/My-Research-Project", content: load(key), encoding: "base64"})`, at most two 8 MiB parts concurrently with `Promise.allSettled`; inspect **both** results and require their returned Git SHAs. Do not leave unawaited promises when an exec isolate ends. Target each read/upload batch at **40 seconds or less**, yielding short progress between batches so the parent can keep its owner polls within **60 seconds**. C04 observed roughly 8 parts/minute with two parallel part uploads; this is an observation, not a completion guarantee. If a call runs long, use short tool yields/waits and communicate outstanding work; never invent a successful receipt on timeout. Parent alone polls RAM PID 2926581 / session 16855. This worker and its successor must not access owner IPC or credentials.

## Future publication sequence, not authorized now

1. Parent completes supervised RUN, acceptance and analysis/report/context, archive packaging and distinct final evidence review with blockers corrected. Recheck both workspaces' stop controls, exact frozen source, current selected-RAM availability and honest disclosures. Respect the scheduled stop after C05; no C06 or Test. Obtain the parent's fresh-child execution handoff. Do not infer final readiness from this preparation.
2. Re-run the offline helper for the complete current public files. Verify the full inventory against review/package identities, no private/raw duplication, and exact C04 receipt. Keep those bytes stable through publication. The helper remains marked NOT_FINAL; the executing worker separately records the parent's final evidence acceptance. Keep actual HEAD/index unchanged until remote verification.
3. Read remote `main` with `search_branches` and require the exact expected C04 SHA. Upload only missing/new blobs and require every SHA. Retain compact upload receipts. Build trees sequentially bottom-up with no base and verify each SHA, root last. Never reupload C04 parts.
4. Call `create_commit` with the root tree, exactly C04 as `parent_sha`, no additional parents, and the parent's approved publication message. This creates an unreferenced commit first. Inspect/fetch the actual Git commit metadata before moving `main`.
5. Reconstruct the exact commit **locally** from returned author/committer name, email, date timestamps, ordered parents, tree and message. Git object bytes are `tree ...\nparent ...\nauthor NAME <EMAIL> UNIX OFFSET\ncommitter NAME <EMAIL> UNIX OFFSET\n\nMESSAGE`. C04 used `+0900` for author and committer and **no trailing newline** after the message; its exact bytes are in `/tmp/phase0-3-cycle04-exact-commit.txt`. API dates normalize away the original offset. Try `+0900` first, then `+0000`/`+0800` variants for author and committer as necessary, preserving exact metadata/message. Compare `sha1(b"commit " + length + b"\0" + bytes)` against the remote commit SHA before `git hash-object -t commit -w --stdin`. Never use `git commit-tree`, which can change trailing-newline bytes/identity. If no exact reconstruction matches, pause before ref changes and inspect the actual metadata; do not guess a local commit. Store the exact matched bytes in a new C05 temporary receipt.
6. With the parent still supervising, call `update_ref({repository_full_name, branch_name: "main", sha: NEW_SHA, expected_sha: EXPECTED_C04_SHA, force: false})`. On a lease mismatch inspect the actual head and return to the parent; never force or silently rebase. Fetch/read the actual remote branch and commit again, verify branch SHA, parent, tree and public contents/package identities. This means connected read tools, not a shell `git fetch` of missing thin objects. A failed or unknown ref response requires readback before any retry.
7. Only after exact remote verification and fresh parent authorization, recheck detached local HEAD still equals C04, actual index SHA equals the captured original, every inventory byte/hash and frozen source still match, and prepared index's tree equals the verified remote tree. Preserve unrelated working files. The future guarded alignment uses a cooperative original-index lock (`index.lock`, exclusive creation) and `HEAD.lock`, checks the captured bytes under those locks, fills the index lock from `prepared.index`, and updates detached HEAD with an expected-old-SHA compare-and-swap (or checked exact detached HEAD bytes under `HEAD.lock`). Use atomic replacements; record each step, and pause/reconcile any partial failure. HEAD plus index cannot be atomically replaced as a pair: do not claim a crash-proof transaction. Do not use a worktree reset or checkout that overwrites other workers' files. No such code or update is executed by this preparation helper.
8. Fresh worker writes the C05 publication receipt only after remote/local verification, including current availability disclosures, exact commit/tree, inventory and checks. The receipt cannot include its own commit hash within that same commit without a separate follow-up publication; report this honestly. Return to the parent for final delivery/USER_STOPPED. Do not launch C06 or another worker.

## Preparation verification

The bounded check is syntax compilation in memory plus the current-files dry-run. The helper itself verifies copied-index base identity, actual SHA-256/Git SHA-1, source55 identity, the exact C04 receipt, every immediate-directory tree SHA, allowlist-only index differences, and unchanged real HEAD/index. Current verification paths and outcome are recorded in `PUBLICATION_PREPARATION.json`; dry-run manifests remain outside Git. Training, scientific results, archive closure, final report/review and remote publication are intentionally pending.
