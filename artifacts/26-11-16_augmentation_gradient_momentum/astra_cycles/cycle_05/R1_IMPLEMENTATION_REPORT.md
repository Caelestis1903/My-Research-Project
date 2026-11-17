# C05-R1 implementation correction

Mendel / `01a1e9f2-7a96-77e3-a49b-888f169cce36`, gpt-6.1-sol/high, fork_context=false. Original parent: `01a1dfb8-605b-7430-8462-5fd3da0b048f`. No descendants; permanent return after this handoff.

C05-R1 is closed in the corrected source at the coder stage and requires a **NEW distinct implementation review**. [Confucius's original FAIL](IMPLEMENTATION_REVIEW.md) and [machine receipt](IMPLEMENTATION_REVIEW.json) remain unchanged. This report grants no preparation or engineering release. Confucius, Laplace and Lovelace remain permanently closed.

The production correction changes only `tables.numeric_document`: state-associated names may carry scalar description/hash provenance; dict/list/tuple payloads under those names remain rejected. Unchanged `protocol.json_safe` rejects tensors, arrays, bytes/bytearray, opaque objects and nonfinite values. Recursive credential rejection, extension/path guards, mandatory archive enumeration and state-to-disk policy remain in force. The unchanged pinned manifest and engineering logs remain required archive evidence. No new parser, field whitelist, research algorithm, arm, HPO or design change.

Only three files were written: `experiments/cell_mass_bce_20261117/tables.py` and these R1 Markdown/JSON reports. Parent-owned helpers/metadata were not modified.

| Identity | Before | After |
| --- | --- | --- |
| Candidate source55 | `3f32ce34d76d2ab79a3b8fcb447722862310f7cecb6d037b326626b03e9d7e74` | `78604351e97221abd78318aa1f10e31aacfcc6c97a0d5d0e99e965d87a64bd80` |
| tables.py | `7316e79532c0eb71b99ba302d7004ac4b1030929deb491f50a823d3f1162dfee` | `79d9673e032a4b0eab4bff575fa9594e11bfcb972975a68da53a3a5c6b804965` |

The parent-preserved [original tables copy](/tmp/phase0-3-cycle05-preR1-tables.py) was used read-only and matches the original runtime/reviewer tables hash. The source closure remains exactly 55 paths; all other **54 current file hashes** match original review source_files. The full per-file unchanged proof is in [R1_IMPLEMENTATION_REPORT.json](R1_IMPLEMENTATION_REPORT.json). Source48 remains `d1930f26e2b73d04a2ed504112ca0161948468d5ce00b99b074d5214efdf3b3f`; receipt bytes remain `10f3000ae43e2a394b71cdb2febc291888f06bf216dd99c05ddd7b842186fec3`; all48 current dependencies match. External Adam remains `b099132d262d6cb16eefa48ab4bae789a0655cafb8571e421ef4fe3d2ff7c34c`.

Accepted specification remains `a1f61ae81535bb33873da72185d9400187c5b988e81c0d3a5019c5505e821af9`; design7 remains `9a7e826794f4cdc3fcc7c437e1462149a1dc9372b5d5e3a38981f69fb3af299c`; research6 remains `bfd6230c9cae8971a3a8d793027fc54ba6209d193429340107bf64bb22dc4caf`. Every accepted design input was byte-checked. Original core/runtime Markdown and JSON reports, original review Markdown/JSON and review checks remain byte-identical; their hashes are recorded in the JSON report. Hash-bound core/runtime checks and original passed static/numerical review areas are reused for unchanged files.

Both original failures reproduced before editing on the original candidate: real pinned-manifest `checked_member` and normalized real `runner.observation` each raised `TechnicalPause: Opaque training state field in public record` (tool chunk `e487ac`). Reproduction elapsed 0.5344808310037479s inside the script, 3.010676812s tool process wall; two CPU Linear instances, one lazy source Adam, one private CPU generator and one observation; zero train/forward/backward/Adam calls.

Final `tables.finite_checks()` passed **once** on the corrected source (tool chunk `7319b3`). One RAM compile passed. Four existing tables groups passed, including 17 comparisons,34 metrics,28,900 synthetic paired rows and5,780 summaries. The single extended archive fixture accepts the unchanged real manifest with SHA `42320f2842f04f0d2d49006108db52e0927c9fe392631df2c96f423440fcf27e` and a production-normalized real runner observation containing five64-character model/EMA/Adam/RNG/loader hashes. Both inputs are retained in a tiny three-member archive whose headers, lengths and bytes pass readback. All15 actual-state/container/array/bytes/opaque/credential rejection examples and all4 invalid member names are rejected. Rejection state stays in RAM; the two temporary record files contain numeric/hash provenance only and the isolated temporary directory is removed by its context manager.

Final checks took 0.5553790350677446s elapsed / 0.49770646600000035s process CPU; the full verification script including imports took 1.759894381975755s elapsed / 6.5701603760000005s process CPU (2.2539043149999998s tool process wall). Final fixture cost: two CPU models, one lazy source Adam, one private CPU generator, one observation and one tiny archive. Fail-if-called instrumentation on Linear.forward, Tensor.backward, Adam.step and runner.train_step recorded **zero calls**. Total for this coder including the original reproduction: four CPU model constructions, two lazy Adam constructions, two private generators, two observations, one tiny archive; zero train calls, forwards, backwards, Adam steps and scientific epochs. These counts add no new research training or result analysis.

Every project Python invocation used `PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -B`, Python3.13.5/Torch2.9.0+cu126. Both workspaces' stop controls were ACTIVE, manual_stop_requested=false, termination_reason=null, stop_after_current_cycle=5, cycle06_authorized=false; all four STOP_AUTONOMOUS sentinels were absent. Scheduled completion remains C05 delivery then USER_STOPPED, no C06.

The exact diff against the parent's unchanged pre-R1 copy follows:

```diff
--- before/experiments/cell_mass_bce_20261117/tables.py
+++ after/experiments/cell_mass_bce_20261117/tables.py
@@ -375,7 +375,9 @@
     p.json_safe(value)
     if isinstance(value,dict):
         p.require(not PRIVATE_KEYS.intersection(value),'Private credential field in public record')
-        p.require(not STATE_KEYS.intersection(value),'Opaque training state field in public record')
+        # State names also label descriptive/hash provenance; reject their payload containers.
+        p.require(all(not isinstance(value[k],(dict,list,tuple)) for k in STATE_KEYS.intersection(value)),
+                  'Opaque training state field in public record')
         for v in value.values(): numeric_document(v)
     elif isinstance(value,(tuple,list)):
         for v in value: numeric_document(v)
@@ -518,20 +520,44 @@
     mass_add(acc,ledger,40960); mass_add(acc,dict(ledger,total_mass=1.),21927)
     row=next(r for r in mass_rows({},acc) if r['mass_kind']=='total_mass')
     assert row['update_mean']==.95 and row['row_weighted_mean']==(.9*40960+21927)/(40960+21927)
-    # Numeric-only filesystem fixture is task-scoped /tmp, independent of R5/N5/S5.
+    # C05-R1: real design/engineering provenance belongs in the numeric archive.
+    from .runner import observation
+    import io
+    model=p.torch.nn.Linear(2,2); ema=p.torch.nn.Linear(2,2)
+    opt=p.optimizer(model,dict(lr=.01,weight_decay=1e-4))
+    loader=p.torch.Generator(device='cpu').manual_seed(2022)
+    observed=observation(model,ema,opt,loader,'cpu')
+    names=[n for n,_ in model.named_parameters()]
+    observed['activity']=[observed['activity'][n] for n in names]
+    observed['parameter_dictionary_id']=p.digest([dict(name=n,shape=list(x.shape),dtype=str(x.dtype),numel=x.numel())
+        for n,x in model.named_parameters()])
+    assert all(isinstance(observed[k],str) and len(observed[k])==64 for k in ('model','ema','optimizer','rng','loader'))
+    manifest=checked_member(p.P5/'EXPERIMENT_MANIFEST.json','design/EXPERIMENT_MANIFEST.json','design',root=p.P5,
+        expected_sha='42320f2842f04f0d2d49006108db52e0927c9fe392631df2c96f423440fcf27e')
+    # Only numeric/hash records reach this isolated /tmp fixture; state examples stay in RAM.
     with tempfile.TemporaryDirectory(prefix='c05-lovelace-map-') as directory:
         root=Path(directory); path=root/'numeric.json'; path.write_text('{"count": 3, "test_accessed": false}\n')
         row=checked_member(path,'attempt1/numeric.json','raw',root=root)
-        archive=root/'fixture.tar.gz'; write_archive(archive,[row]); assert archive.is_file()
+        path=root/'engineering.jsonl'
+        path.write_text(json.dumps(dict(observation=observed,test_accessed=False),allow_nan=False)+'\n')
+        engineering=checked_member(path,'attempt1/engineering.jsonl','raw',root=root)
+        archive=root/'fixture.tar.gz'; write_archive(archive,[row,manifest,engineering]); assert archive.is_file()
         for name in ('../escape.json','/absolute.json','attempt1/guardian.json','attempt1/state.pt'):
             try: member_name(name)
             except p.TechnicalPause: pass
             else: raise AssertionError('Invalid archive member accepted')
-        for value in ({'authkey':'secret'},{'model':{'weights':[1,2]}}):
+        rejected=({'authkey':'secret'},{'nested':{'password':'secret'}},{'model':{'weights':[1,2]}},
+            {'model':model.state_dict()},{'ema':ema.state_dict()},{'optimizer':opt.state_dict()},
+            {'rng':p.rng_state('cpu')},{'loader':loader.get_state()},{'selected_models':{'online':model.state_dict()}},
+            {'array':p.np.zeros(1)},{'bytes':b'state'},{'bytearray':bytearray(b'state')},
+            {'opaque':io.BytesIO()},{'model':[1,2]},{'model':(1,2)})
+        for value in rejected:
             try: numeric_document(value)
             except p.TechnicalPause: pass
             else: raise AssertionError('Private/state record accepted')
-    return dict(status='PASS',groups=4,comparisons=17,metrics=34,paired_seed_rows=28900,paired_summary_rows=5780,test_accessed=False)
+    return dict(status='PASS',groups=4,comparisons=17,metrics=34,paired_seed_rows=28900,paired_summary_rows=5780,
+        archive_provenance=dict(manifest_sha256=manifest['sha256'],engineering_hash_fields=5,archive_readback=True,
+            rejected_state_or_credential_records=len(rejected),train_calls=0,forwards=0,backwards=0,adam_steps=0),test_accessed=False)
 
 
 def main():
```

No broad duplicate core/runtime suite, historical archive/raw-data rehash, data unpickling/preparation, GPU action, owner IPC, source freeze, attempt preparation, Test access or scientific analysis was performed. The corrected candidate remains unfrozen; the parent must close Mendel and obtain a fresh distinct R1 review before proceeding.
