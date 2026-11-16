# Scheduler and Resource Budget

NDP-A observed ML1205–1437 worker seconds/trajectory, KR2618–3984; peak allocated about2.51/2.92GB. TV-GRM cost must be measured by the reviewed full-batch smoke.

50 new runs:25ML+25KR, 2500epochs. ML800167 rows => ceil(800167/40960)=20 steps/epoch =>1000 per50epoch trajectory. KR2006132 rows =>49/epoch =>2450/trajectory. Total25*1000+25*2450=86250 parameter updates and86250EMA. Historical CLEAN/NDP-A add zero training. This is one matrix budget; the outer loop has no cycle/experiment/time cap.

Existing per-GPU admission requires memory AND power<85%, host RAM<90%; admit on an eligible card without waiting for every GPU. First unmeasured allocating worker runs alone. OOM preserves logs and PAUSES; no silent batch/epoch/precision/seed reduction.

Retain only current atomic recovery/current selected staging plus at most one champion configuration's five selected seeds PER dataset. Champion selection: largest equal-dataset mean of five-seed EMA Macro AUC increments across the three gammas, ties smaller gamma; no upgrade if all gammas nonpositive on both datasets. Later candidates need identical validation scalar/protocol to replace champion. Release other candidate states after reporting/decision and displaced champion after replacement. Use existing guardian/admission, no accumulating rejected-cycle models or disk tensor files.

Never release current champion/recovery before verifying retained identities. Missing live RAM => PAUSED and exact-protocol rerun of affected trajectory, not recovery from hashes or final-state substitution. On manual stop retain eligible selected RAM for final freeze; absent final candidate means no Test.
