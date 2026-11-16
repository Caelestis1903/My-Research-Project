# Cycle 01 policy-switch runner report

## Execution status

- Experiment: `26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_01`.
- State: `PAUSED_POLICY_SWITCH`; 11/30 complete, 18 pending, 1 paused at an acknowledged epoch, 0 failed.
- Git HEAD: `c4dc36bd38bd4f7a2a3319377a8289ba8185532d`; accepted source SHA-256: `6b16d5e6bd5fcb96d8824a4161ec102af25e5a23ed86ff21f70d21e05eee5ff1`.
- Config: datasets movielens, kuairand; arms AXIS-EVEN, AXIS-IID2, ISOTROPIC-EVEN; seeds 2022, 2023, 2024, 2025, 2026; 50 epochs; batch 40960; FP32; 51,750 Adam and EMA updates per complete trajectory.
- Retained in parent-owned RAM: 12 states, 133723208 bytes. No release performed. Test access: false.

## Acknowledged partial trajectory

- `c73c47a85ca0a71d32fdad9445c8750dc8d84f4206fca94de11cf99aee0e5cce` — kuairand / AXIS-IID2 / seed 2023; ACK epoch 21/50, global step 1029, counts `{"ema_updates": 1029, "optimizer_updates": 1029}`.
  Last acknowledged numeric receipt: `artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_01/attempt1/runs/c73c47a85ca0a71d32fdad9445c8750dc8d84f4206fca94de11cf99aee0e5cce/attempt1/epoch_21.json`; per-epoch progress: `artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_01/attempt1/runs/c73c47a85ca0a71d32fdad9445c8750dc8d84f4206fca94de11cf99aee0e5cce/attempt1/progress.json`.

## Raw run records and logs

The JSON companion contains per-run status and raw selected validation AUC/loss fields for completed trajectories, plus the paused trajectory's last acknowledged loss/validation fields. No incomplete-run tables or scientific analysis were produced.

Logs and raw receipts remain in place:

- `artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_01/attempt1/GPU_SMOKE.log`
- `artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_01/attempt1/launch_ledger.jsonl`
- `artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_01/attempt1/worker_exit_ledger.jsonl`
- `artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_01/attempt1/epoch_commit_ledger.jsonl`
- `artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_01/attempt1/launch_logs/0d91e6dd41402349ba929b3c7876fc966f1996a69aedac96e38db0c639835c6a.log`
- `artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_01/attempt1/launch_logs/315b2f631a9d128ec2429d408644ab5070d6f42c08ef9c26eded3d60f59b512c.log`
- `artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_01/attempt1/launch_logs/3a45e24dbcee542164e829e1b83fe44e2aa69d10d5369271d68b8e9b7eda8802.log`
- `artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_01/attempt1/launch_logs/3a91a8718c67668d2356be8900ab7d2ea42138c480ce3b4948a7f5538baea10e.log`
- `artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_01/attempt1/launch_logs/5bcf9044e17f63b2d6cff56b00c02cdc030d9b5b21b741d8ac9c3f4ccbd3860a.log`
- `artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_01/attempt1/launch_logs/64736e46b18b740b6a93ae50fd1d93f361839a23ad2c4d79f6381e1bfd93f9c6.log`
- `artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_01/attempt1/launch_logs/8b1008ef65ffe408d27bfc6a5e5c449d805e10cedc0c1ce7593dca0296bdec05.log`
- `artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_01/attempt1/launch_logs/b4dfa126baadd6e3cc319254b93c4086c51564be3900a5d5026d7fa428de7be0.log`
- `artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_01/attempt1/launch_logs/c73c47a85ca0a71d32fdad9445c8750dc8d84f4206fca94de11cf99aee0e5cce.log`
- `artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_01/attempt1/launch_logs/cad1ba4d38346d79971ba41410c32c46a15c2a4b124698d7035ca7ff62a8d3b7.log`
- `artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_01/attempt1/launch_logs/de2ab87a2dba6dacbc72e0c65fc088c74a69b0ded49518702ed61cdd52b56b8c.log`
- `artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_01/attempt1/launch_logs/fa92a79586c7fc47b71b387f0aa34dbbc04bfbea7c81e947db150cd7e087f3c6.log`

## Runtime status

No runtime error or failed key was recorded. The parent-issued scheduler SIGTERM caused this policy-switch pause. All scheduler and owned worker processes were confirmed absent; the current ACK state remains retained by the existing storage owner.

JSON companion: `POLICY_SWITCH_RUNNER_REPORT.json`.
