# CODER IMPLEMENTATION REPORT — Training kernel

## Implementation

- `protocol.py` resolves the approved arm, model, budget and dependency fields into an immutable `RunSpec`. Its canonical dispatch preserves the original B0, B1 and A21 optimizer configuration and serialization. Matching short kernels and optimizer controls retain their own identities.
- `sources.py` verifies the two pinned Git closures and external published blobs. The bounded Train/Validation bridge reuses the pinned feature and vocabulary primitives, publishes read-only input arrays and validates the encoder, scaler, array and row identities. The Test loader requires the policy hash, persisted final freeze, and selected-RAM claim.
- `groups.py` extends the native M3oE registry with ordered coordinate masks and per-scope hashes. It registers the domain embedding rows under their three N groups and uses genuine tower reachability across six cells to register HyperGate shared dense parameters.
- `objectives.py` delegates B0, B1 and H0 to the frozen source training step. Candidate and conditioned branches use static PAIR on the actual forward graph and expose one post-backward observer callback. Gaussian uses the pinned GA31-R3 draw and the clean shared PAIR graph. HyperGate uses the native objective and captures its tower logits in that call. The AutoIFS repair entry invokes the native search/reuse helpers for shuffle, temperature and the hard-ticket transition.
- `optimizers.py` implements prospective per-parameter C1, C2, C12, S, P and M updates. Native Adam owns the coordinate complement. New state includes per-parameter native and auxiliary clocks and preserves the long/short M-shadow histories. Prepared custom states and native inputs are checked before submission. A nonfinite actual native step raises before the custom commit or EMA, allowing the runner to restore its verified epoch boundary.
- `kernel_checks.py` contains standalone finite algebra, transaction and first/tail source-contract checks.

## Checks

- Python syntax and module imports passed using the repository environment. Runtime reported Python 3.13.5, PyTorch 2.9.0+cu126, CUDA runtime 12.6 and no version differences against the package lock.
- Both bounded populations matched pinned CSV, feature arrays, label arrays, ordered row IDs, encoder/scaler and train counts. The verified Train/Validation identity hashes are `1cdbb70149262dd98ec8147132d8573ba4c8da9a4a12142a1df55da0f409da8b` for MovieLens and `d04c8460dea7b817ff99f30e4bb38922dece4b284b86ff46502c98ea1f33dca3` for KuaiRand.
- The independent FP64 C1/C2/C12 and S direction reference had maximum absolute error `7.771561172376096e-16`. Additional checks passed for constant D and the first auxiliary event, P recurrences, no PAIR, active unused raw gradient, native None, coupled WD, c=.1 cold/hot Adam equivalence, M's separate Q99 and Q9/U9 shadows, FP32 norm-match displacement and exact optimizer state roundtrip.
- The Test data bridge rejected an otherwise well-formed claim with absent freeze receipts before entering the loader.
- MovieLens CPU construction verified M3oE, HyperGate and AutoIFS factories against the pinned sources. The M3oE registry recorded 286,208 S_DENSE and 196,944 ordinary shared embedding coordinates. The HyperGate probe registered 436,096 shared dense coordinates with non-None logits gradient access across all six cells.
- On a fixed 256-row MovieLens fixture, B0/A0, B1/P00, and both registered .99 candidate zero aliases/A21 matched the direct frozen loss and model, optimizer, and EMA hashes. The two zero aliases produced the same hashes, including the A21 six-cell shadow state.
- The seed-2022 CPU first/tail checks passed all four actual batch shapes. In each isolated transaction, wrapped H0 matched the direct A21 loss and all three state hashes. The candidate used the same PAIR samples, one clean forward, one raw sweep and one update. Every tensor outside the custom scope matched the H0 update exactly.

| Dataset | Source step | Rows | PAIR hash |
| --- | ---: | ---: | --- |
| MovieLens | 1 | 40960 | `2c633e948e1afc1e0ca5f8c887cd0adf6aa8441bba3089200e6844fdee94e94c` |
| MovieLens | 20 | 21927 | `5350ad6c1995215a6fdaea1e7ea9f09de10cf84c08d753433fd131ceb33250a5` |
| KuaiRand | 1 | 40960 | `c640995bf66ddf5d75e2d19ce6505edad822a2f7d437e7223b51b4aae58198b5` |
| KuaiRand | 49 | 40052 | `742253fc09d09578d050dd1c803f48f66fe557c3581bc0792a97a579bbf94b38` |

The exact matched A21 state hashes, ordered online, optimizer, EMA, were recorded by `python -m experiments.history_information_20261124.kernel_checks --real`.

```text
MovieLens 1   fbeae46bbe6574546911b34789b413d060da77d08f651e3df1ec4ac75d9e7cd7
             b405c0c8738b4d5df409a35a8af3bbb6ffb3531aa46e9269b4385b75aa2c2222
             8b1b42e1623bd8cc5b99c3ff2a6a0faeabc157058fb121449c3f0d8fffff0277
MovieLens 20  fac8b259b3dfe243054490be07b8c6bfb46e52d43b063a267b127e92df6da81e
             1a624f2656b1c9eb56f2b4982f91b2a47edeabb4570c0b065b9a1b435f786120
             be0a890fc2bbe8c9509d098f10c471259b43110a70a66cd947bdcf203c401f24
KuaiRand 1    26c08b32d32fe69ffa6b75c55cabe80575b252f71fd9051605634c3409d18410
             2dc580018e880f091d2543e68a120adcdb757431cc04fb16b475904129a27c93
             3bbfd36a3ff5e410b3f10f85542f3824494935fa469422cea9d0378363b6731a
KuaiRand 49   6a3b5d1eb38174101f255d75caf99d9d60369dfac15c3391d39d1c627ebe0c76
             ea4a6287e145f9dfbd64d209abaae880703dfcc804d9652edea6641247ac8cf5
             dc64ddb5a7d52deb231d1de61ba4534fa17ce5a94102dc5ea2fb31ee1f57d6f6
```

## Review handoff

The CUDA same-device replay and the independent acceptance verdict remain review responsibilities. Conditioned full-batch checks remain tied to their registered triggers. This implementation stage used CPU contract checks and left all training budgets and Test access unconsumed.
