# D2 implementation

- `source.py` reproduces the canonical Train and Validation population.
- `s0.py` checks source identity, Adam fidelity and parameter support.
- `blocks.py` applies native Adam blocks and canonical EMA updates.
- `training.py` executes J_ORIG and verifies epoch, anchor and fork RAM state.
- `diagnostics.py` freezes panels and executes isolated S1 measurements.
- `scheduler.py` supervises the registered S1 job matrix.
- `statistics.py` constructs the registered scalar tables and Validation decision.
