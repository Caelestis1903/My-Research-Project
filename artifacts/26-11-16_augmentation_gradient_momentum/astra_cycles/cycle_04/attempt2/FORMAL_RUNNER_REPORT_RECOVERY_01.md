# Cycle04 attempt2 recovery runner report

## Raw run status

The foreground recovery command exited with code 0 and reported `PAUSED`. Controller PID was 2611647 in tool session 75552; foreground duration was 1418.012 seconds. Source hash and design bundle hash matched the approved receipts. No Test data was accessed and no source files were edited.

The parent-filtered formal progress was 889/1000 ACK epochs and 32,251/34,500 useful Adam updates across 20 formal keys. Fifteen jobs completed 50 epochs; five jobs remained unfinished (111 ACK epochs and 2,249 useful Adam updates). Engineering smoke keys and their four epochs are excluded.

## Technical pause

Worker PID 2619341 (key `2df253901fbd915c33bb3f5a2c6152666931bbd1bc093b16f9aee1828dd1932c`) exited code 1 after 689.663 process-wall seconds. Its traceback records `TechnicalPause: NVML measurement failed code=3` during a resource profile mark; cleanup raised the same pause while closing the profile. The partial train-call count is unknown and remains null. See [TECHNICAL_PAUSE.json](TECHNICAL_PAUSE.json) and [worker foreground log](runs/2df253901fbd915c33bb3f5a2c6152666931bbd1bc093b16f9aee1828dd1932c/foreground.log).

## Physical execution accounting

Seven formal worker processes were started and all seven exited: six with code 0 and one with code 1. Known returned train calls sum to 2,876; one failed worker has an unknown partial call. The sum of observed process-wall durations is 6,973.316 seconds. Those worker durations overlap and are not elapsed run time. Raw rows remain in `physical_attempts.jsonl`; no unknown partial duration or call count was inferred.

## Handoff and tables

The selected-RAM runner handoff was not run because the formal run paused before all 20 jobs completed. Tables export was not run. Consequently there is no table receipt or `RESOURCE_EXECUTION_AUDIT` from this attempt.

This report contains execution accounting only, with no research analysis. A fresh recovery review is required before any further run.
