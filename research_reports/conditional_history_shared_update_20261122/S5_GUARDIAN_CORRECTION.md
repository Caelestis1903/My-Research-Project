# S5 selected-state lookup

The accepted result is stored at `<attempt>/runs/<key>/result.json`. The guardian owns `<attempt>/guardian.json`. S5 used `parents[3]`, which pointed one directory above the guardian. Freeze and evaluate both use the corrected `selected_ram` lookup at `parents[2]`. The regression fixture stores a guardian identity at the runner's actual directory depth, exercises the production generation calculation, and checks the path supplied to each transport call.

## Live Validation readback

The read-only check called `accepted_result(path, "movielens", arm, 2023)` followed by `selected_ram(row)` for these two completed jobs. Both passed their full 50-epoch Validation curve, accepted-result, current registry, selected-metric and tensor-identity checks. Both resolved the same owner generation `f2a6427b61b812f225f910e6991ede53507fd72383c1a03eaac50db9ea01d353`.

| Arm | Trajectory | Selected epoch | Fixed EMA tensor SHA-256 |
| --- | --- | ---: | --- |
| A10 | `6f458790c6be9480e837d93fc4fbdd4fc831e1aee4817375b6f23845ccd71cd1` | 20 | `b5ff208db80e1a33f4043621d9a78e847ac9e60d714e74ff3279a23f3a531170` |
| A21 | `4d7ec2bae1fb7aa953fa9e9c9de03292efe5f779a109dd7d0e5d7fd70014b9ee` | 20 | `8e8034277dafec5003e1d43cb2b43bcb5ec3a7444a6545dbb766f1cd6f11fd52` |

## Matrix check

Using the resolved A21 configuration at λ=0.05, the S5 matrix builder produced nine arms and eleven directed comparisons per dataset. All 90 Validation rows joined across five seeds. The 58 archived rows and 32 accepted local rows passed the common endpoint/support adapter. Every internal selected trajectory, epoch and model hash matched the accepted A decision tables. Each of the 28 archived core rows also matched its S0 endpoint metrics and identity.

This check used the Validation adapters and the current guardian's read-only operations. Production freeze and Test evaluation remain subject to independent review and selection approval.
