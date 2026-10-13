# EXP-1 original-return evaluation

Classification: **EXP1_BASELINES_CREDIBLE**
Episodes: 270/270. Protocol SHA256: `91e85e5bae605ff25930dbfb82f577a2e581388cff08d1a5155ce42300b353b8`.

Execution validity is separate from episode motion eligibility (return >=650 and no native failure). Training replicates n=3 per method; episode bootstrap intervals are descriptive within-checkpoint intervals.

| Method | Train seed | Episodes | Mean return | Return gate | Native failures | Motion eligible |
|---|---:|---:|---:|---:|---:|---:|
| DreamerV3 | 0 | 30 | 138.98203346005036 | 1 | 29 | 1 |
| DreamerV3 | 1 | 30 | 283.5587989624893 | 6 | 23 | 6 |
| DreamerV3 | 2 | 30 | 302.4279137786732 | 4 | 22 | 4 |
| TD-MPC2 | 0 | 30 | 56.95125445539982 | 0 | 30 | 0 |
| TD-MPC2 | 1 | 30 | 37.702391972329586 | 0 | 30 | 0 |
| TD-MPC2 | 2 | 30 | 31.887503326140212 | 0 | 30 | 0 |
| SAC | 0 | 30 | 3.246809870972063 | 0 | 30 | 0 |
| SAC | 1 | 30 | 3.245484647800733 | 0 | 30 | 0 |
| SAC | 2 | 30 | 1.8774007322119424 | 0 | 30 | 0 |

| Method | Mean of three checkpoint means | Median of three checkpoint means |
|---|---:|---:|
| DreamerV3 | 241.6562487337376 | 283.5587989624893 |
| TD-MPC2 | 42.180383251289875 | 37.702391972329586 |
| SAC | 2.789898416994913 | 3.245484647800733 |

Native termination: robot root or tool below the benchmark height threshold. Native info does not distinguish which condition. Horizon truncation is separate. No additional failure detector was used.

Official-curve sanity check (read only after all 270 episodes): frozen benchmark `logs/main_results.json`, task `window`. These are published training-curve endpoint values, not evaluations under the new matched 30-episode protocol; source seed labels are not asserted to match this run's training RNGs.

| Method | Official curve endpoint returns (seed labels 0/1/2) | Final training steps (millions) |
|---|---|---|
| DreamerV3 | 227.479371, 134.596452, 108.697521 | 10.0, 10.0, 10.0 |
| TD-MPC2 | 48.402886, 52.612392, 88.246283 | 2.0, 2.0, 2.0 |
| SAC | 3.418726, 178.253466, 3.804900 | 10.0, 10.0, 10.0 |

Observed sanity interpretation: DreamerV3 remains at hundreds of return and TD-MPC2 at tens, consistent in order of magnitude with the published endpoints. All three reproduced SAC checkpoints are in the low-return regime represented by two published seeds; the published third SAC seed reached about 178.25, so this reproduction does not reproduce that higher-performing outcome. This difference is reported, not tuned away. It does not by itself establish a loading/evaluation bug: all frozen checkpoints restored, native action paths ran with finite actions and rewards, all fixed episodes completed, and all nine workers exited 0. No scientific episodes were rerun or selected.

Force work performed: FALSE. No force signals queried or saved; no nIAE_F, contact analysis, or EXP-2 execution.

Stop at EXP-1.
