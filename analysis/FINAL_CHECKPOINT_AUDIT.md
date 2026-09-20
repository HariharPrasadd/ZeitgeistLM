# Final checkpoint audit

Training ended at step 11,320 on September 20, 2026, at 05:31 EDT. The final
checkpoint is `/data/checkpoints/final_2967M.pt` on the `zeitgeistlm-data`
Modal Volume. The training app is stopped. `latest.pt` held the same final step
when it was copied. Training began at 02:14 EDT, giving 3 h 17 min of elapsed
wall time including restarts, validation, checkpointing, and the rollback period.

The model has 124,478,208 parameters. With 262,144 token presentations per
optimizer step, the final checkpoint records 2,967,470,080 token presentations.
The training manifest contains 2,967,392,256 tokens, so this is about 1.00003
effective corpus passes. Windows were sampled with replacement, so this is
not one exhaustive sequential epoch. A 6ND estimate gives 2.22e18 training
FLOPs, excluding attention overhead and non-training work.

The final training-step loss was 2.2683. The last 20-batch validation estimate
was 2.1927. These are noisy and should not be compared directly to the fixed
window audit below.

For a credit-bounded comparison, both the 1B and final checkpoints were scored
on identical deterministic stratified windows: 48 from 2021 validation and 32
from 2022 future test. These are 1024-token windows selected two per available
month/kind stratum. The future test had already been used in prior exploratory
analysis, so it is no longer an untouched final test.

| Checkpoint | 2021 loss | 2022 loss | 2021 true-date gain vs fixed 2020 | 2022 true-date gain |
| --- | ---: | ---: | ---: | ---: |
| 1.000B tokens | 2.30559 | 2.50115 | 0.00229 | 0.00535 |
| 2.967B tokens | 2.17497 | 2.32259 | 0.00678 | 0.01158 |

The final checkpoint improves held-out loss by 0.13062 on these validation
windows and 0.17856 on these future-test windows. Temporal conditioning also
helps more on this sample, though the gains remain small in absolute NLL.

Matched-seed generation was repeated for three prompts at 2016, 2020, 2022,
and extrapolated 2023. The 2016-vs-2023 next-token Jensen-Shannon divergences
at the final checkpoint are 0.0183, 0.0196, and 0.1232 nats for `bro really
thought`, `when you realize`, and `nobody:`. Outputs and all per-window values
are in `final_checkpoint_results.json` and `comparison_1b_results.json`.

The older 485-item regex-family genealogy diagnostic gives 50.4% same-family
links at the final checkpoint, versus 53.1% at 1B; time-penalized links give
51.0% versus 52.3%. This lexical benchmark did not improve with more training.
It is separate from the broader unsupervised genealogy graph. The learned
time embedding remains rank one by construction (`nn.Linear(1, n_embd)`).

The initial credit-bounded audit did not rerun the larger experiments. They
were subsequently completed on the final checkpoint; see
`FINAL_FOUR_RESULTS.md` for the full sweep, emergence, genealogy, and mutation
generation results.
