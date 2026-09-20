# ZeitgeistLM: fixed-checkpoint analysis

## Scope and provenance

This is an exploratory analysis of the **1,000,079,360-token checkpoint**, saved before the unstable training run's spike. Training data ended December 2020. Validation is 2021; the 2022 January–August test split was opened here after choosing this checkpoint. The 2023 coordinate is an extrapolation with **no 2023 ground truth in this corpus**. The rerun from the 1B checkpoint continues separately and is not mixed into these measurements.

`analysis/analysis_modal.py` sampled cleaned submissions on Modal and ran inference on an A100. Large Parquet and token shards stayed on the Modal Volume. `collect_results.json` records the sample frame; `analyze_results.json` contains all generated text, per-window scores, geometry measurements, and genealogy nodes/edges. `summarize.py` creates flat tables, plots, and `summary.json`. Reproduce with `modal run analysis/analysis_modal.py --phase collect`, then `modal run analysis/analysis_modal.py --phase analyze`, then `python analysis/summarize.py` from the repository root. This analysis uses immutable milestone checkpoint names, not `latest.pt`.

## 1. Time-conditioned generation

With the same GPT-2 token seed, top-50 sampling, temperature 0.8, and random seed 42, the model produces different continuations at the midpoints of 2016, 2020, 2022, and 2023. For `nobody:`, the 2016 sample begins as ordinary quoted dialogue, while 2020–2023 samples start with a `Me:` meme-dialogue pattern. The first-next-token distribution's 2016-versus-2023 Jensen–Shannon divergence is **0.0962 nats** for `nobody:`, versus **0.0173** for `bro really thought` and **0.0247** for `when you realize`. Total variation is 0.367, 0.114, and 0.173 respectively. This is demonstrable conditioning, but sensitivity varies greatly by prompt.

These are illustrative stochastic samples, not evidence that the model predicts 2023 memes. Continuations often include `<|endoftext|>` and `<|subreddit_...|>` because the training stream has packed documents and subreddit delimiters. The generation utility currently keeps sampling after a delimiter; a public demo should end at the first EOS and display the continuation before it. The raw, unedited samples are in `analyze_results.json`.

## 2. Held-out future forecasting

We drew **12 windows for each month × content-kind stratum**, deterministic seed 20260920, from both held-out splits. Each window has 1,024 predicted tokens and was scored twice with identical token IDs: at its stored timestamp and with time frozen at 2020-07-01. The paired comparison isolates the value of the time coordinate for these windows. BF16 inference accumulates cross-entropy in FP32. The interval below resamples months 10,000 times; it measures sampling variability over these sampled months, not uncertainty across different training runs or communities.

| Split | Windows / tokens | Actual-time loss | Fixed-2020 loss | Actual-time perplexity | Loss gain (95% month bootstrap interval) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2021 validation | 288 / 294,912 | 2.32285 | 2.32474 | 10.205 | 0.00190 (0.00094–0.00288) |
| 2022 Jan–Aug test | 192 / 196,608 | 2.60516 | 2.61161 | 13.533 | 0.00645 (0.00513–0.00761) |

Actual time improves the 2022 sample in **all eight sampled months**, but the improvement is small: roughly 0.25% in per-token loss against a mid-2020 time condition. The 2022 absolute loss is substantially higher than 2021, particularly for submissions (2.686 versus 2.313); dataset composition and changing language may both matter, so this difference alone does not quantify cultural drift. This is genuine extrapolation beyond the training interval (`τ > 1`), albeit only through August 2022. See `forecast.png` and `forecast_windows.csv`.

## 3. Phrase genealogy and branching

We scanned March and September cleaned **submissions** from 2011–2022 where available, matching ten predefined phrase patterns, and kept at most eight stable-hash-selected occurrences per family/year. One pattern (`bro really thought`) had no matching rows in this sample frame. The resulting **485 real posts** cover nine families. This is a sparse phrase sample, not the complete occurrence history; 2022 only has March in this corpus. Earlier innocuous uses of “among us” are a known word-sense false positive.

For each full title/text (up to 128 GPT-2 tokens), we captured the final-token output of the model's final layer norm, then linked it to its closest earlier sampled vector by cosine similarity. A second graph subtracts **0.03 cosine units per elapsed year**. Of 484 edges, **53.1%** connect matching predefined phrase-family labels without the penalty, and **52.3%** do with it; a shuffled-label control averages **12.1%**. The penalty reduces the mean ancestor gap from **1.76 to 0.79 years**.

Crucial control: a simple chronological **TF–IDF word/bigram cosine** graph reaches **79.3%** same-family edges on the same 484 comparisons. Thus the current hidden-state graph does **not** beat a lexical baseline on this diagnostic. Its hubs (for example, an older doge post receiving 11 penalized child links) are graph branching points, **not verified semantic mutation, influence, migration, or persistence**. The common phrase itself, community vocabulary, and post length can dominate similarity. `genealogy.png`, `genealogy_nodes.csv`, and `genealogy_edges.csv` preserve the graph for inspection, but a convincing cultural-lineage claim needs curated variant labels, independent ground truth, and stronger lexical/time-matched controls. We did not infer branch survival or migration from this sparse sample.

A follow-up in [GENEALOGY_V2.md](GENEALOGY_V2.md) tests mean pooling, intermediate layers, fixed-time extraction, lexical gating, a small nonlexical probe, and manual link inspection. Model-only same-family accuracy rises to **74.6%**; a lexical top-20/model reranker reaches **81.2%**, compared with **79.3%** for TF–IDF. The paired uncertainty interval includes zero, so the reranker gain is exploratory.

A separate [broad unsupervised graph](BROAD_GENEALOGY.md) now samples **17,014 posts** without phrase labels and retains **1,938** sufficiently similar earlier-post links. It surfaces recognizable templates alongside style and spam artifacts. This is the genealogy exploration; the 485-post regex exercise is only a diagnostic benchmark.

## 4. Learned time geometry

The encoder is `e(τ)=Wτ+b`. For any checkpoint, the centered embeddings have rank at most one **by architecture**: PCA finds 100% of their variance in the first component and Isomap cannot discover a nontrivial curve. At 1B tokens, `||W||₂=0.6695`, `||b||₂=0.6102`, and the second/first singular-value ratio across 101 sampled time coordinates is about **1.9×10⁻⁷**. The 2023 midpoint is `τ≈1.2494`, outside the training range. The more meaningful learned geometry lies in the model's *response* to time, not the linear embedding path itself; the next-token divergences above measure one part of that response.

## 5. Training-time emergence

We repeated the same 2016-versus-2023 first-token probes at four immutable checkpoints. Values are Jensen–Shannon divergence in nats; they reflect *time sensitivity*, not necessarily historically correct outputs.

| Tokens seen | `bro really thought` | `nobody:` | `when you realize` | `||W||₂` |
| ---: | ---: | ---: | ---: | ---: |
| 100M | 0.0461 | 0.0863 | 0.0337 | 0.5492 |
| 250M | 0.0340 | 0.1560 | 0.0323 | 0.5647 |
| 500M | 0.0264 | 0.0574 | 0.0131 | 0.6373 |
| 1B | 0.0173 | 0.0962 | 0.0247 | 0.6695 |

Temporal sensitivity is present at 100M but **not monotonic** across these probes; stronger sensitivity cannot stand in for better forecasting. The encoder norm increases, while its PCA geometry remains a straight line. A rigorous emergence curve would score the same held-out windows at every checkpoint and test a larger, prespecified prompt set. See `temporal_sensitivity.png` and `summary.json`.

## Takeaways for the later paper/demo

The defensible quantitative result is a **small paired 2022 forecasting benefit** from extrapolated time; the visually legible result is that `nobody:` shifts toward a meme dialogue format under later timestamps. The linear embedding trajectory itself is mathematically trivial, and the current genealogy graph is weaker than a simple lexical baseline. The website can show the graph as an explicitly exploratory similarity map, while the main empirical claim should center on the paired held-out comparison and clearly label 2023 as extrapolation rather than verified prediction.
