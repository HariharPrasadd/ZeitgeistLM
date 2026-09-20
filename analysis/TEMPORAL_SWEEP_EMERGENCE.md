# Likelihood sweep and retrospective emergence test

Both analyses use the immutable 1,000,079,360-token checkpoint trained through 2020. They are exploratory follow-ups: 2022 was already examined in earlier work, so these are not fresh confirmatory tests.

## Continuous likelihood sweep

We reused the same 288 validation and 192 test windows from the paired forecasting analysis. Each fixed 1,024-token target sequence was scored at the midpoints of every year from 2011 through 2023, changing only the continuous time input. The complete 480 × 13 loss matrix is in [temporal_sweep_results.json](temporal_sweep_results.json). No text, weights, or community markers changed between dates.

Average 2021 loss falls from **2.5695 at 2011** to **2.3247 at 2020**, reaches **2.3233 at 2021**, and rises slightly to **2.3259 at 2023**. Average 2022 loss falls from **2.8542 at 2011** to **2.6116 at 2020**, reaches **2.6050 at 2022**, and is **2.6055 at 2023**. Thus the aggregate curves have a broad low-loss region near the observed years. Individual windows are much less precisely dated: 82 of 192 test windows have their lowest value at the 2023 grid endpoint, while 48 select 2022. The curve demonstrates a learned time response, not reliable document dating or meme forecasting.

Reproduce with `modal run analysis/temporal_sweep_modal.py`. The sweep ran on an A100 while token shards stayed on the Modal Volume.

## Emergence test

Candidates were fixed using only 2017–2020 posts from the earlier broad, stratified submission sample. Among 410 eligible two-to-four-word phrases, each had at least three sampled occurrences and **three different natural preceding text spans** from pre-2021 posts. We did not add a subreddit prefix. Candidate selection used a deterministic phrase hash and did not inspect future counts.

Modal CPUs then scanned all cleaned submissions in 2018–August 2022: **17,401,364 documents** in total. A phrase's monthly outcome was counted once per document and normalized by the month's eligible submission count. On an A100, the model scored each phrase as a continuation of each of its three real pre-cutoff prefixes at mid-2018, mid-2019, and mid-2020. The score averages phrase-token log likelihood across contexts; it includes the phrase's first token by predicting it from the last prefix token. The model predictor is the 2020-minus-2018 score. The outcome is 2021–2022 usage-rate growth versus 2019–2020. The baseline is the observed 2019-to-2020 frequency trend.

**Result:** the model score slope has Spearman correlation **0.110** with later growth; the simple pre-cutoff frequency trend has correlation **0.541** on the same 410 phrases. After ranking out that frequency trend, the model's partial rank correlation is about **−0.100**. The top model-score decile has mean future log growth **−0.077**, compared with **−0.150** overall; the top frequency-trend decile has mean growth **+0.161**. The [phrase-level results](emergence_results.json) contain all model scores, counts, and rates.

This does **not** support a headline claim that ZeitgeistLM anticipates meme emergence. The candidates are short overlapping n-grams and many are generic text fragments; the retrospective 2022 outcome has already been examined in other analyses; community composition shifts remain possible despite rate normalization; and these phrase rows are dependent. The result is best described as an exploratory negative benchmark and a reason to avoid selecting a handful of impressive examples post hoc. A stronger future test would lock an independently curated candidate list and matched controls before obtaining new future data.

Reproduce in stages with `modal run analysis/emergence_modal.py --phase select`, then `--phase count`, `--phase score`, and `--phase report`. Candidate lists, monthly counts, and model scores remain on the Modal Volume; only the compact final report is local.
