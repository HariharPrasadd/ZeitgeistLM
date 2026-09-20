# Genealogy follow-up: representation and lexical controls

## Setup

We reused the unchanged 485 real submission titles/texts, nine phrase-family labels, and the stable 1,000,079,360-token checkpoint from [FINDINGS.md](FINDINGS.md). Every predicted parent is strictly earlier than its child. The labels come from phrase-pattern retrieval, so same-family accuracy strongly rewards word overlap; it is a diagnostic of phrase retrieval, not ground truth for cultural descent. The 2021–2022 subset has 144 links and was already examined in the first analysis, so it is **not an untouched test** for this follow-up.

`genealogy_v2.py` captures block 4, 7, 10, and 12 states plus final layer norm. It compares the original last-content-token representation with the mean of all content-token states. Position zero, the continuous-time pseudo-token, is excluded from pooling. These extraction inputs contain no subreddit marker. We encode every title at either its real timestamp or a shared 2018-07-01 timestamp; real chronology is always retained when choosing an earlier parent. The same word/bigram TF–IDF baseline is recomputed on exactly these texts. The time penalty is 0.03 cosine units per year. The lexical gate retrieves the top 20 earlier TF–IDF candidates and reranks them by model cosine minus that penalty. The hybrid directly mixes scaled semantic cosine, lexical cosine, and the penalty.

Reproduce with `modal run analysis/genealogy_v2_modal.py`; the result is [genealogy_v2_results.json](genealogy_v2_results.json). Large model files remain on the Modal Volume.

## What changed

| Earlier-parent method | Same-family, all 484 | Through 2020, 340 | 2021–2022, 144 |
| --- | ---: | ---: | ---: |
| Original final-layer, final token, real time | 53.1% | 54.1% | 50.7% |
| Block 7, mean content tokens, fixed time | 71.9% | 70.6% | 75.0% |
| Block 10, mean content tokens, fixed time | **74.6%** | 74.1% | 75.7% |
| Final norm, mean content tokens, fixed time | 73.6% | 74.7% | 70.8% |
| TF–IDF word/bigram | 79.3% | 78.2% | 81.9% |
| TF–IDF top 20 → final-norm mean rerank | **81.2%** | **79.4%** | 85.4% |
| TF–IDF top 20 → block-7 mean rerank | **81.2%** | 78.2% | **88.2%** |

Mean pooling changes the result far more than choosing a different layer. Block 10 is the strongest model-only representation tested. Fixing the time coordinate slightly improves final-norm mean accuracy (73.6% versus 72.3% at real timestamps). It also prevents date information from doing some of the neighbor selection: for block-10 mean vectors, the unpenalized mean parent gap is 1.81 years at fixed time versus 0.97 years at real time. Adding an explicit time penalty to fixed-time block-10 vectors reduces the gap to 0.66 years, while same-family accuracy changes from 74.6% to 74.4%. This supports separating text representation from chronological preference.

The top-20 reranker gives a small numerical improvement over TF–IDF. Selecting final-norm mean by its through-2020 score gives 85.4% versus 81.9% on 2021–2022: 17 links fixed and 12 broken relative to TF–IDF. A paired child bootstrap gives a 95% interval of **−4.2 to +11.1 percentage points** for the 3.5-point gain; an exact discordant-pair test gives p=0.46. The block-7 reranker gives a larger later-year gain of 6.3 points, but its corresponding interval is **−0.7 to +13.2 points**, p=0.14. These are exploratory choices among many inspected representations, and neither margin establishes a reliable improvement. The direct weighted hybrid did not beat the lexical gate.

## Nonlexical probe and link inspection

On ten **hand-authored** anchor/literal/paraphrase/unrelated groups, TF–IDF ranks the paraphrase above the unrelated text in 3/10 cases. Fixed-time block-7 mean vectors do so in 8/10; block-10 and final-norm mean do so in 7/10. TF–IDF ranks literal mutations correctly in 10/10. This small synthetic diagnostic suggests the model can capture some relationships after word overlap disappears, but it is not a measured score on real meme mutations or a curated benchmark.

We inspected the 30 highest-cosine cross-family block-10 links with TF–IDF similarity below 0.12. Some are culturally plausible associations: a Vine meme-economy discussion links to advice about Pepe/Harambe/Doge meme investing; “rare The_Donald pepe” links to “Rare Sniff doge” using the rare-meme template; a Harambe anniversary/reinvestment post links to a discussion of buying or selling rare Pepes. Other links arise from shared emoticons, emoji, all caps, Reddit posting boilerplate, or plainly unrelated topics. Because the sample is selected by the model's own high scores and the judgment is subjective, these examples are illustrations, not a precision estimate. Cross-family links should not be presented as discovered influence.

## Time likelihood and next evaluation

Scoring these texts at their real dates rather than fixed 2018 lowers mean next-token loss by only **0.00490 nats/token** (4.51477 versus 4.51967), with mixed signs across phrase families. A proposed ancestor score based on `P(descendant text | descendant time) − P(descendant text | ancestor time)` depends on the ancestor's *date* but not its *text*. It therefore cannot, by itself, distinguish two candidate ancestors from the same date or demonstrate descent. A pairwise mutation score would need an explicit conditioning mechanism or a carefully defined contrastive task.

The next rigorous test is a blinded, human-labeled set of real literal variants, nonlexical paraphrases, and unrelated controls sampled independently of the phrase-pattern retrieval. Evaluate TF–IDF, model-only, and hybrid methods on that set, with time and subreddit matched negatives. Until then, describe the graph as a chronology-constrained similarity map rather than a cultural family tree.
