# Four final experiments at 2.967B token presentations

All four experiments below use the immutable `final_2967M.pt` checkpoint
(2,967,470,080 token presentations). The training interval ends in 2020.
Validation is 2021; the available 2022 months are retrospective future test
data. The model was not retrained. The old 1B reports remain intact for a
same-sample comparison. The 2022 data had already been inspected in earlier
analyses, so none of these are fresh confirmatory tests.

## 1. Continuous likelihood sweep

The model scored the **same 480 held-out 1,024-token windows** as before: 288
from 2021 and 192 from 2022. Each window was evaluated at 13 annual midpoint
time coordinates, 2011–2023. Only the scalar time input changed.

| Window year | Loss at 2020 | Loss at 2021 | Loss at 2022 | Loss at 2023 | Best aggregate year |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2021 validation | 2.1812 | **2.1745** | 2.1809 | 2.1988 | 2021 |
| 2022 test | 2.4550 | **2.4394** | 2.4429 | 2.4575 | 2021 |

The true 2021 coordinate beats fixed 2020 by 0.0068 nats/token. The true 2022
coordinate beats fixed 2020 by 0.0121 nats/token, but **2021 is slightly better
than 2022 on 2022 text** by 0.0035. Among individual 2022 windows, 131/192
minimize at 2021, 38/192 at 2022, and 6/192 at 2023. The final model has a
stronger and more concentrated time response than the 1B model, but its best
extrapolated date is biased early. This supports meaningful temporal
conditioning, not precise future dating.

Full per-window curves: [final_2967M_sweep.json](final_2967M_sweep.json).

## 2. Emergence detection

The candidate list, three natural pre-2021 contexts per phrase, and full-corpus
frequency counts were fixed in the 1B experiment. We rescored all **410
phrases** at mid-2018, mid-2019, and mid-2020 with the final weights. The
outcome remains 2021–2022 phrase usage-rate growth over 2019–2020.

| Predictor | Spearman correlation with later growth |
| --- | ---: |
| Final model's 2020-minus-2018 log-probability slope | 0.142 |
| Same model score at 1B | 0.110 |
| Observed 2019-to-2020 frequency trend | **0.541** |

The final model improves slightly over 1B, but the elementary frequency trend
is far stronger. After adjusting ranks for that trend, the final model's
partial rank correlation is about **−0.102**. The top model-score decile has
mean future log growth −0.013 versus −0.150 overall; the top frequency-trend
decile reaches +0.161. These short n-grams overlap and often represent generic
phrases. This does **not** establish reliable anticipation of new memes.

Phrase-level report: [final_2967M_emergence.json](final_2967M_emergence.json).

## 3. Meme genealogy

We re-embedded the same **17,014 broadly sampled submissions** with final-model
block-10 mean states at a fixed 2018 time coordinate. The existing artifact
filter removed 506 rows. Exact chronological nearest-neighbor search on the
remaining **16,508** posts, a 0.02 cosine/year time penalty, and a minimum raw
cosine of 0.85 yielded **2,644 links**, versus 1,671 at 1B. The final graph has
325 components of at least three nodes. More links at a fixed cosine threshold
do not prove better semantic ancestry: representation scale and density can
change with training.

The full graph's cross-subreddit link rate is 36.9%, but after manual review of
18 recognizable motif components it is only **5.1%**. Those components include
Dogelore “le ... has arrived,” “oof/ouch/owie,” starter packs, meme-investment
language, “It do be like that,” and the “I don't always ...” format. The
reviewed portable graph has **709 nodes and 691 forward-in-time links**; it is
acyclic. Style-dominated, generic, and crypto-promotion components were left
out of this display graph. Links mean “similar earlier post,” not observed
copying or causal descent.

Local [summary](final_2967M_genealogy.json) and
[portable reviewed graph](final_2967M_curated.json), also exported as
[Cytoscape JSON](final_2967M_curated_cytoscape.json) and
[GraphML](final_2967M_curated.graphml). The full linked graph and embeddings
remain on the Modal Volume at
`/data/analysis/final_2967M/genealogy/`.

## 4. Mutation generation

A prespecified panel of **12 textual prompts** was generated at 2016, 2020,
2022, and extrapolated 2023 with identical sampling seed 42, temperature 0.8,
top-k 50, and 64 new tokens. The median 2016-versus-2023 next-token
Jensen–Shannon divergence was **0.038 nats** (range 0.010–0.123). The
“bro really thought” prompt moves from “it was hilarious” in 2016 toward “it
was a good meme” in 2022–2023; “nobody:” shifts toward “Me:” list formatting.

The raw decoder ends its first document immediately for **7/12 prompts** at
each of 2020, 2022, and 2023. A separate, explicitly constrained run disallows
EOS for the first 16 generated tokens and yields nonempty text for all prompts,
but often produces repetition, URL fragments, or moderator/bot boilerplate.
Thus the model measurably changes continuation probabilities with time, while
the sampled mutations are uneven and should be curated only as examples, not
claimed as reliable forecasts of future memes.

[Unconstrained samples and distribution shifts](final_2967M_mutation.json) ·
[EOS-constrained samples](final_2967M_mutation_constrained.json).

## Overall

The strongest result is the held-out likelihood response to continuous time.
Genealogy gives a useful exploratory graph of recurring formats but remains
lexically dominated and noncausal. Emergence prediction is weak relative to a
simple frequency baseline. Generation is visibly time sensitive for some
prompts, but immediate EOS and boilerplate limit its reliability.
