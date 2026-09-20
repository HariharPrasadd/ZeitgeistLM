# Broad, model-derived genealogy: first pass

This graph starts from a broad sample of cleaned Reddit submissions. **No phrase regex or family label selects its posts or links.** The earlier 485-post phrase benchmark remains a separate diagnostic.

## Method

The CPU sampler scanned all available cleaned submission months from 2011 through August 2022. It kept up to 150 stable-hash-selected posts per year and subreddit, with 20–500 characters and at least four whitespace-separated words. It removed exact text repeats across the assembled sample. This gave **17,014 posts** from **139 monthly shards**. Stratification gives smaller communities and early years representation, but it is not a prevalence-weighted sample of Reddit. It samples submissions only; comments are outside this first graph. The small local [sample report](broad_genealogy_sample_report.json) records counts. Full sampled text remains on the Modal Volume.

An A100 extracted mean-pooled content-token states from transformer block 10 of the stable **1,000,079,360-token checkpoint**. All texts were represented at the same 2018-07-01 temporal coordinate, so date does not directly change the text vector. The model input omitted subreddit markers for this analysis. For every post, exact nearest-neighbor search considered only strictly earlier posts and scored cosine similarity minus **0.02 per elapsed year**. Exact cross-year text repeats were already removed. A link was retained only if its raw cosine was at least **0.85**. This floor is exploratory, not a calibrated probability of common ancestry. The 99.9th percentile of 100,000 random-pair cosines was only 0.756; using that alone would be too permissive after searching thousands of candidates for each post.

The full sample, float32 embeddings, links, and summaries are in `/data/analysis/broad_genealogy/` on the `zeitgeistlm-data` Modal Volume. The initial permissive graph is preserved there as `edges_initial.json` and `summary_initial.json`. Code: [broad_genealogy.py](broad_genealogy.py) and [broad_genealogy_modal.py](broad_genealogy_modal.py). Reproduce with `modal run analysis/broad_genealogy_modal.py --phase all` from the repository root. Each annual CPU sample is saved separately, so failed runs can be resumed by `--phase collect`, `--phase graph`, or `--phase refine` as appropriate.

## First result

The stricter graph retains **1,938 links**, or **11.4%** of possible child posts; the others remain unlinked. It has **221 connected components of at least three posts**. **34.2%** of retained links cross subreddits. The largest components have 156, 110, 63, 50, and 46 nodes. The local [summary](broad_genealogy_summary.json) contains their dates, community counts, early posts, hubs, and 30 highest-similarity edges.

Several components recover recognizable *forms* without regex selection: an “oof/ouch/owie” family in r/bonehurtingjuice (110 nodes, 2017–2022), “le … has arrived” in r/dogelore (46 nodes, 2018–2022), and variants of “It do be like that” across meme communities (28 nodes, 2018–2022). There are also separate starter-pack and “invest in this template” components. These are model-derived groupings, though shared literal wording plainly contributes to their similarity.

The graph also shows failure modes. Its largest component (156 nodes) groups heavily decorated Unicode text from r/surrealmemes by **visual/encoding style**, even when topics differ. Other components contain generic copypasta, all-caps posts, or spam-like advice headlines. A near-duplicate such as “It be like that sometimes.” versus “It be like that sometimes!” survives exact deduplication and can yield cosine above 0.99. A high similarity score can reflect formatting, repeated template words, or near-copying rather than semantic mutation. A connected component is therefore a candidate motif, not a demonstrated cultural lineage.

## Next controls

Before claiming branching, persistence, or migration, audit representative edges within each displayed component and compare them with lexical similarity. A useful follow-up is to cap near-duplicate and style-dominated components, then test whether meaningful links survive across words and subreddits. Because this is an exploratory graph with a chosen threshold and no labeled descent data, we should present it as a **time-constrained cultural similarity map**.

A reviewed, spam-trimmed, library-ready display subgraph is documented in [FINAL_GENEALOGY.md](FINAL_GENEALOGY.md).
