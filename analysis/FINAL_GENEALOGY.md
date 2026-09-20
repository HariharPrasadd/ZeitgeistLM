# Final portable genealogy graph

This is a **directed temporal similarity graph** of real Reddit submissions, ready for D3, Cytoscape.js, Gephi, or NetworkX. The edges are inferred nearest earlier posts, not observed copying or causal descent.

## Files

- [final_genealogy_graph.json](final_genealogy_graph.json): library-neutral `nodes` and `links` arrays. Each node has an ID, full text, display label, timestamp, year, subreddit, component, and post-hoc motif name. Each link goes from earlier `source` to later `target` and records cosine, year gap, and ranking score.
- [final_genealogy_cytoscape.json](final_genealogy_cytoscape.json): importable `elements.nodes[].data` and `elements.edges[].data` for Cytoscape.js.
- [final_genealogy.graphml](final_genealogy.graphml): directed GraphML for Gephi or NetworkX.

For D3, use `const {nodes, links} = await d3.json('final_genealogy_graph.json')`. For Cytoscape.js, use `elements: (await fetch('final_genealogy_cytoscape.json').then(r => r.json())).elements`. Node positions can use `created_utc` on the horizontal axis and `component` or `motif` on the vertical axis; a force layout can run within each component. These files contain the sampled public post text, so display it with appropriate context and escaping.

## Construction and curation

The broad pipeline sampled **17,014** submissions from 2011–August 2022 without phrase labels. The final pass excluded **506** posts: 254 dominated by combining marks, 176 containing URLs, 26 matching obvious bot/spam patterns, 23 dominated by a single repeated word, and 27 punctuation-only near-copies. It recomputed chronological nearest neighbors on the remaining **16,508** embeddings, so excluded posts could not remain as ancestors. The same block-10 mean-pooled representation, fixed 2018 time coordinate, 0.02 cosine/year penalty, and minimum raw cosine of 0.85 produced **1,671** links before display curation.

I then reviewed the components and exported **17 recognizable motif components** containing **381 nodes and 364 links**. Examples include “oof/ouch/owie,” starter packs, several separate Dogelore “le … has arrived” groups, meme-investment language, “It do be like that,” “Thank you, very cool,” and “surgery on a grape.” The component names and shared `motif` fields were added *after* link construction for display; they never influenced retrieval. Components dominated by SEO-like posts, decorative Unicode, generic subreddit headers, or posting boilerplate were excluded from the portable graph. The full cleaned graph, candidate previews, and the pre-filter version remain on the Modal Volume for audit. [final_genealogy_candidates.json](final_genealogy_candidates.json) records the component review sample.

The exported graph is a forest: each node has at most one incoming earlier-parent link. Disconnected components with the same motif label were **not** connected by hand. All 364 links were checked to point strictly forward in time; GraphML was reloaded and checked to be acyclic.

## Limits

The 0.85 threshold and reviewed component selection are exploratory choices, not calibrated evidence of common descent. Many strong links reconstruct literal phrases or formats. The curated display intentionally favors interpretable groups and should not be used to estimate prevalence, semantic-retrieval accuracy, or cross-community migration. Some branches can reflect shared wording without one post influencing another. Use the graph as a browsable hypothesis generator and label links “similar earlier post.”

Rebuild the graph on Modal with `modal run analysis/final_genealogy_modal.py --phase prepare`, then export reviewed component indices with `--phase export --indices ...`. Run `python3 analysis/export_genealogy_formats.py` to regenerate the Cytoscape and GraphML files. The [deployed exporter](https://modal.com/apps/hariharprasadd/main/deployed/zeitgeistlm-final-genealogy) uses the saved Modal Volume graph, so the checkpoint and full corpus need not sit on the laptop.
