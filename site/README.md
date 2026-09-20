# ZeitgeistLM research paper website

The site is a single-column research report with left-side section navigation. Its claims and figures come from the fixed-checkpoint outputs in `../analysis/`. React renders the paper; shadcn/ui provides the controls for the forecast figure and the one conditional-generation example. Plus Jakarta Sans is bundled locally.

Run `python build_data.py` after changing analysis results, then `npm install` and `npm run dev`. `npm run build` produces a static `dist/` folder for deployment.

The site bundles only compact figures, sampled public posts, and fixed generation examples. The 1B-token model weights remain on the `zeitgeistlm-data` Modal Volume. `inference_modal.py` deploys a separate GPU web function that loads the checkpoint once per container and responds to bounded text-generation requests. The frontend reads its URL from `VITE_MODAL_GENERATE_URL`; if unavailable, the measured preset examples and every other interaction still work.

Deploy the API with `modal deploy inference_modal.py`. A10 containers scale to zero after two idle minutes, so the first request can take longer. The endpoint is public; requests are capped at 160 characters, 64 prompt tokens, and 64 new tokens, with at most two containers. It should be disabled after the public demo if continued GPU usage is unwanted.

Run `npm run build && modal deploy web_modal.py` to serve the static paper through Modal. The public API URL is a build-time Vite setting in `.env.production`; update it before rebuilding if the API deployment changes.
