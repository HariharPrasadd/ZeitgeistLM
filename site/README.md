# ZeitgeistLM chat

A minimal React interface for dated text generation. The page uses Geist, plain
CSS, and no component library. The slider covers January 2008 through December
2024 and marks dates outside the 2011–2020 training interval as extrapolation.
Each prompt is completed independently by the final 2.967B-token checkpoint.

Run `npm install` and `npm run dev` locally. `VITE_MODAL_GENERATE_URL` points to
the separate Modal GPU endpoint. Build with `npm run build`, then deploy the
page with `modal deploy web_modal.py`. Deploy model changes with
`modal deploy inference_modal.py`. The GPU container scales to zero when idle.
