"""Serve the minimal ZeitgeistLM chat through a lightweight Modal web function."""

from pathlib import Path

import modal


app = modal.App("zeitgeistlm-paper")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("fastapi>=0.115,<1")
    .add_local_dir(Path(__file__).resolve().parent / "dist", remote_path="/app/site")
)


@app.function(image=image, scaledown_window=300)
@modal.asgi_app()
def paper():
    """Return static chat assets without requiring a long-lived server."""
    from fastapi import FastAPI
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    web = FastAPI()
    root = Path("/app/site")
    web.mount("/assets", StaticFiles(directory=root / "assets"), name="assets")

    @web.get("/")
    def index():
        # Revalidate the document on every visit; hashed assets can remain cached.
        return FileResponse(root / "index.html", headers={"Cache-Control": "no-store"})

    @web.get("/favicon.svg")
    def favicon():
        return FileResponse(root / "favicon.svg", media_type="image/svg+xml")

    return web
