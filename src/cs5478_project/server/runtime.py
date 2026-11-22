"""Importable Uvicorn factory shared by normal startup and reload worker processes."""

import json
import os
from pathlib import Path

from fastapi.staticfiles import StaticFiles

from cs5478_project.server.app import create_app


def create_visualizer_app():
    """Reconstruct the CLI configuration in each fresh Uvicorn worker."""
    options = json.loads(os.environ.get("CS5478_VISUALIZER_OPTIONS", "{}"))
    web_dir = Path(options.pop("web_dir", "web/dist"))
    if options.get("artifact_roots") is not None:
        options["artifact_roots"] = [Path(path) for path in options["artifact_roots"]]
    app = create_app(**options)
    if (web_dir / "index.html").is_file():
        app.mount("/", StaticFiles(directory=web_dir, html=True), name="frontend")
    return app
