"""Serve live batch rollouts, local artifacts, and the built visualizer frontend."""

import argparse
import json
import os
from pathlib import Path

import uvicorn

from cs5478_project.defaults import EPISODE_FRAMES, MAX_DYNAMIC_ACTORS, MAX_STATIC_OBJECTS


def main(argv: list[str] | None = None) -> None:
    """Start the simulation API; Vite can proxy it during frontend development."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--artifact-root",
        action="append",
        type=Path,
        help="Directory of replay JSON or normalized .pt scenes; repeat for multiple roots.",
    )
    parser.add_argument(
        "--demo",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include synthetic fixtures for debugging (default: disabled).",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--compile", action="store_true", dest="compile_step")
    parser.add_argument(
        "--reload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reload the development server when Python source changes (default: enabled).",
    )
    parser.add_argument(
        "--max-actors",
        type=int,
        default=MAX_DYNAMIC_ACTORS,
        help="Dynamic actor capacity, including ego.",
    )
    parser.add_argument("--max-static-objects", type=int, default=MAX_STATIC_OBJECTS)
    parser.add_argument("--max-frames", type=int, default=EPISODE_FRAMES)
    parser.add_argument("--max-map-segments", type=int, default=2048)
    parser.add_argument("--max-batch-size", type=int, default=256)
    parser.add_argument(
        "--web-dir", type=Path, default=Path("web/dist"), help="Built frontend directory."
    )
    args = parser.parse_args(argv)
    if args.max_map_segments < 1:
        parser.error("--max-map-segments must be positive")
    os.environ["CS5478_VISUALIZER_OPTIONS"] = json.dumps(
        {
            "artifact_roots": [str(path.resolve()) for path in args.artifact_root]
            if args.artifact_root
            else None,
            "include_demo": args.demo,
            "device": args.device,
            "compile_step": args.compile_step,
            "max_actors": args.max_actors,
            "max_static_objects": args.max_static_objects,
            "max_frames": args.max_frames,
            "max_map_segments": args.max_map_segments,
            "max_batch_size": args.max_batch_size,
            "web_dir": str(args.web_dir.resolve()),
        }
    )
    if (args.web_dir / "index.html").is_file():
        print(f"Visualizer: http://{args.host}:{args.port}", flush=True)
    else:
        print("Frontend build not found. Use Vite or build it with: cd web && npm run build")
    uvicorn.run(
        "cs5478_project.server.runtime:create_visualizer_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        reload_dirs=[str(Path(__file__).resolve().parents[1])] if args.reload else None,
    )


if __name__ == "__main__":
    main()
