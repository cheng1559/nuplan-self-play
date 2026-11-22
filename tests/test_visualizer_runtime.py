"""The importable reload worker preserves CLI configuration and static assets."""

import pytest
from fastapi.testclient import TestClient

from cs5478_project.cli.serve import main
from cs5478_project.server.runtime import create_visualizer_app


@pytest.mark.parametrize("flags,expected_reload", [([], True), (["--no-reload"], False)])
def test_cli_uses_importable_factory_and_preserves_worker_options(
    tmp_path, monkeypatch, flags, expected_reload
):
    captured = {}
    monkeypatch.setattr(
        "cs5478_project.cli.serve.uvicorn.run",
        lambda app, **kwargs: captured.update(app=app, **kwargs),
    )
    monkeypatch.setenv("CS5478_VISUALIZER_OPTIONS", "{}")
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("<html>Reload worker</html>")
    main(
        [
            "--artifact-root",
            str(tmp_path),
            "--device",
            "cpu",
            "--compile",
            "--max-actors",
            "16",
            "--web-dir",
            str(web),
            *flags,
        ]
    )
    assert captured["app"] == "cs5478_project.server.runtime:create_visualizer_app"
    assert captured["factory"] is True
    assert captured["reload"] is expected_reload
    assert bool(captured["reload_dirs"]) == expected_reload
    client = TestClient(create_visualizer_app())
    result = client.get("/api/v1/rollout-sources").json()
    assert result["sources"] == []
    assert result["settings"]["device"] == "cpu"
    assert result["settings"]["compiled"] is True
    assert result["settings"]["max_actors"] == 16
    assert "Reload worker" in client.get("/").text
