import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_public_text.py"
SPEC = importlib.util.spec_from_file_location("check_public_text", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
public_copy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(public_copy)


def test_scan_tracked_checks_text_and_private_paths(monkeypatch, tmp_path: Path):
    private_path = "C:" + "/Users/example/private"
    (tmp_path / "README.md").write_text(private_path, encoding="utf-8")
    monkeypatch.setattr(public_copy, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        public_copy,
        "tracked_paths",
        lambda: ["README.md", "events/private-frame.jpg"],
    )

    hits = public_copy.scan_tracked([])

    assert any("Windows 使用者絕對路徑" in hit for hit in hits)
    assert any("events/private-frame.jpg" in hit for hit in hits)


def test_scan_tracked_skips_binary_content(monkeypatch, tmp_path: Path):
    (tmp_path / "asset.bin").write_bytes(b"\xff\xfe\x00\x01")
    monkeypatch.setattr(public_copy, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(public_copy, "tracked_paths", lambda: ["asset.bin"])

    assert public_copy.scan_tracked([]) == []
