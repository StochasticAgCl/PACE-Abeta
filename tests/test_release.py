"""Release checks use temporary synthetic files only."""

import importlib.util
from pathlib import Path


path = Path(__file__).resolve().parents[1] / "scripts/check_release.py"
spec = importlib.util.spec_from_file_location("check_release", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_private_table_is_rejected(tmp_path):
    (tmp_path / "measurements.csv").write_text("id,value\nexample,1\n")
    result = module.check(tmp_path)
    assert not result["passed"]
    assert any("Runtime data" in issue for issue in result["issues"])


def test_ignored_runtime_content_is_still_rejected(tmp_path):
    folder = tmp_path / "data"
    folder.mkdir()
    (folder / "values.json").write_text('{"example": 1}')
    assert not module.check(tmp_path)["passed"]


def test_portable_english_source_passes(tmp_path):
    (tmp_path / "example.py").write_text('"""Calculate a mean."""\ndef mean(x):\n    return sum(x) / len(x)\n')
    assert module.check(tmp_path)["passed"]
