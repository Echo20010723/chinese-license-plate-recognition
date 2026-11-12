import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_missing_explicit_dataset_is_respected_without_breaking_import(tmp_path):
    missing = tmp_path / "missing"
    env = dict(os.environ, CCPD_DATA_ROOT=str(missing))
    code = "import config; print(config.DATA_ROOT); config.require_dataset()"
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                            capture_output=True, text=True)
    assert result.stdout.strip() == str(missing)
    assert result.returncode != 0
    assert "Set CCPD_DATA_ROOT" in result.stderr
    assert not missing.exists()
