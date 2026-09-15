from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_demo_runs_with_no_network_and_no_key(tmp_path: Path) -> None:
    """E8-T2: the first thing a reviewer runs must never fail."""
    proc = subprocess.run(
        [sys.executable, "-c", "from scripts.demo import main; raise SystemExit(main())"],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(Path.cwd()), "HOME": str(tmp_path)},
        cwd=Path.cwd(),
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "sharpe" in out
    assert "2020-06-30" in out, "the point-in-time check did not run"
    assert "no network" in out


def test_demo_does_not_import_network_or_api_clients() -> None:
    source = Path("scripts/demo.py").read_text()
    for forbidden in ("urllib", "requests", "httpx", "anthropic", "openai", "os.environ"):
        assert forbidden not in source, f"demo depends on {forbidden}"
