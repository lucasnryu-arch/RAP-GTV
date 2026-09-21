from pathlib import Path
import subprocess
import sys


def test_minimal_example_runs() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "examples" / "minimal_example.py")],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "convergence=True" in completed.stdout
    assert "nodes=36" in completed.stdout
