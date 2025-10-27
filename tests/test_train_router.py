import os, sys, stat
from pathlib import Path
import subprocess
import textwrap

def test_train_router_list_and_dry_run(tmp_path: Path):
    root = tmp_path
    # Create legacy script at expected location
    script = root / "train" / "train_kamma_4d_nofifo_random_RLIL.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("#!/usr/bin/env python\nprint('legacy train ok')\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)

    base_args = [
        sys.executable, "-m", "options_sim.cli.train",
        "--algorithm", "kamma",
        "--action-dim", "4d",
        "--fifo", "nofifo",
        "--curriculum", "random",
        "--learning-strategy", "rlil",
        "--root", str(root),
    ]

    # --list should print candidates and not execute
    r1 = subprocess.run(base_args + ["--list"], capture_output=True, text=True)
    assert r1.returncode == 0
    assert "train_kamma_4d_nofifo_random_RLIL.py" in r1.stdout

    # --dry-run should resolve and not execute
    r2 = subprocess.run(base_args + ["--dry-run"], capture_output=True, text=True)
    assert r2.returncode == 0
    assert "Resolved to:" in r2.stdout
