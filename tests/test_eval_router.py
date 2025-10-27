import os, sys, stat
from pathlib import Path
import subprocess

def test_eval_router_dry_run(tmp_path: Path):
    root = tmp_path

    # Create eval script that the router expects for (kamma, 4d)
    eval_script = root / "eval" / "eval_kamma_4d.py"
    eval_script.parent.mkdir(parents=True, exist_ok=True)
    eval_script.write_text("#!/usr/bin/env python\nprint('legacy eval ok')\n")
    eval_script.chmod(eval_script.stat().st_mode | stat.S_IEXEC)

    # Create a fake checkpoint at the constructed path:
    # /my_runs/{run}/{run}_s{train_seed}/checkpoints/agent_{step}.pt
    run = "kamma_4d_nofifo_random_RLIL"
    train_seed = 1
    step = 500000
    ckpt = (root / "my_runs" / run / f"{run}_s{train_seed}" / "checkpoints" / f"agent_{step}.pt")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"fake")

    args = [
        sys.executable, "-m", "options_sim.cli.eval",
        "--algorithm", "kamma",
        "--action-dim", "4d",
        "--fifo", "nofifo",
        "--curriculum", "random",
        "--strategy", "RLIL",
        "--train-seed", str(train_seed),
        "--seed", "777",
        "--checkpoint-step", str(step),
        "--root", str(root),
        "--dry-run",
    ]
    r = subprocess.run(args, capture_output=True, text=True)
    assert r.returncode == 0
    assert "Legacy eval script" in r.stdout
    assert "Checkpoint" in r.stdout and "agent_500000.pt" in r.stdout
