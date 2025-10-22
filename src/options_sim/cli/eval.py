from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import Dict, Tuple, List

# (algorithm, action_dim) -> relative eval script path
EVAL_MAP: Dict[Tuple[str, str], str] = {
    ("kaddpg", "1d"): "eval/eval_kaddpg_1d.py",
    ("kaddpg", "2d"): "eval/eval_kaddpg_2d.py",
    ("kamma",  "4d"): "eval/eval_kamma_4d.py",
}

VALID_FIFO = ("fifo", "nofifo")

def _build_argparser() -> argparse.ArgumentParser:
    epilog = r"""
Examples:

  # Evaluate a KAMMA 4D model (train seed 1) at step 500k with eval seed 777.
  options-sim-eval --algorithm kamma --action-dim 4d \
    --fifo nofifo --curriculum random --strategy RLIL \
    --train-seed 1 --seed 777 --checkpoint-step 500000 \
    --root /workspace/Options_In_Simulation

  # Use an explicit checkpoint path (overrides step-based construction):
  options-sim-eval -a kaddpg -d 2d --fifo fifo --curriculum random --strategy RLIL \
    --train-seed 42 --seed 42 \
    --checkpoint-path /abs/path/to/agent_1000000.pt \
    -- --episodes 50 --metrics success_rate
"""
    p = argparse.ArgumentParser(
        prog="options-sim-eval",
        description="Route to a legacy eval script and build checkpoint path consistent with TRAIN runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )

    # which eval script
    p.add_argument("--algorithm", "-a", required=True, choices=["kaddpg", "kamma"])
    p.add_argument("--action-dim", "-d", required=True, choices=["1d", "2d", "4d"])

    # train naming knobs (MUST match train CLI to reconstruct the run dir)
    p.add_argument("--fifo", "-f", required=True, choices=list(VALID_FIFO))
    p.add_argument("--curriculum", "-c", required=True)
    p.add_argument("--strategy", "-s", required=True, help="e.g., RLIL (case-sensitive to match TRAIN)")

    # seeds: train seed selects the folder; eval seed sets RNG/name
    p.add_argument("--train-seed", type=int, required=True, help="Seed used during TRAINING (to pick the right folder).")
    p.add_argument("--seed", type=int, required=True, help="Eval RNG seed (and eval experiment name).")

    # checkpoint resolution
    group = p.add_mutually_exclusive_group()
    group.add_argument("--checkpoint-step", type=int, default=None,
                       help="Checkpoint step number (agent_<step>.pt).")
    group.add_argument("--checkpoint-path", type=str, default=None,
                       help="Explicit checkpoint path; overrides --checkpoint-step.")

    # default root mirrors your environment
    p.add_argument("--root", default="/workspace/Options_In_Simulation",
                   help="Repo root (default: /workspace/Options_In_Simulation).")

    p.add_argument("--dry-run", action="store_true", help="Print resolved script/args and exit.")
    return p

def _split_argv(argv: List[str]) -> tuple[list[str], list[str]]:
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1 :]
    return argv, []

def _resolve_eval_script(algorithm: str, action_dim: str, root: Path) -> Path:
    key = (algorithm, action_dim)
    if key not in EVAL_MAP:
        raise SystemExit(f"[ERROR] No eval script mapping for {key}. Valid: {list(EVAL_MAP.keys())}")
    script_path = (root / EVAL_MAP[key]).resolve()
    if not script_path.exists():
        raise SystemExit(f"[ERROR] Eval script not found: {script_path}")
    return script_path

def _run_name(algo: str, action_dim: str, fifo: str, curriculum: str, strategy: str) -> str:
    # must mirror TRAIN exactly; your examples use lowercase except strategy often 'RLIL'
    return f"{algo}_{action_dim}_{fifo}_{curriculum}_{strategy}"

def _checkpoint_from_train(root: Path, run: str, train_seed: int, step: int) -> Path:
    # /my_runs/{run}/{run}_s{seed}/checkpoints/agent_{step}.pt
    leaf = f"{run}_s{train_seed}"
    return (root / "my_runs" / run / leaf / "checkpoints" / f"agent_{step}.pt").resolve()

def main(argv: List[str] | None = None) -> int:
    router_argv, legacy_argv = _split_argv(list(argv or sys.argv[1:]))
    ap = _build_argparser()
    args = ap.parse_args(router_argv)

    root = Path(args.root).resolve()
    script_path = _resolve_eval_script(args.algorithm, args.action_dim, root)

    run = _run_name(args.algorithm, args.action_dim, args.fifo, args.curriculum, args.strategy)

    # Eval experiment naming (kept simple)
    eval_name = f"{run}_EVAL_s{args.seed}"

    # Resolve checkpoint path
    if args.checkpoint_path:
        ckpt = Path(args.checkpoint_path).resolve()
    else:
        if args.checkpoint_step is None:
            raise SystemExit("[ERROR] Either --checkpoint-path or --checkpoint-step is required.")
        ckpt = _checkpoint_from_train(root, run, args.train_seed, args.checkpoint_step)

    if not ckpt.exists():
        parent = ckpt.parent
        msg = f"[ERROR] Checkpoint not found: {ckpt}"
        if parent.exists():
            found = sorted(p.name for p in parent.glob("agent_*.pt"))
            msg += f"\n[INFO] Available under {parent}:\n  " + "\n  ".join(found or ["<none>"])
        raise SystemExit(msg)

    forwarded = [
        "--seed", str(args.seed),
        "--experiment-name", eval_name,
        "--checkpoint", str(ckpt),
    ]
    if legacy_argv:
        forwarded += legacy_argv  # keep custom flags after --

    cmd = [sys.executable, str(script_path), *forwarded]

    # Fallback envs if legacy script doesn't parse these yet
    env = os.environ.copy()
    env["EVAL_SEED"] = str(args.seed)
    env["EVAL_CHECKPOINT"] = str(ckpt)

    print("[INFO] Legacy eval script:", script_path)
    print("[INFO] Train run name    :", run)
    print("[INFO] Eval name         :", eval_name)
    print("[INFO] Checkpoint        :", ckpt)
    print("[INFO] Exec              :", " ".join(map(str, cmd)))
    if args.dry_run:
        return 0

    return subprocess.call(cmd, env=env)

if __name__ == "__main__":
    raise SystemExit(main())
