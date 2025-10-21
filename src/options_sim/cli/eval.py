from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys
from typing import Dict, Tuple, List

# Hard mapping: (algorithm, action_dim) -> eval script path and base experiment name
EVAL_MAP: Dict[Tuple[str, str], Tuple[str, str]] = {
    ("kaddpg", "1d"): ("eval/eval_kaddpg_1d.py", "KA-DDPG_1D_g1"),
    ("kaddpg", "2d"): ("eval/eval_kaddpg_2d.py", "KA-DDPG_2D_g1"),
    ("kamma",  "4d"): ("eval/eval_kamma_4d.py",  "KAMMA_4D_g1"),
}

def _build_argparser() -> argparse.ArgumentParser:
    epilog = r"""
Examples:

  # Evaluate a KA-DDPG 1D model trained with seed 123 at step 500k.
  options-sim-eval --algorithm kaddpg --action-dim 1d \
    --train-seed 123 --seed 777 --checkpoint-step 500000

  # Use an explicit checkpoint path (overrides step-based construction):
  options-sim-eval --algorithm kaddpg --action-dim 2d \
    --seed 42 --train-seed 42 \
    --checkpoint-path ./my_runs/KA-DDPG_2D_g1_s42/KA-DDPG_2D_g1_s42/checkpoints/agent_1000000.pt \
    -- --episodes 50 --metrics success_rate
"""
    p = argparse.ArgumentParser(
        prog="options-sim-eval",
        description="Route to a legacy EVAL script and build checkpoint path from train naming.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )

    # choose which eval script to run
    p.add_argument("--algorithm", required=True, choices=["kaddpg", "kamma"])
    p.add_argument("--action-dim", required=True, choices=["1d", "2d", "4d"])

    # seeds
    p.add_argument("--train-seed", type=int, required=True,
                   help="Seed used during TRAINING (to find the right checkpoint folder).")
    p.add_argument("--seed", type=int, required=True,
                   help="Seed to use for this EVAL run (sets eval RNG and eval experiment name).")

    # checkpoint resolution
    group = p.add_mutually_exclusive_group()
    group.add_argument("--checkpoint-step", type=int, default=500000,
                       help="Checkpoint step number (builds agent_<step>.pt under my_runs).")
    group.add_argument("--checkpoint-path",
                       help="Explicit checkpoint path; if provided, overrides --checkpoint-step.")

    # roots / dirs
    p.add_argument("--root", default=".",
                   help="Repo root. my_runs is resolved beneath this (default: current directory).")

    # dry-run & passthrough display
    p.add_argument("--dry-run", action="store_true", help="Print resolved script/args and exit.")
    return p

def _split_argv(argv: List[str]) -> tuple[list[str], list[str]]:
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1 :]
    return argv, []

def _resolve_eval_script(algorithm: str, action_dim: str, root: Path) -> Tuple[Path, str]:
    key = (algorithm, action_dim)
    if key not in EVAL_MAP:
        raise SystemExit(f"[ERROR] No eval script mapping for {key}. "
                         f"Valid: {list(EVAL_MAP.keys())}")
    rel_path, base_name = EVAL_MAP[key]
    script_path = (root / rel_path).resolve()
    if not script_path.exists():
        raise SystemExit(f"[ERROR] Eval script not found: {script_path}")
    return script_path, base_name

def _build_checkpoint_path(root: Path, base_name: str, train_seed: int,
                           checkpoint_step: int) -> Path:
    # Train convention: <base>_s{train_seed} twice in path, then checkpoints/agent_<step>.pt
    exp = f"{base_name}_s{train_seed}"
    return (root / "my_runs" / exp / exp / "checkpoints" / f"agent_{checkpoint_step}.pt").resolve()

def main(argv: List[str] | None = None) -> int:
    router_argv, legacy_argv = _split_argv(list(argv or sys.argv[1:]))
    ap = _build_argparser()
    args = ap.parse_args(router_argv)

    root = Path(args.root).resolve()
    script_path, base_name = _resolve_eval_script(args.algorithm, args.action_dim, root)

    # Eval experiment naming: <base>_EVAL_s<eval-seed>
    eval_name = f"{base_name}_EVAL_s{args.seed}"

    # Checkpoint: explicit path wins; otherwise construct from training naming convention
    if args.checkpoint_path:
        ckpt = Path(args.checkpoint_path).resolve()
    else:
        ckpt = _build_checkpoint_path(root, base_name, args.train_seed, args.checkpoint_step)

    # Compose argv forwarded to the legacy eval script
    forwarded = [
        "--seed", str(args.seed),                    # eval RNG
        "--experiment-name", eval_name,              # eval experiment name
        "--checkpoint", str(ckpt),                   # path to trained agent
    ] + legacy_argv  # preserve any extra flags after '--'

    cmd = [sys.executable, str(script_path), *forwarded]
    print("[INFO] Resolved eval:", script_path)
    print("[INFO] Eval name    :", eval_name)
    print("[INFO] Checkpoint   :", ckpt)
    print("[INFO] Exec         :", " ".join(map(str, cmd)))
    if args.dry_run:
        return 0

    return subprocess.call(cmd)

if __name__ == "__main__":
    raise SystemExit(main())
