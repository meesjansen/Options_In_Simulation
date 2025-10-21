# src/options_sim/cli/train.py
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import os
from typing import List, Tuple


NORMALIZE = {
    # algorithm
    "kamma": "kamma",
    "kaddpg": "kaddpg",
    # action dims
    "1d": "1d",
    "2d": "2d",
    "4d": "4d",
    # fifo
    "fifo": "fifo",
    "nofifo": "nofifo",
    # curriculum
    "random": "random",
    "gv": "gv",
    "bd": "bd",
    # learning strategy
    "rlil": "rlil",
    "il": "il",
}


def _build_argparser() -> argparse.ArgumentParser:
    epilog = r"""
Examples:
  # Router-only flags:
  options-sim-train --algorithm kaddpg --action-dim 1d --fifo nofifo --curriculum random --learning-strategy rlil --root .

  # Pass extra args to the legacy script AFTER '--':
  options-sim-train --algorithm kaddpg --action-dim 1d --fifo nofifo --curriculum random --learning-strategy rlil --root . -- \
      --stiffness 600 --damping 80
"""
    p = argparse.ArgumentParser(
        prog="options-sim-train",
        description="Route to a legacy training script; optionally pass extra args to it after '--'.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    p.add_argument("--algorithm", required=True, choices=["kamma", "kaddpg"])
    p.add_argument("--action-dim", required=True, choices=["1d", "2d", "4d"])
    p.add_argument("--fifo", required=True, choices=["fifo", "nofifo"])
    p.add_argument("--curriculum", required=True, choices=["random", "gv", "bd"])
    p.add_argument("--learning-strategy", required=True, choices=["rlil", "il"])

    p.add_argument("--dry-run", action="store_true", help="Print the resolved script and exit.")
    p.add_argument("--list", action="store_true", help="List candidate legacy scripts with scores.")

    p.add_argument(
        "--root",
        default=".",
        help="Path to the repo root where legacy scripts live (default: current directory).",
    )
    p.add_argument(
        "--map",
        default="scripts/legacy_map.json",
        help="Optional explicit mapping file (relative to --root).",
    )
    p.add_argument(
        "--search-dirs",
        default="train,legacy,scripts,experiments,.",
        help="Comma-separated dirs (relative to --root) to search for train_*.py",
    )
    return p

def _split_argv(argv: List[str]) -> tuple[list[str], list[str]]:
    """Split argv on '--'. Left -> router argparse; right -> legacy script."""
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1 :]
    return argv, []

def _tuple_key(algorithm: str, action_dim: str, fifo: str, curriculum: str, learning_strategy: str) -> str:
    return "::".join([algorithm, action_dim, fifo, curriculum, learning_strategy])

def _gather_candidates(root: Path, search_dirs: List[str]) -> List[Path]:
    files: List[Path] = []
    for d in search_dirs:
        p = (root / d).resolve()
        if not p.exists():
            continue
        for file in p.rglob("*.py"):
            name = file.name.lower()
            if name.startswith("train_") or re.search(r"\btrain\b", name):
                files.append(file)
    return files

def _score_file(path: Path, tokens: List[str]) -> float:
    name = path.name.lower()
    score = 0.0
    for t in tokens:
        if t in name:
            score += 1.0
    if score == float(len(tokens)):
        score += 1.0  # perfect-match bonus
    if name.startswith("train_"):
        score += 0.5
    return score

def _load_override_map(map_path: Path) -> dict:
    if map_path.exists():
        try:
            return json.loads(map_path.read_text())
        except Exception as e:
            print(f"[WARN] Failed to parse {map_path}: {e}", file=sys.stderr)
    return {}

def _resolve(
    root: Path,
    algorithm: str,
    action_dim: str,
    fifo: str,
    curriculum: str,
    learning_strategy: str,
    map_rel: str,
    search_rel: str,
    list_only: bool = False,
) -> Tuple[Path | None, List[str], List[Tuple[float, Path]]]:
    tokens = [
        NORMALIZE[algorithm],
        NORMALIZE[action_dim],
        NORMALIZE[fifo],
        NORMALIZE[curriculum],
        NORMALIZE[learning_strategy],
    ]

    # 1) explicit mapping wins
    mapping = _load_override_map((root / map_rel).resolve())
    key = _tuple_key(algorithm, action_dim, fifo, curriculum, learning_strategy)
    if key in mapping:
        p = (root / mapping[key]).resolve()
        return (p if p.exists() else None), tokens, []

    # 2) otherwise, filename token scoring
    search_dirs = [s.strip() for s in search_rel.split(",") if s.strip()]
    candidates = _gather_candidates(root, search_dirs)
    scored = [(_score_file(p, tokens), p) for p in candidates]
    scored.sort(reverse=True, key=lambda x: x[0])

    if list_only:
        return None, tokens, scored

    if scored and scored[0][0] >= 6.0:  # guard against random train*.py
        return scored[0][1], tokens, scored
    return None, tokens, scored

def _run_legacy(script_path: Path, legacy_argv: list[str]) -> int:
    """Invoke the legacy script in a subprocess, passing only legacy_argv."""
    cmd = [sys.executable, str(script_path), *legacy_argv]
    return subprocess.call(cmd, env=dict(**os.environ))  # inherit env (ASSETS_DIR, etc.)

def main(argv: List[str] | None = None) -> int:

    # Split arguments at '--'
    argv = list(argv or sys.argv[1:])
    router_argv, legacy_argv = _split_argv(argv)

    # Parse router args
    ap = _build_argparser()
    args = ap.parse_args(router_argv)

    # Resolve legacy script
    root = Path(args.root).resolve()
    script_path, tokens, scored = _resolve(
        root=root,
        algorithm=args.algorithm,
        action_dim=args.action_dim,
        fifo=args.fifo,
        curriculum=args.curriculum,
        learning_strategy=args.learning_strategy,
        map_rel=args.map,
        search_rel=args.search_dirs,
        list_only=args.list,
    )

    if args.list:
        print(f"Tokens: {tokens}")
        for sc, p in scored[:20]:
            print(f"{sc:>4.1f}  {p}")
        return 0

    if script_path is None:
        print("[ERROR] Could not resolve a legacy training script.", file=sys.stderr)
        print(f"  tokens: {tokens}", file=sys.stderr)
        print("Tip: add an explicit mapping in --map (default scripts/legacy_map.json).", file=sys.stderr)
        example = {
            _tuple_key("kamma", "4d", "fifo", "gv", "rlil"): "train/train_kamma_4d_fifo_gv_rlil.py"
        }
        print(json.dumps(example, indent=2), file=sys.stderr)
        return 2

    print(f"[INFO] Resolved to: {script_path}  (tokens matched: {tokens})")
    if args.dry_run:
        return 0

    # Run legacy, passing only post-'--' args
    rc = _run_legacy(script_path, legacy_argv)
    return rc

if __name__ == "__main__":
    raise SystemExit(main())
