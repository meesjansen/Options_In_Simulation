# src/options_sim/cli/artifacts.py
# Generate time-series CSV + PNG from TF event files for the 6 reward components (env0)
# Usage:
#   PYTHONPATH="$PWD/src" /isaac-sim/python.sh -m options_sim.cli.artifacts --run kamma_4d_nofifo_random_RLIL_s1
#   PYTHONPATH="$PWD/src" /isaac-sim/python.sh -m options_sim.cli.artifacts --run /abs/path/to/run_dir
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# TensorBoard (works in Isaac-Sim after: /isaac-sim/python.sh -m pip install --user tensorboard)
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# The six tags (exact matches you showed) plus a couple safe variants
TAG_CANDIDATES = {
    "v_delta":        ["Reward_comp_env0 / v_delta", "Reward_comp_env0/v_delta", "reward_comp_env0/v_delta"],
    "desired_v":      ["Reward_comp_env0 / desired_v", "Reward_comp_env0/desired_v", "reward_comp_env0/desired_v"],
    "current_v":      ["Reward_comp_env0 / current_v", "Reward_comp_env0/current_v", "reward_comp_env0/current_v"],
    "omega_delta":    ["Reward_comp_env0 / omega_delta", "Reward_comp_env0/omega_delta", "reward_comp_env0/omega_delta"],
    "desired_omega":  ["Reward_comp_env0 / desired_omega", "Reward_comp_env0/desired_omega", "reward_comp_env0/desired_omega"],
    "current_omega":  ["Reward_comp_env0 / current_omega", "Reward_comp_env0/current_omega", "reward_comp_env0/current_omega"],
}


def _find_run_dir(run_arg: str, base: Path | None) -> Path:
    p = Path(run_arg)
    if p.exists():
        return p.resolve()
    # try under provided base
    if base:
        cand = (base / run_arg).resolve()
        if cand.exists():
            return cand
    # search under my_runs/**/<run_arg>
    root = Path.cwd() / "my_runs"
    hits = list(root.rglob(run_arg))
    if len(hits) == 1:
        return hits[0].resolve()
    elif len(hits) > 1:
        # pick the one that contains events
        with_events = [h for h in hits if any(h.glob("events.out.tfevents.*"))]
        if len(with_events) == 1:
            return with_events[0].resolve()
        # fall back to the shortest path
        return sorted(hits, key=lambda x: len(str(x)))[0].resolve()
    raise SystemExit(f"[artifacts] Could not locate run directory for '{run_arg}'. "
                     f"Pass a full path or use --base /path/to/parent.")


def _load_events(run_dir: Path) -> EventAccumulator:
    if not run_dir.exists():
        raise SystemExit(f"[artifacts] Run directory not found: {run_dir}")
    if not any(run_dir.glob("events.out.tfevents.*")):
        raise SystemExit(f"[artifacts] No TensorBoard event files found in: {run_dir}")
    ea = EventAccumulator(str(run_dir))
    ea.Reload()
    return ea


def _collect_timeseries(ea: EventAccumulator) -> tuple[list[int], dict[str, list[float]], dict[str, str]]:
    # Build step-indexed table for all six tags
    table: dict[int, dict[str, float]] = {}
    matched: dict[str, str] = {}
    for col, cands in TAG_CANDIDATES.items():
        series = None
        hit = None
        for tag in cands:
            try:
                series = ea.Scalars(tag)
                if series:  # non-empty
                    hit = tag
                    break
            except KeyError:
                continue
        matched[col] = hit  # may be None
        if series:
            for ev in series:
                row = table.setdefault(ev.step, {})
                row[col] = float(ev.value)

    steps = sorted(table.keys())
    cols = list(TAG_CANDIDATES.keys())
    data = {k: [table[s].get(k, float("nan")) for s in steps] for k in cols}
    return steps, data, matched


def _write_csv(run_dir: Path, steps: list[int], data: dict[str, list[float]]) -> Path:
    out = run_dir / "reward_components_env0_timeseries.csv"
    cols = ["step"] + list(TAG_CANDIDATES.keys())
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, s in enumerate(steps):
            row = {"step": s}
            for k in TAG_CANDIDATES.keys():
                v = data[k][i] if i < len(data[k]) else float("nan")
                row[k] = v
            w.writerow(row)
    return out


def _write_png(run_dir: Path, steps: list[int], data: dict[str, list[float]]) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True)
    pairs = [
        ("v_delta",        axes[0, 0]),
        ("desired_v",      axes[0, 1]),
        ("current_v",      axes[0, 2]),
        ("omega_delta",    axes[1, 0]),
        ("desired_omega",  axes[1, 1]),
        ("current_omega",  axes[1, 2]),
    ]
    for name, ax in pairs:
        ax.plot(steps, data.get(name, []))
        ax.set_title(name.replace("_", " "))
        ax.set_xlabel("step")
        ax.set_ylabel("value")
    fig.suptitle("Reward components (env0)")
    out = run_dir / "reward_components_env0.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Create timeseries CSV + PNG for 6 reward component tags from a TF events run directory."
    )
    ap.add_argument("--run", required=True,
                    help="Run folder name or absolute path (e.g., 'kamma_4d_nofifo_random_RLIL_s1' "
                         "or '/workspace/.../kamma_4d_nofifo_random_RLIL_s1').")
    ap.add_argument("--base", default=None,
                    help="Optional parent directory that contains the run folder (e.g., my_runs/kamma_4d_nofifo_random_RLIL).")
    ap.add_argument("--mirror-to-artifacts", action="store_true",
                    help="Also copy CSV/PNG into artifacts/<run>/ for easy publishing.")
    args = ap.parse_args(argv)

    base = Path(args.base).resolve() if args.base else None
    run_dir = _find_run_dir(args.run, base)
    ea = _load_events(run_dir)
    steps, data, matched = _collect_timeseries(ea)

    if not steps:
        print(f"[artifacts] No scalar data found for the six tags in: {run_dir}", file=sys.stderr)
        print(f"[artifacts] Tags attempted:", file=sys.stderr)
        for k, v in TAG_CANDIDATES.items():
            print(f"  {k}: {v}", file=sys.stderr)
        sys.exit(2)

    csv_path = _write_csv(run_dir, steps, data)
    png_path = _write_png(run_dir, steps, data)

    print(f"✅ wrote CSV: {csv_path}")
    print(f"✅ wrote PNG: {png_path}")
    print("Matched tags:")
    for k, v in matched.items():
        print(f"  {k:14s} -> {v}")

    if args.mirror_to_artifacts:
        art = Path.cwd() / "artifacts" / run_dir.name
        art.mkdir(parents=True, exist_ok=True)
        for src in (csv_path, png_path):
            dst = art / src.name
            dst.write_bytes(src.read_bytes())
        print(f"📦 mirrored to: {art}")


if __name__ == "__main__":
    main()
