# src/options_sim/cli/eval_artifacts.py
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# TensorBoard reader (Isaac-Sim python has tensorboard preinstalled)
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Headless plotting
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math


# --- Tag candidates ----------------------------------------------------------
# What you showed in W&B:
TRACKING_ERR_CANDS = [
    "Info / rew_Tracking error",
    "Info/rew_Tracking error",
    "Info/rev_Tracking error",      # defensive, just in case of typos
    "tracking_error",
]
DESIRED_VEL_CANDS = [
    "Info / rew_Desired velocity",
    "Info/rew_Desired velocity",
    "desired_velocity",
    "command/vel_x",
]


# --- utils -------------------------------------------------------------------
def _find_run_dir(run_arg: str, base: Optional[Path]) -> Path:
    p = Path(run_arg)
    if p.exists():
        return p.resolve()
    if base:
        cand = (base / run_arg).resolve()
        if cand.exists():
            return cand
    root = Path.cwd() / "my_runs"
    hits = [h for h in root.rglob(run_arg) if h.is_dir()]
    if not hits:
        raise SystemExit(f"[eval_artifacts] Could not locate run directory: {run_arg}")
    if len(hits) > 1:
        # prefer ones that have TB events
        with_events = [h for h in hits if any(h.glob("events.out.tfevents.*"))]
        if with_events:
            hits = with_events
        # shortest path heuristic
        hits.sort(key=lambda x: len(str(x)))
    return hits[0].resolve()


def _load_events(run_dir: Path) -> EventAccumulator:
    if not any(run_dir.glob("events.out.tfevents.*")):
        raise SystemExit(f"[eval_artifacts] No TensorBoard event files in: {run_dir}")
    ea = EventAccumulator(str(run_dir))
    ea.Reload()
    return ea


def _get_series(ea: EventAccumulator, candidates: List[str]):
    """Return list of (step, value) using first tag that exists; else None."""
    for tag in candidates:
        try:
            scalars = ea.Scalars(tag)
            if scalars:
                return tag, [(ev.step, float(ev.value)) for ev in scalars]
        except KeyError:
            pass
    return None, None


def _align_by_step(a: List[Tuple[int, float]], b: List[Tuple[int, float]]) -> Tuple[List[float], List[float]]:
    """Left-join on step: return values aligned as (x_vals, y_vals)."""
    if not a or not b:
        return [], []
    map_b = {s: v for s, v in b}
    xs, ys = [], []
    for s, va in a:
        if s in map_b:
            xs.append(va)  # e.g., desired velocity
            ys.append(map_b[s])  # e.g., tracking error
    return xs, ys


def _moving_average(vals: List[float], k: int) -> List[float]:
    if k <= 1 or not vals:
        return vals
    out = []
    s = 0.0
    q = []
    for v in vals:
        q.append(v)
        s += v
        if len(q) > k:
            s -= q.pop(0)
        out.append(s / len(q))
    return out


# --- main --------------------------------------------------------------------
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="Build tracking-error vs speed artifact from eval run TF events."
    )
    ap.add_argument("--run", required=True,
                    help="Run folder name or absolute path (e.g., eval_KAMMA_..., or full path).")
    ap.add_argument("--base", default=None,
                    help="Optional parent directory containing the run (e.g., my_runs/eval_kamma_4d).")
    ap.add_argument("--vmin", type=float, default=1.0,
                    help="Fallback x-axis start (m/s) if desired-velocity tag not found.")
    ap.add_argument("--vmax", type=float, default=2.0,
                    help="Fallback x-axis end (m/s) if desired-velocity tag not found.")
    ap.add_argument("--smooth", type=int, default=0,
                    help="Optional moving-average window over y (tracking error). 0 = no smoothing.")
    ap.add_argument("--mirror-to-artifacts", action="store_true",
                    help="Also copy outputs to artifacts/<run>/ for easy publishing.")
    args = ap.parse_args(argv)

    base = Path(args.base).resolve() if args.base else None
    run_dir = _find_run_dir(args.run, base)
    ea = _load_events(run_dir)

    # read series
    te_tag, te_series = _get_series(ea, TRACKING_ERR_CANDS)
    dv_tag, dv_series = _get_series(ea, DESIRED_VEL_CANDS)

    if not te_series:
        raise SystemExit(
            "[eval_artifacts] Tracking error series not found.\n"
            f"Tried tags: {TRACKING_ERR_CANDS}\n"
            f"Run: {run_dir}"
        )

    # build x (speed) and y (tracking error)
    if dv_series:
        xs, ys = _align_by_step(dv_series, te_series)
        x_label = "desired velocity (m/s)"
        x_note = f"(tag: {dv_tag})"
    else:
        # fallback: synthetic ramp from vmin to vmax over len(te_series)
        n = len(te_series)
        xs = [args.vmin + (args.vmax - args.vmin) * i / max(n - 1, 1) for i in range(n)]
        ys = [v for _, v in te_series]
        x_label = "desired velocity (m/s)"
        x_note = f"(synthetic ramp {args.vmin}→{args.vmax})"

    if args.smooth and args.smooth > 1:
        ys = _moving_average(ys, args.smooth)

    # write CSV
    csv_path = run_dir / "tracking_error_vs_speed.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["speed_m_s", "tracking_error_m_s"])  # y is distance error; change to _m_s if you truly log a rate
        for x, y in zip(xs, ys):
            w.writerow([x, y])

    # write PNG
    png_path = run_dir / "tracking_error_vs_speed.png"
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(xs, ys)
    plt.title("Tracking error vs commanded speed")
    plt.xlabel(x_label)
    plt.ylabel("tracking error (m/s)")
    plt.grid(True, alpha=0.3)
    plt.xlim(args.vmin, args.vmax)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

    print(f"✅ wrote CSV: {csv_path}")
    print(f"✅ wrote PNG: {png_path}")
    print(f"Matched tags: tracking_error -> {te_tag}; desired_velocity -> {dv_tag or 'fallback ramp'}")
    if dv_tag:
        print(f"Note: x-axis {x_note}")
    else:
        print(f"Note: x-axis {x_note}")

    if args.mirror_to_artifacts:
        art = Path.cwd() / "artifacts" / run_dir.name
        art.mkdir(parents=True, exist_ok=True)
        for src in (csv_path, png_path):
            (art / src.name).write_bytes(src.read_bytes())
        print(f"📦 mirrored to: {art}")


if __name__ == "__main__":
    main()
