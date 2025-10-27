from pathlib import Path
from types import SimpleNamespace
import math

from options_sim.cli import artifacts as A

class FakeEA:
    def __init__(self, scalars_by_tag):
        self._scalars = scalars_by_tag
    def Scalars(self, tag):
        vals = self._scalars.get(tag, [])
        # Return objects with .step and .value like TB
        return [SimpleNamespace(step=i, value=v) for i, v in enumerate(vals, start=0)]

def test_collect_and_write_artifacts(tmp_path: Path, monkeypatch):
    # Build data for a subset of tags (others will be NaN)
    # Using the first candidate names in TAG_CANDIDATES is fine.
    sbt = {
        "Reward_comp_env0 / v_delta":        [0.1, 0.2, 0.3],
        "Reward_comp_env0 / desired_v":      [1.0, 1.5, 2.0],
        "Reward_comp_env0 / current_v":      [0.9, 1.4, 1.9],
        "Reward_comp_env0 / omega_delta":    [0.05, 0.07, 0.09],
        "Reward_comp_env0 / desired_omega":  [0.2, 0.2, 0.2],
        "Reward_comp_env0 / current_omega":  [0.18, 0.19, 0.21],
    }
    ea = FakeEA(sbt)
    steps, data, matched = A._collect_timeseries(ea)
    assert steps == [0, 1, 2]
    assert set(data.keys()) == set(A.TAG_CANDIDATES.keys())
    assert matched["v_delta"] is not None

    # Write CSV/PNG
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    csv_path = A._write_csv(run_dir, steps, data)
    png_path = A._write_png(run_dir, steps, data)
    assert csv_path.exists()
    assert png_path.exists()
