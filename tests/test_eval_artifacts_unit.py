from pathlib import Path
from types import SimpleNamespace

from options_sim.cli import eval_artifacts as E

class FakeEA:
    def __init__(self, scalars_by_tag):
        self._scalars = scalars_by_tag
    def Scalars(self, tag):
        vals = self._scalars.get(tag, [])
        return [SimpleNamespace(step=i, value=v) for i, v in enumerate(vals, start=0)]

def test_tracking_vs_speed_helpers(tmp_path: Path):
    # Build matching tags using your candidate lists
    te_tag = E.TRACKING_ERR_CANDS[0]  # "Info / rew_Tracking error"
    dv_tag = E.DESIRED_VEL_CANDS[0]   # "Info / rew_Desired velocity"

    scalars = {
        te_tag: [0.5, 0.4, 0.6, 0.3],
        dv_tag: [1.0, 1.25, 1.5, 1.75],
    }
    ea = FakeEA(scalars)

    tag, te_series = E._get_series(ea, E.TRACKING_ERR_CANDS)
    assert tag == te_tag and len(te_series) == 4

    tag, dv_series = E._get_series(ea, E.DESIRED_VEL_CANDS)
    assert tag == dv_tag and len(dv_series) == 4

    xs, ys = E._align_by_step(dv_series, te_series)
    assert xs == [1.0, 1.25, 1.5, 1.75]
    assert ys == [0.5, 0.4, 0.6, 0.3]

    # moving-average smoothing
    smoothed = E._moving_average(ys, k=2)
    assert len(smoothed) == 4 and abs(smoothed[1] - (0.5+0.4)/2) < 1e-9
