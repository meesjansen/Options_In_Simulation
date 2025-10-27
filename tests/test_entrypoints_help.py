import subprocess, sys

def _run_mod(mod, *args):
    # Run "python -m module args..." with a short timeout
    return subprocess.run([sys.executable, "-m", mod, *args],
                          capture_output=True, text=True, timeout=30)

def test_train_help():
    r = _run_mod("options_sim.cli.train", "--help")
    assert r.returncode == 0
    assert "options-sim-train" in r.stdout or "Route to a legacy" in r.stdout

def test_eval_help():
    r = _run_mod("options_sim.cli.eval", "--help")
    assert r.returncode == 0
    assert "options-sim-eval" in r.stdout or "legacy eval script" in r.stdout

def test_artifacts_help():
    r = _run_mod("options_sim.cli.artifacts", "--help")
    assert r.returncode == 0
    txt = (r.stdout + r.stderr).lower()
    assert "timeseries" in txt and ("csv" in txt or "png" in txt)

def test_eval_artifacts_help():
    r = _run_mod("options_sim.cli.eval_artifacts", "--help")
    assert r.returncode == 0
    assert "tracking-error vs" in (r.stdout + r.stderr).lower()
