"""MonoGrow (experiments/monogrow/fit_test.py) on our converted plants. Same CLI as fit_test.py:
  python baselines/run_monogrow.py full fit.variant=headline fit.source=nearest fit.objective=nearest fit.sequences=<seq> \
      fit.prior=... fit.ckpt=... fit.out=...   (data.tp3d_dir / data.point_splits are set here unless given)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _ours  # noqa: E402

_ours.patch([_ours.MONO])
import fit_test  # noqa: E402  (monogrow/fit_test.py)

if __name__ == "__main__":
    argv = sys.argv[1:] or ["full"]
    keys = {a.split("=")[0] for a in argv[1:]}
    argv += [f"data.tp3d_dir={_ours.TP3D}" if "data.tp3d_dir" not in keys else "",
             f"data.point_splits={_ours.SPLITS}" if "data.point_splits" not in keys else ""]
    fit_test.main([a for a in argv if a])
