"""RichardsNet (experiments/richards/fit_test.py) on our converted plants. Same CLI as fit_test.py, e.g.
  python baselines/run_richards.py --variant headline --template nearest --sequences pheno4d_maize_M01 --out runs/...
fit_test.py builds DataConfig() itself (no override hook), so DataConfig is replaced by one pointing at our tree."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _ours  # noqa: E402

_ours.patch([_ours.RICH, _ours.MONO])
import config as C  # noqa: E402
import fit_test  # noqa: E402  (richards/fit_test.py; its `from common import ...` also puts monogrow on sys.path)

fit_test.DataConfig = lambda: C.DataConfig(tp3d_dir=_ours.TP3D, point_splits=_ours.SPLITS)
if __name__ == "__main__":
    fit_test.main()
