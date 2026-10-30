"""All knobs of MonoGrow as plain dataclasses (no config framework).

`smoke()` runs on a laptop CPU in minutes; `full()` is the one-GPU configuration.
Paths are absolute so every script can be run from any working directory.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SCRATCH = Path("/private/tmp/claude-501/-Users-ayush-dev-plantpipeline/59224c09-250d-4b23-b25b-9df144709fe3/scratchpad")

SPECIES = ["maize", "tomato", "sorghum", "tobacco", "sugar_beet"]
SPECIES_ID = {s: i for i, s in enumerate(SPECIES)}

# Pheno4D plants that are the same physical plants as TrackPlant3D maize / tomato2 (sources/README.md).
PHENO4D_LEAK = ["M01", "M02", "M04", "T01", "T05"]


@dataclass
class DataConfig:
    unified: Path = REPO / "data" / "unified"
    # TrackPlant3D v2 PLYs (ETH zip extracted by experiments/growfields/smoke_cpu.sh) and GrowFields point splits.
    # (on a box without the scratchpad, e.g. the AWS GPU instance, the growfields checkout's copies are used)
    tp3d_dir: Path = HERE.parent / "growfields" / "data" / "TrackPlant3D" / "versions" / "v2" / "gt_corrected_v2"  # same layout on the Mac and on the GPU box
    point_splits: Path = REPO / "experiments" / "growfields" / "growfields" / "configs" / "datasets" / "TrackPlant3D" / "point_splits"
    datasets: tuple[str, ...] = ("trackplant3d", "pheno4d", "sugar4d")
    exclude_sequences: tuple[str, ...] = tuple(PHENO4D_LEAK)
    tp3d_splits: tuple[str, ...] = ("train",)          # TrackPlant3D val/test never enter the prior
    other_splits: tuple[str, ...] = ("train", "val", "test")  # Pheno4D / Sugar4D internal splits are not GF's
    min_frames: int = 2                                 # organ tracks shorter than this are dropped
    min_points: int = 3                                 # PSA drops organ frames with < 3 points
    cache: Path = HERE / "cache"


@dataclass
class ModelConfig:
    hidden: int = 256
    hidden_layers: int = 5          # GrowFields convention: first sine layer + hidden_layers sine layers + linear out
    omega: float = 30.0
    z_dim: int = 32
    species_dim: int = 8
    n_species: int = len(SPECIES)
    scale_mm: float = 100.0         # canonical mm / scale_mm are the network's spatial units
    a_max: float = 60.0             # organ age (days) normaliser
    p_max: float = 60.0             # plant age (days) normaliser
    monotone: bool = True           # False = plain 3-d velocity head (ablation)
    eps: float = 1e-3               # guard on ||x|| for x_hat (scaled units = 0.1 mm)
    step_per_frame: bool = False    # True = GrowFields' one Euler step per frame gap; False = 1-day grid
    dt_days: float = 1.0


@dataclass
class TrainConfig:
    steps: int = 150_000
    max_tracks: int = 0             # > 0: random subset of organ tracks (smoke runs)
    organs_per_step: int = 10
    points_per_step: int = 8192     # split equally over the sampled organs (min 256 / organ)
    p_consecutive: float = 0.5      # GF: 50 % consecutive-frame pairs, else random (s, k>s)
    balance_species: bool = True    # sample organs so each species gets an equal share (sugar beet is 65 % of tracks)
    max_horizon_days: float = 14.0  # cap on day(k) - day(s) when sampling random pairs
    lr_theta: float = 1e-4
    grad_clip: float = 1.0          # global grad-norm clip over theta and z (a step-5000 blow-up without it)
    lr_z: float = 1e-3
    z_init_std: float = 0.1
    lambda_z: float = 1e-4          # * mean over organs of ||z||^2
    lambda_w: float = 0.0           # * mean ||P_perp w||^2 (scaled units/day)^2
    lambda_g: float = 0.0           # * mean softplus(g)
    log_every: int = 100
    ckpt_every: int = 5000
    lr_decay: str = "none"         # none | cosine (to 0 over `steps`); late drift of the growth head seen without decay
    keep_every: int = 25000         # also keep tagged ckpt_<step>.pt / prior_<step>.pt
    seed: int = 0
    threads: int = 2
    out: Path = HERE / "runs" / "full"
    prior_ridge: float = 1e-3       # added to the diagonal of each species covariance
    prior_shrink: float = 5.0       # pseudo-count shrinkage of Sigma_s towards the pooled covariance


@dataclass
class FitConfig:
    variant: str = "headline"       # headline | interpolation | extrapolation
    split: str = "test"             # GrowFields plant split: test (8) or val (4)
    sequences: tuple[str, ...] = ()  # subset, empty = all of the split
    n_z: int = 1000
    lr_z: float = 1e-2
    lambda_prior: float = 1e-3      # * (z-mu)^T Sigma^-1 (z-mu), per organ
    n_theta: int = 0                # stage 2: fine-tune theta for this many steps (0 = off)
    lr_theta: float = 1e-6
    lambda_w: float = 0.0           # * mean ||P_perp w||^2 during fitting (0 = off)
    source: str = "first"           # rollout source for the exported predictions: first (GrowFields) | nearest observed frame
    objective: str = "first"        # fitting loss: first = multi-step CD of one rollout from the first observed frame (GrowFields);
                                    # nearest = mean over consecutive observed pairs (s,t) of CD(rollout of s to day t, frame t) = what source=nearest scores
    reuse: str = ""                 # another fit.out: re-export its fitted codes / theta (sequences/<plant>/fit_state.pt) instead of fitting
    points_per_organ: int = 1024
    seed: int = 0
    threads: int = 2
    prior: Path = HERE / "runs" / "full" / "prior.pt"
    ckpt: Path = HERE / "runs" / "full" / "ckpt.pt"
    out: Path = HERE / "runs" / "full" / "fit_headline"
    holdout_last_n: int = 3
    stride: int = 2
    gt_fill_single_frame: bool = True  # GF parity: organs observed < 2 times get no model, GT is copied


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    fit: FitConfig = field(default_factory=FitConfig)


def full() -> Config:
    return Config()


def smoke() -> Config:
    c = Config()
    c.train = dataclasses.replace(
        c.train, steps=300, max_tracks=20, organs_per_step=4, points_per_step=2048, max_horizon_days=6.0,
        log_every=25, ckpt_every=300, out=HERE / "runs" / "smoke",
    )
    c.fit = dataclasses.replace(
        c.fit, n_z=100, points_per_organ=512, prior=c.train.out / "prior.pt", ckpt=c.train.out / "ckpt.pt",
        out=c.train.out / "fit_headline", sequences=("sorghum_highlight_plant2",),
    )
    return c


CONFIGS = {"full": full, "smoke": smoke}


def apply_overrides(cfg: Config, overrides: list[str]) -> Config:
    """`section.key=value` overrides from the command line, typed by the dataclass field."""
    for o in overrides:
        key, val = o.split("=", 1)
        sec, name = key.split(".")
        section = getattr(cfg, sec)
        cur = getattr(section, name)
        if isinstance(cur, bool):
            v = val.lower() in ("1", "true", "yes")
        elif isinstance(cur, int):
            v = int(val)
        elif isinstance(cur, float):
            v = float(val)
        elif isinstance(cur, Path):
            v = Path(val).expanduser().resolve()
        elif isinstance(cur, tuple):
            v = tuple(x for x in val.split(",") if x)
        else:
            v = val
        setattr(section, name, v)
    return cfg


def load(name: str, overrides: list[str] | None = None) -> Config:
    return apply_overrides(CONFIGS[name](), overrides or [])


def as_dict(cfg: Config) -> dict:
    return {k: {kk: (str(vv) if isinstance(vv, Path) else vv) for kk, vv in v.items()}
            for k, v in dataclasses.asdict(cfg).items()}
