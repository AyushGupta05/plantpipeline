"""Checks that need no experiment directory: run with `pytest tests/test_standalone.py`.

The rest of the suite inspects saved PLYs. These cover the pieces the published
protocol relies on that can be tested directly: the `--deterministic` loss, pose
interpolation, and that metrics do not depend on whether they are computed in an
organ's canonical frame or in plant space.
"""

import os

import numpy as np
import pytest
import torch
from pytorch3d.loss import chamfer_distance
from scipy.spatial.transform import Rotation

from growfields.pipeline.config import DETERMINISTIC_ENV
from growfields.utils.metrics import chamfer_loss, compute_chamfer_distance, compute_epe
from growfields.utils.transforms import interpolate_rigid_transform

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _rigid(rotvec, translation, reflect=False):
    M = np.eye(4)
    M[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    if reflect:
        M[:3, :3] = M[:3, :3] @ np.diag([1.0, 1.0, -1.0])
    M[:3, 3] = translation
    return M


def _apply(M, pts):
    return pts @ M[:3, :3].T + M[:3, 3]


# ---------------------------------------------------------------------------
# --deterministic trains with the same loss as the default path
# ---------------------------------------------------------------------------

class TestDeterministicChamfer:
    """The atomic-free Chamfer loss must equal PyTorch3D's, value and gradient.

    README and REPRODUCIBILITY.md both call it equivalent; a PyTorch3D upgrade or
    an edit to `chamfer_loss` that broke this would silently change training.
    """

    def _loss_and_grad(self, monkeypatch, deterministic, x, y):
        monkeypatch.setenv(DETERMINISTIC_ENV, "1" if deterministic else "0")
        x = x.clone().requires_grad_(True)
        loss = chamfer_loss(x, y)
        loss.backward()
        return loss.detach(), x.grad

    def test_value_and_gradient_match_pytorch3d(self, monkeypatch):
        g = torch.Generator().manual_seed(0)
        x = torch.randn(2, 300, 3, generator=g)
        y = torch.randn(2, 200, 3, generator=g)

        loss_det, grad_det = self._loss_and_grad(monkeypatch, True, x, y)
        loss_p3d, grad_p3d = self._loss_and_grad(monkeypatch, False, x, y)

        torch.testing.assert_close(loss_det, loss_p3d)
        torch.testing.assert_close(grad_det, grad_p3d)

    def test_default_path_is_pytorch3d(self, monkeypatch):
        monkeypatch.delenv(DETERMINISTIC_ENV, raising=False)
        x, y = torch.rand(1, 50, 3), torch.rand(1, 40, 3)
        assert torch.equal(chamfer_loss(x, y), chamfer_distance(x, y)[0])


# ---------------------------------------------------------------------------
# Pose interpolation stays rigid
# ---------------------------------------------------------------------------

class TestInterpolateRigidTransform:

    @pytest.mark.parametrize("reflect", [False, True], ids=["proper", "reflected"])
    def test_endpoints_are_exact(self, reflect):
        A = _rigid([0.1, -0.2, 0.3], [1.0, 2.0, 3.0], reflect)
        B = _rigid([-0.5, 0.4, 1.2], [4.0, -1.0, 0.5], reflect)
        np.testing.assert_allclose(interpolate_rigid_transform(A, B, 0.0), A, atol=1e-12)
        np.testing.assert_allclose(interpolate_rigid_transform(A, B, 1.0), B, atol=1e-12)

    @pytest.mark.parametrize("reflect", [False, True], ids=["proper", "reflected"])
    def test_intermediate_poses_are_rigid(self, reflect):
        A = _rigid([0.1, -0.2, 0.3], [1.0, 2.0, 3.0], reflect)
        B = _rigid([-0.5, 0.4, 1.2], [4.0, -1.0, 0.5], reflect)
        for f in np.linspace(0.1, 0.9, 5):
            R = interpolate_rigid_transform(A, B, f)[:3, :3]
            np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
            assert np.sign(np.linalg.det(R)) == np.sign(np.linalg.det(A[:3, :3]))

    def test_midpoint_is_half_the_rotation(self):
        A = _rigid([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        B = _rigid([0.0, 0.0, np.pi / 2], [2.0, 0.0, 0.0])
        M = interpolate_rigid_transform(A, B, 0.5)
        np.testing.assert_allclose(M, _rigid([0.0, 0.0, np.pi / 4], [1.0, 0.0, 0.0]), atol=1e-12)


# ---------------------------------------------------------------------------
# Canonical-frame and plant-space metrics agree
# ---------------------------------------------------------------------------

class TestMetricsInvariantToAlignment:
    """Per-organ metrics are computed in each organ's canonical frame, full-plant
    metrics in plant space. They agree because every alignment is rigid: CD and
    EPE only use distances, which a rigid transform preserves.
    """

    @pytest.mark.parametrize("reflect", [False, True], ids=["proper", "reflected"])
    def test_cd_and_epe_unchanged_by_rigid_transform(self, reflect):
        rng = np.random.default_rng(0)
        pred, gt = rng.normal(size=(400, 3)) * 50, rng.normal(size=(350, 3)) * 50
        pred_tip, gt_tip = rng.normal(size=(1, 3)) * 50, rng.normal(size=(1, 3)) * 50
        M = _rigid([0.7, -1.1, 2.0], [120.0, -40.0, 15.0], reflect)

        cd, _ = compute_chamfer_distance(pred, gt)
        cd_moved, _ = compute_chamfer_distance(_apply(M, pred), _apply(M, gt))
        assert cd_moved == pytest.approx(cd, rel=1e-4)
        assert compute_epe(_apply(M, pred_tip), _apply(M, gt_tip)) == pytest.approx(
            compute_epe(pred_tip, gt_tip), rel=1e-9
        )

    def test_dataset_alignments_are_rigid(self):
        """Every composed organ alignment in one sequence is a rotation (no scale or shear)."""
        dataset_path = os.path.join(REPO_ROOT, "data", "TrackPlant3D", "versions")
        if not os.path.isdir(os.path.join(dataset_path, "v2")):
            pytest.skip("TrackPlant3D v2 not downloaded; the first training run fetches it")
        plant_shape_analysis = pytest.importorskip("plant_shape_analysis")
        from omegaconf import OmegaConf

        from growfields.models.organ_model_manager import OrganModelManager
        from growfields.pipeline.data import load_plant_dataset

        seq = "tobacco_control_plant1"
        cfg = OmegaConf.load(os.path.join(REPO_ROOT, "configs/per_organ/shared_continuous.yaml"))
        cfg.dataset.dataset_path = dataset_path
        dataset = load_plant_dataset(
            cfg.dataset, seq,
            split_file=os.path.join(
                REPO_ROOT, "configs/datasets/TrackPlant3D/point_splits", f"{seq}_split.json"
            ),
            timepoint_stride=1, normalize=False, device="cpu",
        )
        manager = OrganModelManager(
            dataset, OmegaConf.to_container(cfg.default_model, resolve=True), {}, device="cpu"
        )

        n_checked = 0
        compose = plant_shape_analysis.LeafSequencesDataset.compose_transformations
        for stages in manager.organ_transformations_full.values():
            for comp in compose(stages):
                R = np.asarray(comp["composed_matrix"], dtype=np.float64)[:3, :3]
                np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-9)
                n_checked += 1
        assert n_checked > 0
