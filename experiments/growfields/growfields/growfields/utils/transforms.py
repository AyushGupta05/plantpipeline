"""Rigid-transform helpers for placing canonical organ predictions back in the plant."""

import numpy as np


def interpolate_rigid_transform(A, B, f):
    """SLERP-interpolate between two rigid 4x4 homogeneous matrices.

    Naively blending matrices ((1-f)*A + f*B) linearly interpolates the rotation
    block, which is *not* a rotation for intermediate f — it introduces
    shear/scale and makes interpolated motion look kinked/linear between
    keyframes. Here we SLERP the rotation (quaternion geodesic) and lerp the
    translation, so intermediate transforms stay rigid and rotate along the
    shortest arc. Handles improper rotations (det<0, e.g. from PCA alignment) by
    factoring out a common reflection; falls back to orthonormalized lerp if the
    two endpoints disagree in handedness.
    """
    from scipy.spatial.transform import Rotation, Slerp

    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    RA, RB = A[:3, :3], B[:3, :3]
    d = (1.0 - f) * A[:3, 3] + f * B[:3, 3]
    dA, dB = np.linalg.det(RA), np.linalg.det(RB)

    def _slerp_proper(Ra, Rb):
        key = Rotation.from_matrix(np.stack([Ra, Rb]))
        return Slerp([0.0, 1.0], key)(f).as_matrix()

    if dA > 0 and dB > 0:
        R = _slerp_proper(RA, RB)
    elif dA < 0 and dB < 0:
        F = np.diag([1.0, 1.0, -1.0])  # common reflection -> make both proper
        R = _slerp_proper(RA @ F, RB @ F) @ F
    else:
        # Handedness mismatch (rare): orthonormalized lerp preserves rigidity.
        M = (1.0 - f) * RA + f * RB
        U, _, Vt = np.linalg.svd(M)
        R = U @ Vt

    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R
    out[:3, 3] = d
    return out
