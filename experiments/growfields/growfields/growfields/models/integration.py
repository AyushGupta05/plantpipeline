"""Advecting points through a velocity field, and the time normalisation it needs.

A model predicts velocity at (x, t); turning that into a displaced point cloud
means integrating, either with fixed Euler substeps or an ODE solver. These
helpers are the single place that happens -- `utils/io_utils.py` and
`scripts/generate_interpolation.py` reuse them when replaying saved models.
"""

import torch
from torch import nn

try:
    from torchdiffeq import odeint as _odeint
    from torchdiffeq import odeint_adjoint as _odeint_adjoint
except ImportError:
    _odeint = None
    _odeint_adjoint = None


# Solver methods that require a fixed step size (as opposed to adaptive solvers).
# Used to decide whether ode_n_steps applies to fractional interpolation.
_FIXED_STEP_METHODS = frozenset({
    "euler", "rk4", "midpoint", "heun", "explicit_adams", "fixed_adams",
})


class _ODEFunc(nn.Module):
    """Wraps ContinuousSiren/SharedContinuousSiren for torchdiffeq.

    torchdiffeq expects ``func(t, y) -> dy/dt``.  This adapter calls the
    underlying SIREN model, optionally passing *organ_id* as a third
    argument (required for SharedContinuousSiren).
    """

    def __init__(self, siren, organ_id=None, latent_code=None):
        super().__init__()
        self.siren = siren
        self.organ_id = organ_id
        self.latent_code = latent_code

    def forward(self, t, y):
        if self.latent_code is not None:
            return self.siren(y, t, self.latent_code)
        if self.organ_id is not None:
            return self.siren(y, t, self.organ_id)
        return self.siren(y, t)


def make_t_norm_helpers(days, first_day, day_range, num_frames, use_day_based_tnorm):
    """Build ``get_t_norm`` / ``get_dt`` closures for ``_integrate_segment``.

    Args:
        days: Sequence of day values for each frame index.
        first_day: First day of the organ's full range.
        day_range: ``last_day - first_day``.
        num_frames: Total number of frames.
        use_day_based_tnorm: If True, day-based; else frame-index-based.

    Returns:
        ``(get_t_norm, get_dt)`` callables.
    """
    if use_day_based_tnorm:
        def get_t_norm(frame_idx):
            return (days[frame_idx] - first_day) / day_range

        def get_dt(frame_idx):
            if frame_idx + 1 < len(days):
                return get_t_norm(frame_idx + 1) - get_t_norm(frame_idx)
            # At boundary (extrapolation): repeat the last training interval
            return get_t_norm(frame_idx) - get_t_norm(frame_idx - 1)
    else:
        def get_t_norm(frame_idx):
            return float(frame_idx) / (num_frames - 1) if num_frames > 1 else 0.0

        def get_dt(_frame_idx):
            return 1.0

    return get_t_norm, get_dt


def _integrate_segment(
    model, x, t_start_idx, t_end_idx, get_t_norm, get_dt,
    use_odeint=False, ode_rtol=1e-4, ode_atol=1e-5,
    ode_method="dopri5", use_adjoint=True,
    backward=False,
    organ_id=None, n_substeps=1, latent_code=None,
    ode_n_steps=None,
):
    """Integrate a point cloud between two frame indices.

    Supports two integration backends and two directions:

    * **Euler** (``use_odeint=False``): manual ``x += v * dt`` loop.
    * **ODE**  (``use_odeint=True``):  adaptive solver via *torchdiffeq*,
      applied frame-by-frame.

    * **Forward**  (``backward=False``): integrate from *t_start* → *t_end*.
    * **Backward** (``backward=True``):  integrate from *t_end* → *t_start*
      by reversing the learned velocity field.

    Args:
        model: ContinuousSiren or SharedContinuousSiren instance.
        x: Starting point cloud ``[N, 3]``.
        t_start_idx: Earlier frame index.
        t_end_idx: Later frame index.
        get_t_norm: ``Callable(frame_idx) -> float`` returning normalised time.
        get_dt: ``Callable(frame_idx) -> float`` returning dt (Euler only).
        use_odeint: Use torchdiffeq adaptive solver instead of Euler.
        ode_rtol: Relative tolerance for adaptive solver.
        ode_atol: Absolute tolerance for adaptive solver.
        ode_method: Solver method (``"dopri5"``, ``"rk4"``, ``"euler"``, …).
        use_adjoint: Use adjoint method for memory-efficient backprop.
        backward: Integrate backward in time.
        organ_id: Passed as 3rd arg to model forward (for SharedContinuousSiren).
        n_substeps: Number of Euler sub-steps per frame interval. Ignored when
            use_odeint=True (the adaptive solver handles its own stepping).
        latent_code: Passed as 3rd arg to model forward (for EncoderDecoderModel).
            Takes precedence over organ_id if both are provided.
        ode_n_steps: Number of steps for fixed-step odeint solvers (rk4, euler, …).
            Computes ``step_size = dt / ode_n_steps`` per frame interval and passes it
            via ``options={'step_size': ...}``. Ignored by adaptive solvers. ``None``
            leaves the solver to choose its own step size (default).

    Returns:
        x_pred: Predicted positions ``[N, 3]``.
        velocities: List of velocity tensors (one per frame step).
    """
    if use_odeint and _odeint is None:
        raise RuntimeError(
            "torchdiffeq is required for ODE integration but is not installed. "
            "Install it with: pip install torchdiffeq"
        )

    # Ensure x is 2D [N, 3]
    if x.dim() == 1:
        x = x.unsqueeze(0)
    x_pred = x.clone()
    velocities = []

    def _call(pts, t_val):
        if latent_code is not None:
            return model(pts, t_val, latent_code)
        if organ_id is not None:
            return model(pts, t_val, organ_id)
        return model(pts, t_val)

    if not backward:
        # === FORWARD ===
        if not use_odeint:
            n = max(int(n_substeps), 1)
            for t in range(t_start_idx, t_end_idx):
                t_start_sub = get_t_norm(t)
                dt_total = get_dt(t)
                dt_sub = dt_total / n
                for s in range(n):
                    t_norm_sub = t_start_sub + s * dt_sub
                    velocity = _call(x_pred, t_norm_sub)
                    x_pred = x_pred + velocity * dt_sub
                velocities.append(velocity)
        else:
            ode_func = _ODEFunc(model, organ_id=organ_id, latent_code=latent_code)
            solver_fn = _odeint_adjoint if use_adjoint else _odeint
            for t in range(t_start_idx, t_end_idx):
                t0 = get_t_norm(t)
                t1 = get_t_norm(t + 1)
                t_span = torch.tensor([t0, t1], dtype=x_pred.dtype,
                                      device=x_pred.device)
                _ode_opts = {'step_size': (t1 - t0) / ode_n_steps} if ode_n_steps is not None else None
                sol = solver_fn(ode_func, x_pred, t_span,
                                rtol=ode_rtol, atol=ode_atol, method=ode_method,
                                options=_ode_opts)
                x_pred = sol[-1]
                velocity = _call(x_pred, t1)
                velocities.append(velocity)
    else:
        # === BACKWARD ===
        if not use_odeint:
            n = max(int(n_substeps), 1)
            for t in range(t_end_idx - 1, t_start_idx - 1, -1):
                t_end_sub = get_t_norm(t + 1)
                dt_total = get_dt(t)
                dt_sub = dt_total / n
                for s in range(n):
                    t_norm_sub = t_end_sub - s * dt_sub
                    velocity = _call(x_pred, t_norm_sub)
                    x_pred = x_pred - velocity * dt_sub
                velocities.append(velocity)
        else:
            ode_func = _ODEFunc(model, organ_id=organ_id, latent_code=latent_code)
            solver_fn = _odeint_adjoint if use_adjoint else _odeint
            for t in range(t_end_idx - 1, t_start_idx - 1, -1):
                t0 = get_t_norm(t + 1)
                t1 = get_t_norm(t)
                t_span = torch.tensor([t0, t1], dtype=x_pred.dtype,
                                      device=x_pred.device)
                _ode_opts = {'step_size': abs(t0 - t1) / ode_n_steps} if ode_n_steps is not None else None
                sol = solver_fn(ode_func, x_pred, t_span,
                                rtol=ode_rtol, atol=ode_atol, method=ode_method,
                                options=_ode_opts)
                x_pred = sol[-1]
                velocity = _call(x_pred, t1)
                velocities.append(velocity)

    return x_pred, velocities


def _integrate_time_span(
    model, x, t_start_norm, t_end_norm, num_substeps=10,
    use_odeint=False, ode_rtol=1e-4, ode_atol=1e-5,
    ode_method="dopri5", use_adjoint=False, organ_id=None, latent_code=None,
    ode_n_steps=None,
):
    """Integrate between two *normalised time values* (not frame indices).

    Designed for interpolation where we need to integrate to arbitrary
    intermediate times that do not correspond to training frame indices.

    * **Euler**: ``num_substeps`` equal-sized steps.
    * **ODE**: single solver call — the adaptive solver handles stepping.

    Args:
        model: ContinuousSiren or SharedContinuousSiren instance.
        x: Starting point cloud ``[N, 3]``.
        t_start_norm: Normalised start time (float).
        t_end_norm: Normalised end time (float).
        num_substeps: Number of Euler sub-steps (ignored when ``use_odeint``).
        use_odeint: Use torchdiffeq adaptive solver.
        ode_rtol: Relative tolerance.
        ode_atol: Absolute tolerance.
        ode_method: Solver method.
        use_adjoint: Use adjoint method.
        organ_id: Conditioning argument for SharedContinuousSiren.
        latent_code: Conditioning argument for EncoderDecoderModel. Takes
            precedence over organ_id if both are provided.
        ode_n_steps: Number of steps for fixed-step odeint solvers (rk4, euler, …).
            Computes ``step_size = (t_end - t_start) / ode_n_steps`` and passes it
            via ``options={'step_size': ...}``. Ignored by adaptive solvers. ``None``
            leaves the solver to choose its own step size (default).

    Returns:
        x_pred: Predicted positions ``[N, 3]``.
    """
    if use_odeint and _odeint is None:
        raise RuntimeError(
            "torchdiffeq is required for ODE integration but is not installed. "
            "Install it with: pip install torchdiffeq"
        )

    x_pred = x.clone()

    if not use_odeint:
        num_substeps = max(num_substeps, 1)
        dt = (t_end_norm - t_start_norm) / num_substeps
        for s in range(num_substeps):
            t_val = t_start_norm + s * dt
            if latent_code is not None:
                velocity = model(x_pred, t_val, latent_code)
            elif organ_id is not None:
                velocity = model(x_pred, t_val, organ_id)
            else:
                velocity = model(x_pred, t_val)
            x_pred = x_pred + velocity * dt
    else:
        ode_func = _ODEFunc(model, organ_id=organ_id, latent_code=latent_code)
        solver_fn = _odeint_adjoint if use_adjoint else _odeint
        t_span = torch.tensor([t_start_norm, t_end_norm],
                              dtype=x_pred.dtype, device=x_pred.device)
        _ode_opts = {'step_size': (t_end_norm - t_start_norm) / ode_n_steps} if ode_n_steps is not None else None
        sol = solver_fn(ode_func, x_pred, t_span,
                        rtol=ode_rtol, atol=ode_atol, method=ode_method,
                        options=_ode_opts)
        x_pred = sol[-1]

    return x_pred


def denormalise(x, min_xyz, scale):
    """Denormalize point cloud from [-1, 1] to original range."""
    return (x + 1) / 2 * scale + min_xyz
