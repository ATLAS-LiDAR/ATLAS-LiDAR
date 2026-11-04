"""
Adverse-condition point-loss operators for ATLAS removal baselines.

This module is the baseline analogue of ``removal.py``. It implements the
three non-adversarial corruption operators described in the baseline plan.
Every operator deletes points *scene-wide* (never inside a chosen sector).
The matched-count operator isolates spatial targeting by preserving ATLAS's
realized point budget. Uniform and ring modes are fixed-severity contextual
controls; they preserve ATLAS's temporal opportunity but not its point budget.

  * ``uniform``  -- delete each point independently with probability p.
  * ``ring``     -- delete all points in N randomly chosen elevation rings.
  * ``matched``  -- delete exactly K_t points chosen uniformly at random
                    from the whole scene, where K_t is ATLAS's realized
                    removal budget for that frame.

FIRING GATE (shared across all three modes).
--------------------------------------------
To keep the temporal attack opportunity identical to ATLAS, every mode is
gated by the same distance-dependent A-HFR firing probability
``ahfr_gate_prob(target_range)`` evaluated on the tracked target within the
32-frame window. This is NOT a weather model -- weather does not "fire" --
it exists purely so the baselines attack on the same frames ATLAS does.
On a non-firing frame the baseline leaves the point cloud untouched, exactly
as ATLAS does, so neither run degrades mAP where the other was inactive.

Two ways to obtain the firing decision:
  * stochastic: draw Bernoulli(ahfr_gate_prob(range)) with a mode-independent
    per-frame RNG, so all three baselines fire on identical frames.
  * reference: replay the realized ``active_this_frame`` (and, for matched,
    the realized removal count) from a paired ATLAS removal run. This is the
    plan-faithful "use the same realized firing decision as ATLAS" path.
"""

import numpy as np

import removal as removal


# --------------------------------------------------------------------------
# Shared geometry (kept consistent with removal.py / the rebuttal generator)
# --------------------------------------------------------------------------

def _cart_to_spherical(points_xyz):
    """radius, azimuth in [-pi, pi], elevation in [-pi/2, pi/2]."""
    points_xyz = np.asarray(points_xyz, dtype=np.float64)
    x, y, z = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]
    planar = np.hypot(x, y)
    radius = np.sqrt(planar ** 2 + z ** 2)
    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, np.maximum(planar, 1e-12))
    return radius, azimuth, elevation


def _wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _sector_mask(points_xyz, az_center, el_center, az_width_deg, el_width_deg):
    """Same sector definition ATLAS uses -- for matched-mode counting only."""
    _, azimuth, elevation = _cart_to_spherical(points_xyz)
    az_half = np.deg2rad(az_width_deg) / 2.0
    el_half = np.deg2rad(el_width_deg) / 2.0
    return (
        (np.abs(_wrap_to_pi(azimuth - az_center)) <= az_half)
        & (np.abs(elevation - el_center) <= el_half)
    )


# --------------------------------------------------------------------------
# Firing gate (shared by all three modes)
# --------------------------------------------------------------------------

def compute_firing_gate(
    target_range_3d,
    rng,
    min_target_range_m,
    allow_close_range_extrapolation,
):
    """
    Decide whether the attack fires this frame using ATLAS's own
    distance-dependent A-HFR success probability.

    Returns (fired: bool, meta: dict). The probability mapping is exactly
    ``removal.ahfr_gate_prob`` -- the same table ATLAS uses -- so the firing
    rate as a function of target range is identical to the attack.
    """
    if (
        target_range_3d < min_target_range_m
        and not allow_close_range_extrapolation
    ):
        return False, {
            "scan_gate_probability": None,
            "scan_gate_draw": None,
            "scan_gate_passed": False,
            "range_gate_passed": False,
            "inactive_reason": "target_too_close",
        }
    probability = float(removal.ahfr_gate_prob(float(target_range_3d)))
    draw = float(rng.random())
    fired = draw < probability
    return fired, {
        "scan_gate_probability": probability,
        "scan_gate_draw": draw,
        "scan_gate_passed": bool(fired),
        "range_gate_passed": True,
        "inactive_reason": None if fired else "scan_gate",
    }


# --------------------------------------------------------------------------
# Elevation-ring binning (Ring-N mode)
# --------------------------------------------------------------------------

def build_elevation_bins(inclinations=None, n_bins=64):
    """
    Return sorted elevation bin edges (radians), ascending.

    If a Waymo top-LiDAR inclination table is available, its midpoints are
    used to construct elevation-layer boundaries aligned with those nominal
    beam inclinations. This remains an elevation-layer approximation unless
    the input points retain an explicit sensor/channel identifier.
    Otherwise a uniform partition of the physical elevation span is used.
    Build the edges once and reuse them for every variant so a given layer
    index always means the same thing.
    """
    if inclinations is not None and len(inclinations) >= 2:
        incl = np.sort(np.asarray(inclinations, dtype=np.float64))
        midpoints = 0.5 * (incl[:-1] + incl[1:])
        lower = incl[0] - (midpoints[0] - incl[0])
        upper = incl[-1] + (incl[-1] - midpoints[-1])
        return np.concatenate([[lower], midpoints, [upper]])
    return np.linspace(np.deg2rad(-25.0), np.deg2rad(5.0), n_bins + 1)


def assign_ring_index(points_xyz, elevation_edges):
    """Map points to elevation-layer indices; out-of-range points get ``-1``.

    Do not clip points outside the modeled elevation span into the lowest or
    highest layer. Clipping would incorrectly make an edge-layer dropout
    delete returns that did not belong to that layer (for example, returns
    from another LiDAR or points outside the top-LiDAR inclination range).
    """
    _, _, elevation = _cart_to_spherical(points_xyz)
    edges = np.asarray(elevation_edges, dtype=np.float64)
    n_rings = len(edges) - 1
    ring = np.searchsorted(edges, elevation, side="right") - 1

    # Include points exactly on the upper edge in the final layer, while all
    # genuinely out-of-range points remain invalid (-1).
    on_upper_edge = np.isclose(elevation, edges[-1], rtol=0.0, atol=1e-12)
    ring[on_upper_edge] = n_rings - 1
    valid = (
        (elevation >= edges[0])
        & (elevation <= edges[-1])
        & (ring >= 0)
        & (ring < n_rings)
    )
    result = np.full(ring.shape, -1, dtype=np.int64)
    result[valid] = ring[valid]
    return result


# --------------------------------------------------------------------------
# Corruption operators. uniform/ring are called ONLY on a firing frame.
# matched is called every tracked frame (it advances its aim state even when
# not firing) and removes points only when ``fired`` is True.
# --------------------------------------------------------------------------

def uniform_dropout(points, drop_prob, rng):
    """Delete each point independently with probability ``drop_prob``."""
    n = points.shape[0]
    remove_mask = rng.random(n) < drop_prob
    keep_mask = ~remove_mask
    info = {
        "mode": "uniform",
        "configured_drop_prob": float(drop_prob),
        "n_points_before": int(n),
        "n_points_removed": int(np.sum(remove_mask)),
        "n_points_after": int(np.sum(keep_mask)),
        "realized_drop_fraction": (
            float(np.sum(remove_mask)) / n if n else None
        ),
    }
    return keep_mask, remove_mask, info


def ring_dropout(points, ring_index, chosen_rings):
    """Delete points in selected elevation layers; invalid index ``-1`` stays."""
    chosen = np.asarray(sorted(chosen_rings), dtype=np.int64)
    remove_mask = (ring_index >= 0) & np.isin(ring_index, chosen)
    keep_mask = ~remove_mask
    info = {
        "mode": "ring",
        "n_rings_removed": int(len(chosen)),
        "removed_ring_indices": chosen.tolist(),
        "n_points_before": int(points.shape[0]),
        "n_points_with_valid_ring": int(np.sum(ring_index >= 0)),
        "n_points_outside_ring_span": int(np.sum(ring_index < 0)),
        "n_points_removed": int(np.sum(remove_mask)),
        "n_points_after": int(np.sum(keep_mask)),
    }
    return keep_mask, remove_mask, info


def matched_frame(
    points,
    target_box,
    state,
    fired,
    rng,
    p_remove,
    az_width_deg,
    el_width_deg,
    max_az_step_deg,
    center_jitter_std_deg,
    atlas_k=None,
):
    """
    Matched-count uniform removal for one frame.

    The aim state (servo-clamped azimuth) is advanced on EVERY tracked frame,
    firing or not, so the sector tracks the target continuously exactly as in
    ATLAS. Points are only removed when ``fired`` is True.

    K_t, the number of points to delete, is either:
      * ``atlas_k`` when a paired ATLAS run supplies the realized count
        (truest "same budget as ATLAS"), or
      * recomputed here from ATLAS's sector + per-point PRR draw when no
        reference is available.

    Exactly K_t points are then deleted uniformly at random from the WHOLE
    scene -- never the sector -- so only spatial placement differs from ATLAS.

    Returns (keep_mask, remove_mask, new_state, info).
    """
    points_xyz = points[:, :3]
    raw_az, raw_el = removal._box_center_to_az_el(target_box)

    previous_az = None if state is None else state.get("az_center")
    previous_el = None if state is None else state.get("el_center")

    # Servo-clamped azimuth (advanced every tracked frame).
    if previous_az is None:
        commanded_before_jitter = raw_az
    else:
        desired_delta = float(_wrap_to_pi(raw_az - previous_az))
        maximum_step = np.deg2rad(max_az_step_deg)
        commanded_before_jitter = float(_wrap_to_pi(
            previous_az + np.clip(desired_delta, -maximum_step, maximum_step)
        ))
    commanded_el = raw_el if previous_el is None else previous_el
    jitter_deg = float(rng.normal(0.0, center_jitter_std_deg))
    commanded_az = float(_wrap_to_pi(
        commanded_before_jitter + np.deg2rad(jitter_deg)
    ))
    new_state = {"az_center": commanded_az, "el_center": float(commanded_el)}

    n = points.shape[0]
    remove_mask = np.zeros(n, dtype=bool)
    number_sector = 0
    k_source = None
    k_t = 0

    if fired:
        if atlas_k is not None:
            # Reference path: use ATLAS's realized removal count directly.
            k_t = int(atlas_k)
            if not (0 <= k_t <= n):
                raise ValueError(
                    f"Invalid ATLAS matched budget K={k_t} for {n} points"
                )
            k_source = "atlas_reference"
        else:
            # Self-contained path: recompute the sector + PRR realization.
            sector_mask = _sector_mask(
                points_xyz, commanded_az, commanded_el,
                az_width_deg, el_width_deg,
            )
            sector_indices = np.flatnonzero(sector_mask)
            number_sector = int(sector_indices.size)
            if number_sector > 0:
                sector_removed = rng.random(number_sector) < p_remove
                k_t = int(np.sum(sector_removed))
            k_source = "recomputed_sector"
        if k_t > 0:
            chosen = rng.choice(n, size=k_t, replace=False)
            remove_mask[chosen] = True

    keep_mask = ~remove_mask
    number_removed = int(np.sum(remove_mask))
    info = {
        "mode": "matched",
        "commanded_azimuth_rad": commanded_az,
        "commanded_elevation_rad": float(commanded_el),
        "az_width_deg": float(az_width_deg),
        "el_width_deg": float(el_width_deg),
        "configured_point_removal_probability": float(p_remove),
        "matched_budget_source": k_source,
        "atlas_sector_point_count": int(number_sector),
        "atlas_matched_budget_K": int(number_removed),
        "n_points_before": int(n),
        "n_points_removed": number_removed,
        "n_points_after": int(np.sum(keep_mask)),
    }
    return keep_mask, remove_mask, new_state, info
