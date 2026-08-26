# LOCAL PATCH vs upstream: three unused plotting imports removed. See
# atlas/PROVENANCE.md. matplotlib and plotly were imported at module level but
# never referenced anywhere in this file (verified by AST, not grep), so importing
# the A-HFR physics required a plotting stack for no reason. No behaviour change.
import numpy as np




def _box_range(box):
    return np.linalg.norm(box[:3])

def _box_volume(box):
    """return volume of bounding box."""
    return abs(box[3] * box[4] * box[5])

def get_centers(gt_boxes):
    """
        
        return bounding box center
    
    """
    if gt_boxes is not None and len(gt_boxes) > 0:
        return gt_boxes[:, :3]
    return np.zeros((0, 3))



def select_initial_target(gt_boxes, desired_range=30.0,
                          tol=5.0, target_class=1, min_range=20.0,
                          class_col=-1, negative = False):
    """
    Pick a target near desired_range .

    Parameters
    ----------
    gt_boxes : (N, 8+) array
        Last column (class_col) is the class label (1=vehicle, 2=ped, 3=cyclist).
    desired_range : float
    tol : float
    target_class : int
        1 = vehicle class
    min_range : float
        Ignore objects closer than this.
    class_col : int
        Column index for the class label. Default -1 (last column).

    Returns
    -------
    idx : int or None — index into gt_boxes
    box : array or None — copy of selected box
    """
    if gt_boxes is None or len(gt_boxes) == 0:
        return None, None

    centers = gt_boxes[:, :3]
    dists = np.linalg.norm(centers, axis=1)
    labels = gt_boxes[:, class_col].astype(int)

    # Build candidate mask
    mask = (centers[:, 0] > 0) & (dists > min_range) & (labels == target_class)
    if(negative):
        mask = (dists > min_range) & (labels == target_class)

    # Prefer candidates within tolerance band of desired_range
    band_mask = mask & (np.abs(dists - desired_range) <= tol)
    idxs = np.where(band_mask)[0]

    if len(idxs) == 0:
        # Fallback: furthest in-front object beyond min_range
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            return None, None
        idx = idxs[np.argmax(dists[idxs])]
        print("falling back to nearest possible box")
    else:
        idx = idxs[np.argmax(np.abs(dists[idxs] - desired_range))]
        # print("whatever this condition is")

    return int(idx), gt_boxes[idx].copy()



def match_with_velocity(prev_box, prev_prev_box, curr_boxes,
                        target_class=1, max_step=6.0, max_volume_ratio=2.0,
                        class_col=-1, dt=0.1):
    """
    Nearest-neighbor matching with constant-velocity prediction and gating.

    Parameters
    ----------
    prev_box, prev_prev_box : arrays or None
    curr_boxes : (N, 8+) array
        Last column (class_col) is the class label.
    target_class : int
    max_step : float — spatial gate in meters
    max_volume_ratio : float — reject if candidate volume differs too much
    class_col : int
        Column index for the class label. Default -1 (last column).
    dt : float
        Seconds between the frames being matched, used to extrapolate the
        constant-velocity prediction. Default 0.1 (Waymo, 10 Hz). nuScenes
        KEYFRAMES are 2 Hz, so callers there must pass dt=0.5 — otherwise the
        prediction is 5x too short and the residual for a fast vehicle exceeds
        `max_step`, silently dropping the track.

    Returns
    -------
    matched_box : array or None
    matched_idx : int or None
    match_dist : float or None
    """
    if curr_boxes is None or len(curr_boxes) == 0:
        return None, None, None

    # Class filter from last column
    labels = curr_boxes[:, class_col].astype(int)
    class_mask = (labels == target_class)
    valid_idxs = np.where(class_mask)[0]
    if len(valid_idxs) == 0:
        return None, None, None
    candidates = curr_boxes[valid_idxs]

    prev_ctr = prev_box[:3]

    if prev_box is not None:
        vx, vy = prev_box[7], prev_box[8]
        pred_ctr = prev_ctr + np.array([vx, vy, 0.0]) * dt
    else:
        pred_ctr = prev_ctr
    

    # Spatial gate
    dists = np.linalg.norm(candidates[:, :3] - pred_ctr[None, :], axis=1)
    order = np.argsort(dists)

    prev_vol = _box_volume(prev_box)

    for j in order:
        if dists[j] > max_step:
            break  # all remaining are farther
        # Size consistency check
        cand_vol = _box_volume(candidates[j])
        if prev_vol > 0:
            ratio = max(cand_vol, prev_vol) / min(cand_vol, prev_vol)
            if ratio > max_volume_ratio:
                continue
        # Passed all gates
        orig_idx = int(valid_idxs[j])
        return curr_boxes[orig_idx].copy(), orig_idx, float(dists[j])

    return None, None, None





#from ndss paper
def ahfr_gate_prob(distance):
    if distance < 10:
        return 0.38   # avg of 27–44%
    elif distance <= 20 and distance >=10:
        return 0.66   # avg of 61–72%
    elif distance <= 30 and distance >=20:
        return 0.46   # avg of 40–50%
    elif distance <= 40 and distance >= 30:
        return 0.82   # avg of 73–87%
    elif distance <=50 and distance >= 40:
        return 0.93   # avg of 81–100%
    else:
        return 1.0

def _wrap_to_pi(a):
    """
    map any angle to range [-pi, pi]
    ensure angular differences are computed correctly (example: 179 deg and -179 deg are close ,not far)
    """
    return (a + np.pi) % (2 * np.pi) - np.pi




def _sector_mask_from_center(points_xyz, az_center, el_center,
                             az_width_deg=20.0, el_width_deg=16.0):
    """
    Boolean mask of points inside the active A-HFR sector.
    returns a mask of all points whose rays fall inside the 20 deg x 16 deg angular sector 
    
    """
    _, theta, phi = cart_2_spherical(points_xyz)
    d_az = _wrap_to_pi(theta - az_center)
    d_el = phi - el_center

    az_half = np.deg2rad(az_width_deg) / 2.0
    el_half = np.deg2rad(el_width_deg) / 2.0

    return (np.abs(d_az) <= az_half) & (np.abs(d_el) <= el_half)


def _clamp_angle_step(prev_angle, new_angle, max_step_rad):
    """
    Move from prev_angle toward new_angle, but limit the per-frame step.
    Useful for servo-like smooth tracking.
    
    - models maximum angular velocity of attackers laser/servo.
    - relevant for roadside tracking where target moves quickly in fov
    """
    delta = _wrap_to_pi(new_angle - prev_angle)
    delta = np.clip(delta, -max_step_rad, max_step_rad)
    return _wrap_to_pi(prev_angle + delta)


def _box_center_to_az_el(box):
    """
    box: [x, y, z, dx, dy, dz, yaw, ...] in ego/LiDAR frame.
    Returns the azimuth/elevation of the box center.
    
    -convert target object's center (x, y, z) into azimuth and elevation angles. 
    - used to determine where attacker should aim in LiDAR's fov
    - box is a proxy for where the victim object is 
    """
    ctr = np.asarray(box[:3], dtype=np.float32)[None, :]
    r, theta, phi = cart_2_spherical(ctr)
    return float(_wrap_to_pi(theta[0])), float(phi[0])



def _inactive_meta(points, az_center, el_center, az_width_deg, el_width_deg,
                   p_remove, reason, target_range_m=None,
                   extrapolated_close_range=False):
    return {
        "active_this_frame": False,
        "az_center_rad": az_center,
        "el_center_rad": el_center,
        "az_width_deg": az_width_deg,
        "el_width_deg": el_width_deg,
        "p_remove": p_remove,
        "n_points_before": int(points.shape[0]),
        "n_points_in_sector": 0,
        "n_points_removed": 0,
        "n_points_after": int(points.shape[0]),
        "target_range_m": target_range_m,
        "range_gate_passed": False if reason == "target_too_close" else True,
        "extrapolated_close_range": bool(extrapolated_close_range),
        "inactive_reason": reason,
    }



def _apply_ahfr_sector(points,
                       az_center,
                       el_center,
                       az_width_deg=20.0,
                       el_width_deg=16.0,
                       p_remove=0.96,
                       scan_gate_prob=1.0,
                       rng=None):
    """
    A-HFR simulator: removes points in a sector with given probability, and optional frame-level gate.

    Returns:
        attacked_points: filtered point array
        meta: dict with attack bookkeeping
    """
    if rng is None:
        rng = np.random.default_rng()


    active_this_frame = (rng.random() < scan_gate_prob)

    meta = {
        "active_this_frame": active_this_frame,
        "az_center_rad": az_center,
        "el_center_rad": el_center,
        "az_width_deg": az_width_deg,
        "el_width_deg": el_width_deg,
        "p_remove": p_remove,
        "n_points_before": int(points.shape[0]),
        "n_points_in_sector": 0,
        "n_points_removed": 0,
        "n_points_after": int(points.shape[0]),
        "range_gate_passed": True,
        "extrapolated_close_range": False,
        "inactive_reason": None,
    }

    if not active_this_frame:
        meta["inactive_reason"] = "scan_gate"
        return points.copy(), meta

    #xyz points
    pts_xyz = points[:, :3]
    
    #get all points in target removal area
    sector_mask = _sector_mask_from_center(
        pts_xyz,
        az_center=az_center,
        el_center=el_center,
        az_width_deg=az_width_deg,
        el_width_deg=el_width_deg
    )

    #indices of nonzero elements in sector_mask 
    sector_idx = np.flatnonzero(sector_mask)
    meta["n_points_in_sector"] = int(sector_idx.size)

    if sector_idx.size == 0:
        meta["inactive_reason"] = "empty_sector"
        return points.copy(), meta
    
    
    #  remove with paper-backed probability.
    remove_draw = rng.random(sector_idx.size) < p_remove
    remove_idx = sector_idx[remove_draw]

    keep_mask = np.ones(points.shape[0], dtype=bool)
    keep_mask[remove_idx] = False

    attacked_points = points[keep_mask]

    meta["n_points_removed"] = int(remove_idx.size)
    meta["n_points_after"] = int(attacked_points.shape[0])

    return attacked_points, meta


def _box_center_range(box):
    """
    Euclidean distance from LiDAR origin to target box center.
    """
    ctr = np.asarray(box[:3], dtype=np.float32)
    return float(np.linalg.norm(ctr))


def spoof_frame_ahfr_roadside(points,
                              target_box,
                              state=None,
                              az_width_deg=20.0,
                              el_width_deg=16.0,
                              p_remove=0.96,
                              max_az_step_deg=15.0,
                              fix_elevation=True,
                              center_jitter_std_deg=0.05,
                              min_target_range_m=6.0,
                              allow_close_range_extrapolation=False,
                              rng=None):
    """
    A-HFR frame attack for a stationary roadside spoofer.

    ------
    points : (N, D) array
        Waymo frame points.
    target_box : array-like length >= 7
        Real actor used only to aim the sector for this frame.
    state : dict or None
        Persistent attack state across frames.
        Expected keys after first call:
            - az_center
            - el_center
    az_width_deg, el_width_deg : float
        Paper-backed default sector.
    p_remove : float
        0.96 or 1.00 are the paper-backed A-HFR settings.
    max_az_step_deg : float
        Max horizontal step per frame to mimic smooth auto-aiming.
    fix_elevation : bool
        True is the paper-faithful setting for flat-road scenarios:
        horizontal tracking only, vertical center fixed.
    scan_gate_prob : float
        Default 1.0. Set lower only as an extra ablation for missed launches.
    rng : np.random.Generator or None

    Returns
    -------
    attacked_points, new_state, meta
    """
    if rng is None:
        rng = np.random.default_rng()

    if state is None:
        state = {}

    
    raw_az, raw_el = _box_center_to_az_el(target_box)
    target_range_m = _box_center_range(target_box)
    
    scan_gate_prob = ahfr_gate_prob(target_range_m)

    if "az_center" not in state:
        az_center = raw_az
    else:
        az_center = _clamp_angle_step(
            state["az_center"],
            raw_az,
            max_step_rad=np.deg2rad(max_az_step_deg)
        )

    if "el_center" not in state:
        el_center = raw_el
    else:
        if fix_elevation:
            el_center = state["el_center"]
        else:
            el_center = raw_el

    az_center = _wrap_to_pi(
        az_center + np.deg2rad(rng.normal(0.0, center_jitter_std_deg))
    )

    new_state = {
        "az_center": az_center,
        "el_center": el_center,
    }

    if target_range_m < min_target_range_m and not allow_close_range_extrapolation:
        meta = _inactive_meta(
            points=points,
            az_center=az_center,
            el_center=el_center,
            az_width_deg=az_width_deg,
            el_width_deg=el_width_deg,
            p_remove=p_remove,
            reason="target_too_close",
            target_range_m=target_range_m,
            extrapolated_close_range=False,
        )
        meta.update({
            "target_box": target_box,
            "mode": "roadside",
            "raw_target_az_rad": raw_az,
            "raw_target_el_rad": raw_el,
            "max_az_step_deg": max_az_step_deg,
            "fix_elevation": bool(fix_elevation),
        })
        return points.copy(), new_state, meta

    attacked_points, meta = _apply_ahfr_sector(
        points=points,
        az_center=az_center,
        el_center=el_center,
        az_width_deg=az_width_deg,
        el_width_deg=el_width_deg,
        p_remove=p_remove,
        scan_gate_prob=scan_gate_prob,
        rng=rng
    )

    meta.update({
        "target_box": target_box,
        "mode": "roadside",
        "raw_target_az_rad": raw_az,
        "raw_target_el_rad": raw_el,
        "max_az_step_deg": max_az_step_deg,
        "fix_elevation": bool(fix_elevation),
        "target_range_m": target_range_m,
        "range_gate_passed": True,
        "extrapolated_close_range": bool(
            target_range_m < min_target_range_m and allow_close_range_extrapolation
        ),
    })

    return attacked_points, new_state, meta

def cart_2_spherical(points):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)
    theta = theta % (2 * np.pi)
    arg = z/r
    phi = np.arcsin(arg)
    return r, theta, phi

