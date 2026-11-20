"""Flag parked vehicles in a tracks table (contract format), so tracking scoring can
ignore them the same way for every tracker. SCT ground truth has no parked vehicles,
but labels_GT and any detector do.

Rule: a track is parked if it spans at least MIN_FRAMES frames and its box centre
moves less than MAX_CENTRE_MOTION_PX over the whole track. The values were measured
on RoundaboutHD (see docs/pipeline_contract.md); keep them fixed for every tracker.
"""
import numpy as np
import pandas as pd

MAX_CENTRE_MOTION_PX = 30.0
MIN_FRAMES = 150  # 10 s at 15 fps; shorter pieces of moving tracks can look still


def centre_motion(tracks: pd.DataFrame) -> pd.Series:
    """Per track_id: how far the box centre moves over the track, in pixels.

    Measured as the diagonal of the 5th-95th percentile range of the centre's x and y,
    so a few jittery or occluded boxes don't count as motion.
    """
    c = pd.DataFrame({"track_id": tracks.track_id,
                      "cx": tracks.x + tracks.w / 2, "cy": tracks.y + tracks.h / 2})
    q = c.groupby("track_id")[["cx", "cy"]].quantile([0.05, 0.95]).unstack()
    return np.hypot(q[("cx", 0.95)] - q[("cx", 0.05)], q[("cy", 0.95)] - q[("cy", 0.05)]).rename("motion_px")


def parked_track_ids(tracks: pd.DataFrame, max_motion_px: float = MAX_CENTRE_MOTION_PX,
                     min_frames: int = MIN_FRAMES) -> set:
    """Track ids that span >= min_frames frames and whose box centre moves < max_motion_px."""
    span = tracks.groupby("track_id").frame.agg(lambda f: f.max() - f.min() + 1)
    motion = centre_motion(tracks)
    return set(motion.index[(motion < max_motion_px) & (span.reindex(motion.index) >= min_frames)])
