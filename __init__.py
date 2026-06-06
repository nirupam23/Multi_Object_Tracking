"""Aerial Guardian: lightweight small-object detection + MOT for drone video."""

from .byte_tracker import BYTETracker, STrack
from .cmc import CameraMotionCompensation
from .kalman_filter import KalmanFilter

__all__ = ["BYTETracker", "STrack", "CameraMotionCompensation", "KalmanFilter"]
__version__ = "0.1.0"
