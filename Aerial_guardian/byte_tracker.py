"""
ByteTrack + Camera-Motion-Compensation tracker.

ByteTrack's key idea (Zhang et al., ECCV 2022): don't throw away low-confidence
detections. Most trackers keep only boxes above a high score threshold, which
loses partially-occluded / tiny / motion-blurred targets -- exactly the failure
mode on drone footage. ByteTrack associates in *two* rounds:

  Round 1: confirmed tracks  <->  HIGH-score detections (IoU + Kalman).
  Round 2: tracks still unmatched after round 1  <->  LOW-score detections.

The low-score round recovers objects that dipped in confidence (occlusion,
small size) without polluting the high-score matches. Tracks that stay
unmatched are kept "lost" for `track_buffer` frames so an ID can be re-found
after a brief disappearance.

Our additions on top of vanilla ByteTrack:
  * multi_gmc(): warps every track's Kalman state by the per-frame affine from
    the CMC module, so predictions stay valid under drone ego-motion.
  * a clean external API: update(detections, frame) -> active tracks.

This file deliberately mirrors the reference ByteTrack structure so the
behaviour is well understood and easy to defend.
"""

import numpy as np

from .kalman_filter import KalmanFilter
from .cmc import CameraMotionCompensation
from .matching import iou_distance, linear_assignment


class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack:
    _count = 0

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    @staticmethod
    def reset_count():
        BaseTrack._count = 0


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score):
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.score = float(score)
        self.tracklet_len = 0
        self.state = TrackState.New
        self.track_id = 0
        self.frame_id = 0
        self.start_frame = 0

    # ---- prediction -------------------------------------------------------
    @staticmethod
    def multi_predict(stracks):
        if len(stracks) <= 0:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_cov = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][7] = 0  # zero the height-velocity for lost tracks
        multi_mean, multi_cov = STrack.shared_kalman.multi_predict(multi_mean, multi_cov)
        for i in range(len(stracks)):
            stracks[i].mean = multi_mean[i]
            stracks[i].covariance = multi_cov[i]

    @staticmethod
    def multi_gmc(stracks, H):
        """Apply global camera motion (2x3 affine) to a batch of track states.

        R8x8 = kron(I4, R) applies the 2x2 rotation/scale block to each of the
        four state pairs (x,y),(a,h),(vx,vy),(va,vh); the translation is added
        to the position component only. Identical to BoT-SORT's GMC step.
        """
        if len(stracks) == 0:
            return
        R = H[:2, :2]
        R8x8 = np.kron(np.eye(4, dtype=float), R)
        t = H[:2, 2]
        for st in stracks:
            mean = R8x8.dot(st.mean)
            mean[:2] += t
            st.mean = mean
            st.covariance = R8x8.dot(st.covariance).dot(R8x8.T)

    # ---- lifecycle --------------------------------------------------------
    def activate(self, kalman_filter, frame_id):
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(
            self.tlwh_to_xyah(self._tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

    # ---- geometry helpers -------------------------------------------------
    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]            # a * h -> w
        ret[:2] -= ret[2:] / 2      # centre -> top-left
        return ret

    @property
    def tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret


# --------------------------------------------------------------------------
# track-list set operations
# --------------------------------------------------------------------------
def joint_stracks(tlista, tlistb):
    exists, res = {}, []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        if not exists.get(t.track_id, 0):
            exists[t.track_id] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {t.track_id: t for t in tlista}
    for t in tlistb:
        stracks.pop(t.track_id, None)
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = [], []
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb


# --------------------------------------------------------------------------
# the tracker
# --------------------------------------------------------------------------
class BYTETracker:
    def __init__(
        self,
        track_thresh=0.5,
        track_buffer=30,
        match_thresh=0.8,
        frame_rate=30,
        use_cmc=True,
        cmc_downscale=2,
    ):
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0

        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.det_thresh = track_thresh + 0.1
        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)

        self.kalman_filter = KalmanFilter()
        self.use_cmc = use_cmc
        self.cmc = CameraMotionCompensation(downscale=cmc_downscale) if use_cmc else None
        BaseTrack.reset_count()

    def update(self, dets, frame=None):
        """
        dets  : np.ndarray of shape [N, 5] = [x1, y1, x2, y2, score]
        frame : BGR image for camera-motion estimation (optional but
                recommended on drone footage).
        returns: list of currently active STrack objects.
        """
        self.frame_id += 1
        activated, refind, lost, removed = [], [], [], []

        if dets is None or len(dets) == 0:
            bboxes = np.empty((0, 4), dtype=np.float32)
            scores = np.empty((0,), dtype=np.float32)
        else:
            dets = np.asarray(dets, dtype=np.float32)
            bboxes, scores = dets[:, :4], dets[:, 4]

        remain = scores > self.track_thresh
        second = np.logical_and(scores > 0.1, scores < self.track_thresh)

        dets_high, scores_high = bboxes[remain], scores[remain]
        dets_low, scores_low = bboxes[second], scores[second]

        detections = [
            STrack(STrack.tlbr_to_tlwh(b), s) for b, s in zip(dets_high, scores_high)
        ]

        # split current tracks into confirmed vs tentative
        unconfirmed, tracked = [], []
        for t in self.tracked_stracks:
            (tracked if t.is_activated else unconfirmed).append(t)

        # predict all active + lost tracks one step forward
        strack_pool = joint_stracks(tracked, self.lost_stracks)
        STrack.multi_predict(strack_pool)

        # >>> camera motion compensation <<<
        if self.use_cmc and frame is not None:
            warp = self.cmc.apply(frame)
            STrack.multi_gmc(strack_pool, warp)
            STrack.multi_gmc(unconfirmed, warp)

        # ---- Round 1: high-score association --------------------------------
        dists = iou_distance(strack_pool, detections)
        matches, u_track, u_det = linear_assignment(dists, thresh=self.match_thresh)
        for it, idet in matches:
            track, det = strack_pool[it], detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind.append(track)

        # ---- Round 2: low-score association ---------------------------------
        detections_second = [
            STrack(STrack.tlbr_to_tlwh(b), s) for b, s in zip(dets_low, scores_low)
        ]
        r_tracked = [
            strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked
        ]
        dists = iou_distance(r_tracked, detections_second)
        matches, u_track, _ = linear_assignment(dists, thresh=0.5)
        for it, idet in matches:
            track, det = r_tracked[it], detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind.append(track)

        for it in u_track:
            track = r_tracked[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost.append(track)

        # ---- tentative tracks need an immediate high-score match -----------
        detections = [detections[i] for i in u_det]
        dists = iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_det = linear_assignment(dists, thresh=0.7)
        for it, idet in matches:
            unconfirmed[it].update(detections[idet], self.frame_id)
            activated.append(unconfirmed[it])
        for it in u_unconfirmed:
            unconfirmed[it].mark_removed()
            removed.append(unconfirmed[it])

        # ---- spawn new tracks ----------------------------------------------
        for inew in u_det:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated.append(track)

        # ---- retire tracks lost for too long -------------------------------
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.mark_removed()
                removed.append(track)

        # ---- bookkeeping ----------------------------------------------------
        self.tracked_stracks = [
            t for t in self.tracked_stracks if t.state == TrackState.Tracked
        ]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )

        return [t for t in self.tracked_stracks if t.is_activated]

    def coast(self, frame=None):
        """Advance tracks one frame WITHOUT running detection.

        Used on frames where the detector is skipped (see pipeline's
        `detect_every`). Runs the Kalman motion model one step forward and
        applies camera-motion compensation on the active + lost pool, so boxes
        keep moving and IDs persist between detections. No association happens,
        so no track is marked lost or removed here -- the next real update()
        handles that. Returns the currently active tracks (predicted positions).
        """
        self.frame_id += 1
        tracked = [t for t in self.tracked_stracks if t.is_activated]
        pool = joint_stracks(tracked, self.lost_stracks)
        STrack.multi_predict(pool)
        if self.use_cmc and frame is not None:
            warp = self.cmc.apply(frame)
            STrack.multi_gmc(pool, warp)
        return [t for t in self.tracked_stracks if t.is_activated]
