# Put this in Echo-Me-compvi/scripts/hand_landmarking/video_sequence_collector.py
from __future__ import annotations

import os
import sys
import time
import argparse
from typing import Optional

import cv2
import numpy as np

# Make sure scripts/ is importable when running this module directly
HERE = os.path.abspath(os.path.dirname(__file__))
SCRIPTS_DIR = os.path.abspath(os.path.join(HERE, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Reuse project components
from hand_landmarking.hand_landmark_detector import HandLandmarkDetector
from hand_landmarking.utils import mp_results_to_canonical_two_hand_vector
from hand_landmarking.optical_flow import (
    track_landmarks_lk,
    extract_left_right_from_mp_results,
    build_flat_from_slots,
)
from data_pipeline.utils import SEQUENCE_LENGTH, DATA_DIR


class VideoSequenceCollector:
    """Collect sequences from a video file using the project's detection pipeline."""

    def __init__(self, out_dir: Optional[str] = None, verbose: bool = True):
        self.out_dir = out_dir or DATA_DIR
        os.makedirs(self.out_dir, exist_ok=True)
        self.verbose = bool(verbose)

        # detector instance
        self.detector = HandLandmarkDetector()

        # optical-flow state
        self.prev_gray: Optional[np.ndarray] = None
        # prev_slots holds last extracted left/right pixel coords and z if available
        # Format expected by optical_flow helpers: (left_xy,left_z,right_xy,right_z,presence)
        self.prev_slots = None

        # sequence counters
        self.sequence_count = 0

    def _save_left_right_json(self, left_sequence, right_sequence, sign_label: str) -> str:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        fname = f"{sign_label}_{timestamp}_{self.sequence_count:04d}.json"
        path = os.path.join(self.out_dir, fname)
        payload = [
            {"label": "left", "sequence": left_sequence},
            {"label": "right", "sequence": right_sequence},
        ]
        # Write compact JSON
        import json

        with open(path, "w") as f:
            json.dump(payload, f, separators=(",", ":"))
        self.sequence_count += 1
        if self.verbose:
            print(f"Saved sequence file: {path} (left_frames={len(left_sequence)}, right_frames={len(right_sequence)})")
        return path

    def _flat_to_left_right(self, flat_126) -> tuple[list[float], list[float]]:
        """Split a 126-length flattened frame into left (63) and right (63)."""
        arr = np.asarray(flat_126).reshape(-1)
        if arr.size < 126:
            tmp = np.zeros((126,), dtype=np.float32)
            tmp[: arr.size] = arr
            arr = tmp
        left = arr[:63].astype(float).tolist()
        right = arr[63:126].astype(float).tolist()
        return left, right

    def _attempt_optical_flow(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Attempt to propagate previous landmarks via LK optical flow.

        Returns a 126-length flat frame if successful, else None.
        """
        if track_landmarks_lk is None or self.prev_slots is None or self.prev_gray is None:
            return None

        left_xy, left_z, right_xy, right_z, pres = self.prev_slots
        pts = []
        slots_info = []
        if left_xy is not None:
            for i in range(left_xy.shape[0]):
                pts.append(left_xy[i])
                slots_info.append((0, i))
        if right_xy is not None:
            for i in range(right_xy.shape[0]):
                pts.append(right_xy[i])
                slots_info.append((1, i))
        if len(pts) == 0:
            return None
        pts = np.array(pts, dtype=np.float32)

        cur_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        try:
            next_pts, st, err = track_landmarks_lk(self.prev_gray, cur_gray, pts)
        except Exception:
            return None
        if next_pts is None or st is None:
            return None
        st = np.asarray(st).reshape(-1)
        if st.sum() < 6:  # require at least some tracked points
            return None

        left_new = None
        right_new = None
        idx = 0
        for (which, i) in slots_info:
            if st[idx]:
                pt = next_pts[idx]
                if which == 0:
                    if left_new is None:
                        left_new = np.zeros((21, 2), dtype=np.float32)
                    left_new[i] = pt
                else:
                    if right_new is None:
                        right_new = np.zeros((21, 2), dtype=np.float32)
                    right_new[i] = pt
            idx += 1

        # Try to use build_flat_from_slots if available (preferred)
        if build_flat_from_slots is not None:
            try:
                # build_flat_from_slots returns (flat, presence)
                flat, presence = build_flat_from_slots(
                    left_new,
                    None,
                    right_new,
                    None,
                    frame_bgr.shape,
                )
                # Check we got meaningful data
                if flat is not None and np.sum(np.abs(np.asarray(flat))) > 0:
                    # update prev_slots presence for future frames
                    self.prev_slots = (left_new, None, right_new, None, presence)
                    self.prev_gray = cur_gray
                    return flat
            except Exception:
                # fallthrough to fallback composition below
                pass

        # Fallback: assemble 126-length flat with z=0 for tracked points
        flat_arr = np.zeros((126,), dtype=np.float32)
        if left_new is not None:
            for i in range(min(21, left_new.shape[0])):
                x, y = float(left_new[i, 0]), float(left_new[i, 1])
                flat_arr[i * 3 + 0] = x
                flat_arr[i * 3 + 1] = y
                flat_arr[i * 3 + 2] = 0.0
        if right_new is not None:
            base = 21 * 3
            for i in range(min(21, right_new.shape[0])):
                x, y = float(right_new[i, 0]), float(right_new[i, 1])
                flat_arr[base + i * 3 + 0] = x
                flat_arr[base + i * 3 + 1] = y
                flat_arr[base + i * 3 + 2] = 0.0
        # Update prev state
        self.prev_slots = (left_new, None, right_new, None, (1 if left_new is not None else 0, 1 if right_new is not None else 0))
        self.prev_gray = cur_gray
        return flat_arr

    def process_video(
        self,
        video_path: str,
        sign_label: str,
        max_sequences: Optional[int] = None,
        max_frames: Optional[int] = None,
        skip_frames: int = 0,
    ) -> int:
        """Process `video_path` and save sequences labeled by `sign_label`."""

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")

        left_buffer = []
        right_buffer = []
        sequences_saved = 0
        frame_idx = 0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None
        if self.verbose:
            print(f"Processing video: {video_path} total_frames={total_frames}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

                # optional skipping
                if skip_frames > 0 and (frame_idx - 1) % (skip_frames + 1) != 0:
                    continue

                # 1) Try MediaPipe detection via detector.process
                flat_vec = None
                try:
                    mp_results = self.detector.process(frame)
                    if mp_results is not None:
                        flat_vec = mp_results_to_canonical_two_hand_vector(mp_results, frame.shape)
                        # If canonical helper returns (flat, presence), unpack
                        if isinstance(flat_vec, tuple):
                            flat_vec = flat_vec[0]
                except Exception:
                    flat_vec = None

                # 2) If no detection (or empty), attempt optical flow propagation
                if flat_vec is None or np.sum(np.abs(np.asarray(flat_vec).reshape(-1))) == 0:
                    opt = self._attempt_optical_flow(frame)
                    if opt is not None and np.sum(np.abs(np.asarray(opt).reshape(-1))) > 0:
                        flat_vec = opt

                # 3) Fallback to last valid frame from buffers if available
                if flat_vec is None or np.sum(np.abs(np.asarray(flat_vec).reshape(-1))) == 0:
                    # try last valid from buffers
                    last_left = None
                    last_right = None
                    for i in range(len(left_buffer) - 1, -1, -1):
                        if left_buffer[i] is not None and np.sum(np.abs(np.asarray(left_buffer[i]))) > 0:
                            last_left = left_buffer[i]
                            break
                    for i in range(len(right_buffer) - 1, -1, -1):
                        if right_buffer[i] is not None and np.sum(np.abs(np.asarray(right_buffer[i]))) > 0:
                            last_right = right_buffer[i]
                            break
                    if last_left is not None and last_right is not None:
                        flat_vec = np.concatenate([np.array(last_left).reshape(-1), np.array(last_right).reshape(-1)])
                    elif last_left is not None:
                        flat_vec = np.concatenate([np.array(last_left).reshape(-1), np.zeros((63,))])
                    elif last_right is not None:
                        flat_vec = np.concatenate([np.zeros((63,)), np.array(last_right).reshape(-1)])
                    else:
                        # final fallback: zeros
                        flat_vec = np.zeros((126,), dtype=np.float32)

                # Now ensure flat_vec is a 126-length array and split
                flat_arr = np.asarray(flat_vec).reshape(-1)
                if flat_arr.size < 126:
                    tmp = np.zeros((126,), dtype=np.float32)
                    tmp[: flat_arr.size] = flat_arr
                    flat_arr = tmp
                left_f, right_f = self._flat_to_left_right(flat_arr)

                # Append to buffers
                left_buffer.append(left_f)
                right_buffer.append(right_f)

                # Keep buffers at most SEQUENCE_LENGTH
                if len(left_buffer) > SEQUENCE_LENGTH:
                    left_buffer.pop(0)
                if len(right_buffer) > SEQUENCE_LENGTH:
                    right_buffer.pop(0)

                # Save when we have a full sequence
                if len(left_buffer) == SEQUENCE_LENGTH and len(right_buffer) == SEQUENCE_LENGTH:
                    self._save_left_right_json(left_buffer.copy(), right_buffer.copy(), sign_label)
                    sequences_saved += 1
                    left_buffer = []
                    right_buffer = []
                    if max_sequences is not None and sequences_saved >= max_sequences:
                        break

                if max_frames is not None and frame_idx >= max_frames:
                    break

        finally:
            cap.release()
            try:
                self.detector.close()
            except Exception:
                pass

        if self.verbose:
            print(f"Finished processing. frames={frame_idx}, sequences_saved={sequences_saved}")
        return sequences_saved


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Collect hand-landmark sequences from a video file")
    parser.add_argument("--video", required=True, help="Path to the input video file (.mp4, etc)")
    parser.add_argument("--label", required=True, help="Sign label to attach to the generated filename")
    parser.add_argument("--out", default=None, help="Output directory for saved sequences (default=data/raw_sequences)")
    parser.add_argument("--max-sequences", type=int, default=None, help="Stop after saving this many sequences")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after processing this many frames")
    parser.add_argument("--skip-frames", type=int, default=0, help="Skip N frames between processed frames (default 0)")
    parser.add_argument("--no-verbose", dest="verbose", action="store_false", help="Turn off progress prints")

    args = parser.parse_args()

    collector = VideoSequenceCollector(out_dir=args.out, verbose=args.verbose)
    collector.process_video(args.video, args.label, max_sequences=args.max_sequences, max_frames=args.max_frames, skip_frames=args.skip_frames)


if __name__ == "__main__":
    _cli()
