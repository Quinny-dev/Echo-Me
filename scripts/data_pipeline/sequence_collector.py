
import cv2
import os
import sys
import time
import uuid
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hand_landmarking.hand_landmark_detector import HandLandmarkDetector
from hand_landmarking.utils import draw_landmarks, mp_results_to_canonical_two_hand_vector
from hand_landmarking.optical_flow import (
    track_landmarks_lk,
    extract_left_right_from_mp_results,
    build_flat_from_slots,
)
from data_pipeline.sequence_buffer import SequenceBuffer
from data_pipeline.utils import SEQUENCE_LENGTH, DATA_DIR

def run_collector(device_index=0, save_dir=DATA_DIR, sequence_length=SEQUENCE_LENGTH, target_width=None):
    
    os.makedirs(save_dir, exist_ok=True)
    detector = HandLandmarkDetector(static_image_mode=False, max_num_hands=2)
    buf = SequenceBuffer(max_length=sequence_length)
    presence_history = []
    propagated_history = []

    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {device_index}")

    prev_gray = None
    # previous per-hand pixel coords and z
    prev_left_xy = None; prev_left_z = None
    prev_right_xy = None; prev_right_z = None
    prev_presence = (0, 0)  # last frame presence (left, right)

    collecting = False
    saved_count = 0
    print("Controls: SPACE toggles capture ON/OFF. ESC quits. When buffer full and capturing, auto-save.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame read failed, exiting.")
            break

        # optionally upscale small frames to target_width for better detection
        if target_width is not None:
            h, w = frame.shape[:2]
            if w < target_width:
                scale = target_width / float(w)
                frame_proc = cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
            else:
                frame_proc = frame
        else:
            frame_proc = frame

        gray = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2GRAY)
        results = detector.process(frame_proc)

        # defaults for this frame
        left_xy = None; left_z = None; right_xy = None; right_z = None
        propagated_flags = np.array([0,0], dtype=np.int8)

        # If MediaPipe returned landmarks -> extract pixel coords & z
        if results is not None and getattr(results, "multi_hand_landmarks", None):
            left_xy, left_z, right_xy, right_z, presence = extract_left_right_from_mp_results(results, frame_proc.shape)
            # update prev references for future optical-flow
            prev_left_xy, prev_left_z = (left_xy.copy() if left_xy is not None else None,
                                         left_z.copy() if left_z is not None else None)
            prev_right_xy, prev_right_z = (right_xy.copy() if right_xy is not None else None,
                                           right_z.copy() if right_z is not None else None)
            prev_presence = (int(bool(presence[0])), int(bool(presence[1])))
        else:
            # If no detection this frame, attempt to propagate previously-seen landmarks using optical flow
            if prev_gray is not None:
                # try propagate left slot if we had it
                if prev_left_xy is not None:
                    p1_left, st_left = track_landmarks_lk(prev_gray, gray, prev_left_xy)
                    if p1_left is not None and st_left is not None:
                        # require at least some points tracked successfully (tunable threshold)
                        if st_left.sum() >= 6:  # at least 6 points tracked -> accept propagated hand
                            left_xy = p1_left
                            left_z = prev_left_z.copy() if prev_left_z is not None else np.zeros((21,), dtype=np.float32)
                            propagated_flags[0] = 1
                        else:
                            left_xy = None; left_z = None
                    else:
                        left_xy = None; left_z = None
                # try propagate right slot if we had it
                if prev_right_xy is not None:
                    p1_right, st_right = track_landmarks_lk(prev_gray, gray, prev_right_xy)
                    if p1_right is not None and st_right is not None:
                        if st_right.sum() >= 6:
                            right_xy = p1_right
                            right_z = prev_right_z.copy() if prev_right_z is not None else np.zeros((21,), dtype=np.float32)
                            propagated_flags[1] = 1
                        else:
                            right_xy = None; right_z = None

                # update prev_... only if propagation happened (so next frame can chain)
                if propagated_flags[0] == 1:
                    prev_left_xy = left_xy.copy()
                    # prev_left_z already preserved
                if propagated_flags[1] == 1:
                    prev_right_xy = right_xy.copy()
            # if prev_gray is None or propagation failed, presence remains zeros

            # presence: derived from propagation success
            presence = (int(bool(left_xy is not None)), int(bool(right_xy is not None)))
            prev_presence = presence

        # Build canonical flat vector (normalized coords) and per-frame presence mask
        flat_vec, presence_mask = build_flat_from_slots(left_xy, left_z, right_xy, right_z, frame_proc.shape, normalize=True)
        buf.append(flat_vec)
        presence_history.append(presence_mask.astype(np.int8))
        propagated_history.append(propagated_flags)

        # Draw landmarks for display using MediaPipe results if present, else draw propagated points
        display = frame_proc.copy()
        try:
            if results is not None and getattr(results, "multi_hand_landmarks", None):
                draw_landmarks(display, results.multi_hand_landmarks)
            else:
                # draw propagated points so user sees fallback (small circles)
                if left_xy is not None:
                    for (x,y) in left_xy:
                        cv2.circle(display, (int(x), int(y)), 3, (0,255,0), -1)
                if right_xy is not None:
                    for (x,y) in right_xy:
                        cv2.circle(display, (int(x), int(y)), 3, (0,200,255), -1)
        except Exception:
            pass

        # overlay info
        cv2.putText(display, f"Buffer: {len(buf)}/{sequence_length}  Saved: {saved_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
        if propagated_flags.any():
            cv2.putText(display, f"PROPAGATED L:{propagated_flags[0]} R:{propagated_flags[1]}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 200, 50), 2)
        if detecting := (results is not None and getattr(results, "multi_hand_landmarks", None)):
            cv2.putText(display, "DETECTED", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Sequence Collector (with optical-flow fallback)", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE toggles capture
            collecting = not collecting
            print("Collecting:", collecting)

        # Auto-save when buffer full and user toggled collecting ON
        if collecting and len(buf) >= sequence_length:
            seq = buf.get_sequence(pad_front=True)  # (sequence_length, 126)
            # build presence mask and propagated mask arrays
            last_presence = np.stack(presence_history[-sequence_length:], axis=0).astype(np.int8)
            last_propagated = np.stack(propagated_history[-sequence_length:], axis=0).astype(np.int8)
            from data_pipeline.utils import export_sequence_json

            basename = f"sequence_{uuid.uuid4().hex}"
            json_path = os.path.join(save_dir, basename + ".json")
            export_sequence_json(seq, json_path, filename_hint=basename)
            print("Saved JSON:", json_path)
            saved_count += 1

            # clear / reset
            buf.clear()
            presence_history = []
            propagated_history = []
            time.sleep(0.5)

        # update prev_gray for next iteration (use proc-frame gray)
        prev_gray = gray.copy()

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

if __name__ == "__main__":
    run_collector()
