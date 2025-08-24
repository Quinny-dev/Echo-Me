
import os
import sys
import time
from datetime import datetime
import numpy as np
import cv2

# ensure project root is importable (so imports below work when running from this file)
HERE = os.path.abspath(os.path.dirname(__file__))            # .../scripts/hand_landmarking
SCRIPTS_DIR = os.path.abspath(os.path.join(HERE, ".."))      # .../scripts
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))  # repo root

if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Now safe to import modules using the project's package layout
from hand_landmarking.hand_landmark_detector import HandLandmarkDetector
from hand_landmarking.utils import draw_landmarks

# exporter lives in data_pipeline.utils (we added export_sequence_json there)
from data_pipeline.utils import export_sequence_json, DATA_DIR

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def timestamp_str():
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

def try_flatten_from_detector(detector, frame):
    """
    Attempt to obtain a flat per-frame vector from the detector.
    Priority:
      1) detector.detect_landmarks(frame) -> expected flattened (63 or 126)
      2) detector.process(frame) + hand_landmarking.utils.mp_results_to_canonical_two_hand_vector
         (if available in your utils)
    Returns:
      - 1D numpy array (dtype=float) or None if detection failed for this frame.
    """
    # 1) try the convenient wrapper
    try:
        vec = detector.detect_landmarks(frame)
        if vec is not None:
            arr = np.array(vec, dtype=float).reshape(-1)
            if arr.size in (63, 126) or (arr.size % 63 == 0):
                return arr
    except Exception:
        pass

    # 2) try lower-level processing -> mp results -> canonicalization util
    try:
        mp_res = detector.process(frame)
        # import here to avoid hard dependency if not present
        from hand_landmarking.utils import mp_results_to_canonical_two_hand_vector
        flat = mp_results_to_canonical_two_hand_vector(mp_res)  # should return list/ndarray
        if flat is not None:
            arr = np.array(flat, dtype=float).reshape(-1)
            return arr
    except Exception:
        pass

    # detection failed or no useful vector
    return None


def main():
    detector = HandLandmarkDetector(static_image_mode=False)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam (index 0).")
        return

    print("Test hand landmarking - Press SPACE to start/stop recording. ESC or 'q' to quit.")
    recording = False
    frames = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame capture failed, exiting.")
                break

            # draw landmarks for visual feedback if detector can produce mp results
            try:
                mp_res = detector.process(frame)
                if getattr(mp_res, "multi_hand_landmarks", None):
                    frame = draw_landmarks(frame, mp_res.multi_hand_landmarks)
            except Exception:
                # ignore: draw only when available
                pass

            # show status on frame
            status_text = "REC" if recording else "READY"
            cv2.putText(frame, f"Status: {status_text} (SPACE to toggle, ESC/q to quit)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if recording else (0, 255, 255), 2)

            cv2.imshow("Webcam - Press ESC to exit", frame)
            key = cv2.waitKey(1) & 0xFF

            # handle keys
            if key == 27 or key == ord('q'):  # ESC or q -> quit
                if recording and frames:
                    # save before quitting
                    print("\nQuitting: saving current recording...")
                    save_sequence(frames, label="test")
                break

            if key == 32:  # SPACE toggles recording
                recording = not recording
                if recording:
                    frames = []  # reset capture buffer
                    print("Recording started...")
                else:
                    # stopped recording -> save if we have frames
                    if frames:
                        print("Recording stopped. Saving sequence...")
                        save_sequence(frames, label="test")
                    else:
                        print("Recording stopped. No frames captured.")
                # small debounce to avoid double toggles
                time.sleep(0.2)

            # if recording, capture the flattened vector for this frame
            if recording:
                vec = try_flatten_from_detector(detector, frame)
                if vec is None:
                    # append a zero vector of appropriate length if we have previously captured frames
                    if frames:
                        vec = np.zeros_like(np.array(frames[-1], dtype=float))
                        frames.append(vec)
                        print(".", end="", flush=True)
                    else:
                        # no detection yet; skip this frame
                        print(" (skip)", end="", flush=True)
                else:
                    frames.append(vec)
                    print(".", end="", flush=True)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


def save_sequence(frames_list, label="test"):
    """
    Convert a list of per-frame 1D arrays/lists into an ndarray and call export_sequence_json.
    Files are written into DATA_DIR by default: data/raw_sequences/test_<timestamp>.json
    """
    if not frames_list:
        print("No frames to save.")
        return None

    seq = np.vstack([np.array(f).reshape(1, -1) for f in frames_list]).astype(float)
    ensure_dir(DATA_DIR)
    filename = f"{label}_{timestamp_str()}.json"
    outpath = os.path.join(DATA_DIR, filename)
    try:
        export_sequence_json(seq, outpath, filename_hint=filename)
        print("\nSaved JSON to:", outpath)
        return outpath
    except Exception as e:
        print("Failed to save JSON:", e)
        return None


if __name__ == "__main__":
    main()
