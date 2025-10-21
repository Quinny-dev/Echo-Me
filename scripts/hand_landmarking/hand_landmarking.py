import os
import sys
import cv2

# Ensure project root is importable
HERE = os.path.abspath(os.path.dirname(__file__))
SCRIPTS_DIR = os.path.abspath(os.path.join(HERE, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import the hand landmark detector and draw utility
from hand_landmarking.hand_landmark_detector import HandLandmarkDetector
from hand_landmarking.utils import draw_landmarks


def main():
    detector = HandLandmarkDetector(static_image_mode=False)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam (index 0).")
        return

    print("Press ESC or 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame capture failed.")
                break

            # Detect hand landmarks
            try:
                mp_res = detector.process(frame)
                if getattr(mp_res, "multi_hand_landmarks", None):
                    frame = draw_landmarks(frame, mp_res.multi_hand_landmarks)
            except Exception:
                pass  # ignore if detection fails

            # Show the frame
            cv2.imshow("Hand Landmarks - Press ESC/q to exit", frame)
            key = cv2.waitKey(1) & 0xFF

            # Exit
            if key == 27 or key == ord('q'):  # ESC or 'q'
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    main()