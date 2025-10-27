import os
import sys
import cv2

# Add project directories to Python path for imports
HERE = os.path.abspath(os.path.dirname(__file__))
SCRIPTS_DIR = os.path.abspath(os.path.join(HERE, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

# Ensure project root is importable
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import the hand landmark detector and drawing utilities
from hand_landmarking.hand_landmark_detector import HandLandmarkDetector
from hand_landmarking.utils import draw_landmarks


def main():
    """Main function to run live hand landmark detection from webcam."""
    # Initialize detector for video mode (not static images)
    detector = HandLandmarkDetector(static_image_mode=False)
    
    # Open webcam (index 0 is default camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam (index 0).")
        return

    print("Press ESC or 'q' to quit.")
    try:
        while True:
            # Capture frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Frame capture failed.")
                break

            # Detect hand landmarks in current frame
            try:
                mp_res = detector.process(frame)
                # Draw landmarks if hands are detected
                if getattr(mp_res, "multi_hand_landmarks", None):
                    frame = draw_landmarks(frame, mp_res.multi_hand_landmarks)
            except Exception:
                pass  # ignore if detection fails

            # Display the frame with landmarks
            cv2.imshow("Hand Landmarks - Press ESC/q to exit", frame)
            key = cv2.waitKey(1) & 0xFF

            # Check for exit keys (ESC or 'q')
            if key == 27 or key == ord('q'):  # ESC or 'q'
                break

    finally:
        # Clean up resources
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    main()