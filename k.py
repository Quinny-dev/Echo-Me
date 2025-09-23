import cv2
import os
import sys
import time
import numpy as np

# Add the project root to sys.path for absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import other modules from their actual folders
from hand_landmarking.hand_landmark_detector import HandLandmarkDetector
from hand_landmarking.utils import draw_landmarks
from data_pipeline.sequence_buffer import SequenceBuffer
from data_pipeline.utils import SEQUENCE_LENGTH, DATA_DIR

# Initialize modules
detector = HandLandmarkDetector()
buffer = SequenceBuffer(SEQUENCE_LENGTH)
os.makedirs(DATA_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)
print("All systems ready.")

# Prompt user *after* initialization
label = input("\nEnter the label for this sign (e.g. hello): ").strip().lower()
recording = False
counting_down = False
sequence_saved = False
countdown_start_time = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    display_frame = frame.copy()

    key = cv2.waitKey(1) & 0xFF

    # ESC = exit
    if key == 27:
        break

    # SPACE = start countdown
    if key == 32 and not recording and not counting_down:
        counting_down = True
        countdown_start_time = time.time()

    # Countdown logic
    if counting_down and not recording:
        seconds_elapsed = int(time.time() - countdown_start_time)
        countdown_remaining = 3 - seconds_elapsed

        if countdown_remaining > 0:
            countdown_text = f"Recording starts in {countdown_remaining}..."
            cv2.putText(
                display_frame, countdown_text, (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4, cv2.LINE_AA
            )
        else:
            recording = True
            counting_down = False

    # Recording logic
    elif recording and not sequence_saved:
        landmarks = detector.detect_landmarks(frame)

        if landmarks is not None:
            buffer.add_frame(landmarks)

        # Draw landmarks
        results = detector.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame = draw_landmarks(frame, results.multi_hand_landmarks)
        display_frame = frame

        # Save when enough frames are collected
        if len(buffer) >= SEQUENCE_LENGTH:
            sequence = buffer.get_sequence()
            file_path = os.path.join(DATA_DIR, f"{label}_{int(time.time())}.npy")
            np.save(file_path, sequence)
            print(f"Saved sequence to {file_path}")
            sequence_saved = True

    # Not recording yet
    if not recording and not counting_down:
        cv2.putText(
            display_frame, "Press SPACE to start capturing", (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
        )

    # Display frame
    cv2.imshow("Sequence Collector", display_frame)

    if sequence_saved:
        time.sleep(2)
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
detector.close()