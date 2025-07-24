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
from data_pipeline.utils import normalize_landmarks_single, sliding_window_blocks

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

cv2.namedWindow("Sequence Collector", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Sequence Collector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

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
        # get raw 126‑vector → reshape to two hands of 21×3
        landmarks = detector.detect_landmarks(frame)
        if landmarks is not None:
            arr = np.array(landmarks, dtype=np.float32).reshape(2, 21, 3)

            # normalize each hand separately
            for h in range(arr.shape[0]):
                arr[h] = normalize_landmarks_single(arr[h])

            # flatten back to length‑126 and add
            buffer.add_frame(arr.flatten().tolist())

        # Draw landmarks
        results = detector.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame = draw_landmarks(frame, results.multi_hand_landmarks)
        display_frame = frame

    # Save when enough frames are collected
    if len(buffer) >= SEQUENCE_LENGTH:
        # 1) get your buffered data: shape (SEQUENCE_LENGTH, 126)
        seq_flat = buffer.get_sequence()  
        # 2) reshape to (T, 2, 21, 3)
        seq = seq_flat.reshape(SEQUENCE_LENGTH, 2, 21, 3)

        # 3) slice into overlapping windows
        STRIDE = 5  # choose your stride
        windows = sliding_window_blocks(seq, SEQUENCE_LENGTH, STRIDE)

        # 4) save each window back flattened
        for idx, w in enumerate(windows):
            fname = f"{label}_{int(time.time())}_w{idx}.npy"
            out_path = os.path.join(DATA_DIR, fname)
            np.save(out_path, w.reshape(SEQUENCE_LENGTH, 126))
            print(f"Saved window {idx} to {out_path}")

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