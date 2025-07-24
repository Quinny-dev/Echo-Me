import cv2
from hand_landmark_detector import HandLandmarkDetector
from utils import draw_landmarks, count_detected_hands

# Initialize the hand landmark detector.
detector = HandLandmarkDetector()

# Start video capture from webcam (index 0).
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from webcam.")
        break

    # Detect hand landmarks.
    landmarks = detector.detect_landmarks(frame)

    if landmarks:
         # Count how many hands are detected using utility function.
        num_hands = count_detected_hands(landmarks)

        if num_hands == 0:
            print("No hands detected in this frame.")
        elif num_hands == 1:
            print("One hand detected in this frame.")
        else:
            print("Two hands detected in this frame.")

        # Convert BGR to RGB for MediaPipe drawing.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.hands.process(rgb_frame)

        # Draw hand landmarks on the frame.
        frame = draw_landmarks(frame, results.multi_hand_landmarks)
    else:
        print("No hands detected.")

    # Show the annotated frame.
    cv2.imshow("Webcam - Press ESC to exit", frame)

    # Exit loop when ESC (keycode 27) is pressed.
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Release resources and close windows.
cap.release()
cv2.destroyAllWindows()
detector.close()