# Camera feed handler for Qt-based video display
import cv2
from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage, QPixmap

class CameraFeed:
    def __init__(self, label, camera_index=0, fps=30, 
                 frame_callback=None, hand_detector=None):
        """
        label: QLabel where frames will be displayed
        frame_callback: optional callback(frame, landmarks) for processing model input
        hand_detector: object with methods:
            - process(frame) -> mediapipe results
            - draw_landmarks(frame, landmarks) -> frame with overlay
        """
        self.label = label
        self.cap = cv2.VideoCapture(camera_index)
        self.frame_callback = frame_callback
        self.hand_detector = hand_detector
        self.draw_landmarks_flag = True  # toggle for overlay
        # Set up timer for periodic frame updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(int(1000 / fps))

    def set_draw_landmarks(self, enabled: bool):
        """Enable or disable the hand overlay."""
        self.draw_landmarks_flag = enabled

    # Update camera frame and process for display
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        landmarks = None

        # Process frame for hand landmarks if detector is provided
        if self.hand_detector:
            try:
                results = self.hand_detector.process(frame)
                landmarks = results

                # Draw landmarks overlay if enabled
                if self.draw_landmarks_flag and results is not None:
                    try:
                        # Check if detector has built-in draw method
                        if hasattr(self.hand_detector, "draw_landmarks"):
                            frame = self.hand_detector.draw_landmarks(frame, results)
                        else:
                            # Use external utility function as fallback
                            from hand_landmarking.utils import draw_landmarks
                            if getattr(results, "multi_hand_landmarks", None):
                                frame = draw_landmarks(frame, results.multi_hand_landmarks)
                    except Exception as e:
                        print("Error drawing landmarks:", e)
            except Exception as e:
                print("Error processing frame for landmarks:", e)

        # Execute callback for additional frame processing
        if self.frame_callback:
            try:
                self.frame_callback(frame, landmarks)
            except Exception as e:
                print("Error in frame callback:", e)

        # Convert OpenCV frame to Qt format and display
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(qt_img))
        except Exception as e:
            print("Error displaying frame:", e)

    # Clean up camera resources
    def release(self):
        self.timer.stop()
        self.cap.release()