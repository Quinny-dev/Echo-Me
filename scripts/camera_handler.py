"""
Camera and Video Handling Module
Handles camera feed and hand landmark detection
"""

from camera_feed import CameraFeed
from hand_landmarking.hand_landmarking import HandLandmarkDetector
from user_data import signal_ready


class CameraHandler:
    """Handles camera and hand detection functionality"""
    
    def __init__(self, camera_label):
        self.camera_label = camera_label
        self.hand_detector = None
        self.camera = None
        self.show_landmarks = True
        
        self.initialize_components()
    
    def initialize_components(self):
        """Initialize hand detector and camera feed"""
        try:
            # Initialize hand detector
            self.hand_detector = HandLandmarkDetector(static_image_mode=False)
            
            # Initialize camera feed
            self.camera = CameraFeed(
                self.camera_label,
                frame_callback=self.process_frame,
                hand_detector=self.hand_detector
            )
            
            # Signal that the GUI is ready
            signal_ready()
            
            return True
            
        except Exception as e:
            print(f"Error initializing camera components: {e}")
            return False
    
    def process_frame(self, frame, landmarks):
        """Process camera frame with hand landmarks"""
        # This method can be extended to handle frame processing
        # Currently just passes through to camera feed
        pass
    
    def set_landmark_visibility(self, show_landmarks):
        """Set whether to show hand landmarks overlay"""
        self.show_landmarks = show_landmarks
        if self.camera:
            self.camera.set_draw_landmarks(show_landmarks)
        print(f"Hand landmark overlay {'enabled' if show_landmarks else 'disabled'}")
    
    def start_camera(self):
        """Start the camera feed"""
        if self.camera:
            try:
                self.camera.start()
                return True
            except Exception as e:
                print(f"Error starting camera: {e}")
                return False
        return False
    
    def stop_camera(self):
        """Stop the camera feed"""
        if self.camera:
            try:
                self.camera.stop()
                return True
            except Exception as e:
                print(f"Error stopping camera: {e}")
                return False
        return False
    
    def cleanup(self):
        """Clean up camera resources"""
        try:
            if self.camera:
                self.camera.stop()
                self.camera = None
            
            if self.hand_detector:
                self.hand_detector = None
                
            print("Camera components cleaned up successfully")
            return True
            
        except Exception as e:
            print(f"Error during camera cleanup: {e}")
            return False
    
    def is_camera_active(self):
        """Check if camera is currently active"""
        return self.camera is not None and hasattr(self.camera, 'is_running') and self.camera.is_running
    
    def get_camera_status(self):
        """Get current camera status information"""
        if not self.camera:
            return {"status": "not_initialized", "message": "Camera not initialized"}
        
        if self.is_camera_active():
            return {"status": "active", "message": "Camera feed is running"}
        else:
            return {"status": "inactive", "message": "Camera feed is stopped"}