import cv2
import threading
import queue
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')

class VideoPipeline:
    def __init__(self, source=0, frame_width=640, frame_height=480, max_queue_size=1):

        self.source = source
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.max_queue_size = max_queue_size
        
        # Queues for pipeline stages (use limited size to drop frames if processing is slow)
        self.raw_queue = queue.Queue(maxsize=self.max_queue_size)
        self.preprocessed_queue = queue.Queue(maxsize=self.max_queue_size)
        self.analyzed_queue = queue.Queue(maxsize=self.max_queue_size)
        
        # Stop event for graceful shutdown
        self.stop_event = threading.Event()
        
        # Capture device
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open video source: {self.source}")
        
        # Set capture properties if applicable
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        # Threads for each stage
        self.capture_thread = threading.Thread(target=self._capture_loop, name="CaptureThread")
        self.preprocess_thread = threading.Thread(target=self._preprocess_loop, name="PreprocessThread")
        self.analyze_thread = threading.Thread(target=self._analyze_loop, name="AnalyzeThread")
        self.display_thread = threading.Thread(target=self._display_loop, name="DisplayThread")
        
        # FPS tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = None

    def start(self):
        """Start the pipeline threads."""
        logging.info("Starting video pipeline...")
        self.start_time = time.time()
        self.capture_thread.start()
        self.preprocess_thread.start()
        self.analyze_thread.start()
        self.display_thread.start()

    def stop(self):
        """Stop the pipeline."""
        logging.info("Stopping video pipeline...")
        self.stop_event.set()
        self.capture_thread.join()
        self.preprocess_thread.join()
        self.analyze_thread.join()
        self.display_thread.join()
        self.cap.release()
        cv2.destroyAllWindows()
        logging.info("Pipeline stopped.")

    def _capture_loop(self):
        """Capture frames from the video source and put them into the raw queue."""
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                logging.warning("Failed to capture frame. Stopping capture.")
                self.stop_event.set()
                break
            try:
                self.raw_queue.put_nowait(frame)
            except queue.Full:
                pass  # Drop frame if queue is full (real-time priority)
                logging.info("Capture loop ended.")

    def _preprocess_loop(self):
        """Preprocess frames: resize and convert to grayscale."""
        while not self.stop_event.is_set():
            try:
                frame = self.raw_queue.get(timeout=0.1)
                # Resize
                frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                # Convert to grayscale (example preprocessing)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.preprocessed_queue.put_nowait(frame)
            except queue.Empty:
                continue
            except queue.Full:
                pass  # Drop if full
                logging.info("Preprocess loop ended.")

    def _analyze_loop(self):
        """Analyze frames: apply edge detection as an example."""
        while not self.stop_event.is_set():
            try:
                frame = self.preprocessed_queue.get(timeout=0.1)
                edges = cv2.Canny(frame, 100, 200)
                self.analyzed_queue.put_nowait(edges)
            except queue.Empty:
                continue
            except queue.Full:
                pass  # Drop if full
                logging.info("Analyze loop ended.")

    def _display_loop(self):
        """Display analyzed frames and calculate FPS."""
        while not self.stop_event.is_set():
            try:
                frame = self.analyzed_queue.get(timeout=0.1)
                # Display the frame
                cv2.imshow('Real-Time Video Pipeline', frame)
                
                # FPS calculation
                self.frame_count += 1
                elapsed_time = time.time() - self.start_time
                self.fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
                
                # Log FPS every 10 frames
                if self.frame_count % 10 == 0:
                    logging.info(f"Current FPS: {self.fps:.2f}")
                
                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop_event.set()
            except queue.Empty:
                continue
                logging.info("Display loop ended.")