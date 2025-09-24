import cv2
import numpy as np
import os
import json
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
import functools
from tqdm import tqdm

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
# Initialize MediaPipe componen
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

def mediapipe_detection(image, model):
    """Function for detection."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    """Function to draw the landmarks."""
    # Draw face connections
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image, results.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
        ) 

    # Draw pose connections
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
        ) 

    # Draw left hand connections
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
        ) 

    # Draw right hand connections  
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        ) 

def concat_keypoints(results):
    """Concatenate all the landmarks into a single array."""
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh]).astype(np.float32)

def process_single_video(video_info):
    """
    Process a single video file - designed to be called by parallel processes.
    video_info is a tuple: (video_path, output_dir, target_fps, label)
    """
    video_path, output_dir, target_fps, label = video_info
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return f"Failed to open video: {video_path}"
        
        keypoints_all_frames = []

        # Get original video FPS
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip = max(int(orig_fps / target_fps), 1)
        frame_count = 0

        with mp_holistic.Holistic(min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue

                frame = cv2.resize(frame, (640, 480))

                # Make detections
                image, results = mediapipe_detection(frame, holistic)

                # Extract keypoints
                keypoints = concat_keypoints(results)
                keypoints_all_frames.append(keypoints)

        cap.release()

        # Save keypoints
        if keypoints_all_frames:
            keypoints_all_frames = np.array(keypoints_all_frames)
            os.makedirs(output_dir, exist_ok=True)
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(output_dir, f"{video_name}.npy")
            np.save(output_path, keypoints_all_frames)
            return f"Successfully processed: {video_path} -> {output_path} ({len(keypoints_all_frames)} frames)"
        else:
            return f"No frames processed for: {video_path}"
    
    except Exception as e:
        return f"Error processing {video_path}: {str(e)}"

def process_videos_parallel(json_file="label_video_map.json", 
                           base_video_path="Videos", 
                           base_output_dir="Data", 
                           target_fps=25, 
                           max_workers=None):
    """
    Process all videos in parallel using multiprocessing.
    
    Args:
        json_file: Path to the JSON file with video mappings
        base_video_path: Directory containing video files
        base_output_dir: Root directory to save keypoints
        target_fps: Target FPS for processing
        max_workers: Maximum number of parallel processes (None = use all CPU cores)
    """
    
    # Load the JSON mapping
    with open(json_file, "r") as f:
        video_json = json.load(f)

    # Prepare list of all videos to process
    video_tasks = []
    for label, videos in video_json.items():
        label_dir = os.path.join(base_output_dir, label)
        for video_file in videos:
            video_path = os.path.join(base_video_path, video_file)
            if os.path.exists(video_path):
                video_tasks.append((video_path, label_dir, target_fps, label))
            else:
                print(f"Warning: Video not found: {video_path}")

    print(f"Found {len(video_tasks)} videos to process")
    
    # Determine number of workers
    if max_workers is None:
        max_workers = min(cpu_count(), len(video_tasks))
    
    print(f"Using {max_workers} parallel processes")
    
    # Process videos in parallel
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_video = {
            executor.submit(process_single_video, video_task): video_task 
            for video_task in video_tasks
        }
        
        # Process completed tasks with progress bar
        completed = 0
        for future in tqdm(as_completed(future_to_video), 
                          total=len(video_tasks), 
                          desc="Processing videos"):
            video_task = future_to_video[future]
            try:
                result = future.result()
                print(f"✓ {result}")
            except Exception as exc:
                print(f"✗ Video {video_task[0]} generated an exception: {exc}")
            completed += 1
    
    end_time = time.time()
    print(f"\n🎉 Completed processing {len(video_tasks)} videos in {end_time - start_time:.2f} seconds")
    print(f"Average time per video: {(end_time - start_time) / len(video_tasks):.2f} seconds")

def process_videos_chunk_parallel(json_file="label_video_map.json", 
                                 base_video_path="Videos", 
                                 base_output_dir="Data", 
                                 target_fps=25, 
                                 chunk_size=None):
    """
    Alternative approach: Process videos in chunks using multiprocessing.Pool
    This can be more memory-efficient for very large datasets.
    """
    
    # Load the JSON mapping
    with open(json_file, "r") as f:
        video_json = json.load(f)

    # Prepare list of all videos to process
    video_tasks = []
    for label, videos in video_json.items():
        label_dir = os.path.join(base_output_dir, label)
        for video_file in videos:
            video_path = os.path.join(base_video_path, video_file)
            if os.path.exists(video_path):
                video_tasks.append((video_path, label_dir, target_fps, label))

    print(f"Found {len(video_tasks)} videos to process")
    
    # Determine chunk size
    num_processes = cpu_count()
    if chunk_size is None:
        chunk_size = max(1, len(video_tasks) // num_processes)
    
    print(f"Using {num_processes} processes with chunk size {chunk_size}")
    
    start_time = time.time()
    
    # Process in parallel using Pool
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_single_video, video_tasks, chunksize=chunk_size),
            total=len(video_tasks),
            desc="Processing videos"
        ))
    
    # Print results
    for result in results:
        if "Successfully" in result:
            print(f"✓ {result}")
        else:
            print(f"✗ {result}")
    
    end_time = time.time()
    print(f"\n🎉 Completed processing {len(video_tasks)} videos in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    # Example usage - choose one of these approaches:
    
    # Approach 1: ProcessPoolExecutor (recommended for better control and progress tracking)
    process_videos_parallel(
        json_file="label_video_map.json",
        base_video_path="Videos",
        base_output_dir="Data",
        target_fps=25,
        max_workers=None  # Use all available cores
    )
    
    # Approach 2: multiprocessing.Pool (alternative, sometimes more memory efficient)
    # process_videos_chunk_parallel(
    #     json_file="label_video_map.json",
    #     base_video_path="Videos", 
    #     base_output_dir="Data",
    #     target_fps=25,
    #     chunk_size=2
    # )