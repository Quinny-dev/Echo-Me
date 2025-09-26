#%pip install torch torchvision transformers decord opencv-python torchaudio sounddevice

import torch
import numpy as np
import cv2
import time
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
from torchvision.transforms import Compose, Lambda, Resize, Normalize
from decord import VideoReader, cpu
from decord.bridge import set_bridge
import sounddevice as sd
import os

# Set Decord to use PyTorch backend
set_bridge('torch')

# Load finetuned model and processor
model_ckpt = '/path/to/your/model/directory'  # Update this path
image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
model = VideoMAEForVideoClassification.from_pretrained(model_ckpt, ignore_mismatched_sizes=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Preprocessing parameters
mean = image_processor.image_mean
std = image_processor.image_std
clip_duration = 1.71  # seconds
fps = 30  # assume 30 FPS
num_frames = model.config.num_frames
frame_window = int(num_frames)  # number of frames to collect per clip

# Define video transformations
resize_to = (
    image_processor.size.get("height", 224),
    image_processor.size.get("width", 224),
)

val_transform = Compose([
    Lambda(lambda x: x / 255.0),
    Resize(resize_to, antialias=True),
    Lambda(lambda x: x.permute(1, 0, 2, 3)),  # (C, T, H, W) -> (T, C, H, W)
    Normalize(mean=mean, std=std),
])

# Predict gloss from a clip of frames
def predict_clip(frames: list):
    with torch.no_grad():
        # Convert to tensor: [T, H, W, C]
        video_tensor = torch.tensor(np.stack(frames))  # (T, H, W, C)
        video_tensor = video_tensor.permute(3, 0, 1, 2)  # (C, T, H, W)
        video_tensor = val_transform(video_tensor)  # (T, C, H, W)
        video_tensor = video_tensor.unsqueeze(0).to(device)  # (1, T, C, H, W)
        outputs = model(pixel_values=video_tensor)
        pred_id = outputs.logits.argmax(-1).item()
        return model.config.id2label[pred_id]


# Capture live video and predict
def live_classification():
    print("Starting live classification (press 'q' to quit)...")
    cap = cv2.VideoCapture(0)

    frame_buffer = []
    last_prediction_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize and convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (resize_to[1], resize_to[0]))
            frame_buffer.append(frame)

            # Show live feed
            cv2.imshow('Live ASL Gloss Detection', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            # If enough frames collected
            if len(frame_buffer) == frame_window:
                gloss = predict_clip(frame_buffer)
                print(f"Predicted Gloss: {gloss}")
                # Speak(gloss)  # Optional: say the word aloud
                frame_buffer = []  # Clear buffer
                time.sleep(0.3)  # Wait before next capture

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Exited live classification.")


if __name__ == "__main__":
    live_classification()