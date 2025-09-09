import os
import json
from tqdm import tqdm
from pathlib import Path

from scripts.hand_landmarking.video_sequence_collector import VideoSequenceCollector

# ---- Config ----
LABEL_VIDEO_MAP = Path(r"start_kit\label_video_map.json")
VIDEO_BASE_DIR = Path(r"C:\Users\juano\Downloads\videos")  # folder containing .mp4 files
OUTPUT_DIR = Path("Data")
OUTPUT_DIR.mkdir(exist_ok=True)

SEQUENCES_JSON = OUTPUT_DIR / "sequences_by_label.json"
MISSING_JSON = OUTPUT_DIR / "missing_videos.json"

collector = VideoSequenceCollector(out_dir=str(OUTPUT_DIR), verbose=True)


# ---- Helpers ----
def load_json_safe(path, default_factory):
    if path.exists():
        with open(path, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return default_factory()
    return default_factory()


def save_progress():
    """Write sequences and missing videos to disk."""
    with open(SEQUENCES_JSON, "w") as f:
        json.dump(sequences_by_label, f, indent=2)
    with open(MISSING_JSON, "w") as f:
        json.dump(missing_videos, f, indent=2)


# ---- Load input + state ----
with open(LABEL_VIDEO_MAP, "r") as f:
    label_video_map = json.load(f)

sequences_by_label = load_json_safe(SEQUENCES_JSON, dict)
missing_videos = load_json_safe(
    MISSING_JSON, lambda: {label: [] for label in label_video_map.keys()}
)

# Count total videos for nicer progress display
total_videos = sum(len(v) for v in label_video_map.values())
print(f"Total labels: {len(label_video_map)}, total videos: {total_videos}\n")

# ---- Main loop ----
with tqdm(total=total_videos, desc="Overall Progress", unit="video") as overall_pbar:
    for label, video_list in tqdm(label_video_map.items(), desc="Labels", unit="label"):
        sequences_by_label.setdefault(label, {})
        missing_videos.setdefault(label, [])

        for video_file in tqdm(video_list, desc=f"{label}", leave=False, unit="video"):
            if video_file in sequences_by_label[label]:
                overall_pbar.update(1)
                continue  # already processed

            video_path = VIDEO_BASE_DIR / video_file
            if not video_path.exists():
                if video_file not in missing_videos[label]:
                    missing_videos[label].append(video_file)
                overall_pbar.update(1)
                continue

            # Reset state for each new video
            collector.prev_gray = None
            collector.prev_slots = None

            sequences = collector.process_video(str(video_path), sign_label=label)
            print(
                f"[INFO] Label={label}, Video={video_file}, "
                f"Frames={len(sequences[0]) if sequences else 0}, "
                f"Sequences Collected={len(sequences)}"
            )

            if not sequences:
                if video_file not in missing_videos[label]:
                    missing_videos[label].append(video_file)
            else:
                sequences_by_label[label][video_file] = sequences

            # Save progress after each video
            save_progress()
            overall_pbar.update(1)

# ---- Final save ----
save_progress()
print("\nProcessing complete. Sequences and missing videos saved.")