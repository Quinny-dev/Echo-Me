import json
from pathlib import Path

def generate_label_videoid_map(wlasl_json_path, output_path):
    # Load WLASL_v0.3.json
    with open(wlasl_json_path, "r") as f:
        wlasl_data = json.load(f)

    dataset = {}

    # Iterate over each entry (sign/label)
    for entry in wlasl_data:
        gloss = entry["gloss"]  # the label
        video_files = [f"{instance['video_id']}.mp4" for instance in entry["instances"]]

        dataset[gloss] = video_files

    # Save mapping JSON
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Label-to-video file mapping saved to {output_path}")

if __name__ == "__main__":
    # Paths relative to start_kit folder
    base_dir = Path(__file__).parent
    wlasl_json_path = base_dir / "WLASL_v0.3.json"
    output_path = base_dir / "label_video_map.json"

    generate_label_videoid_map(wlasl_json_path, output_path)
