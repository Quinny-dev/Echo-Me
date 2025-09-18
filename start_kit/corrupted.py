import subprocess
import os

def is_video_corrupted(video_path):
    command = [
        'ffmpeg',
        '-v', 'error',
        '-i', video_path,
        '-f', 'null',
        '-'
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            return True
        if "error" in result.stderr.lower() or "corrupt" in result.stderr.lower():
            return True
        return False
    except FileNotFoundError:
        print("FFmpeg not found. Please ensure it's installed and in your PATH.")
        return None
    except Exception as e:
        print(f"An error occurred while checking {video_path}: {e}")
        return None


video_dir = r"C:/Users/Ken/Documents/GitHub/Echo-Me/start_kit/videos_new"
video_files = os.listdir(video_dir)
corrupted_videos = []

total_files = len(video_files)

for idx, video in enumerate(video_files, start=1):
    video_path = os.path.join(video_dir, video)
    print(f"Checking file {idx}/{total_files}: {video}")  # 👈 ID + filename
    
    if is_video_corrupted(video_path):
        corrupted_videos.append(video)

# Print summary
print("\nCorrupted videos found:", corrupted_videos)

# Save to text file
output_file = "corrupted_new.txt"
with open(output_file, "w", encoding="utf-8") as f:
    for video in corrupted_videos:
        f.write(video + "\n")

print(f"\nCorrupted video list saved to {output_file}")
