import os
import json


filenames = set(os.listdir(r"C:\Users\Ken\Documents\GitHub\Echo-Me\start_kit\videos_other"))

content = json.load(open(r"C:\Users\Ken\Documents\GitHub\Echo-Me\start_kit\WLASL_v0.3.json"))

missing_ids = []

for entry in content:
    instances = entry['instances']

    for inst in instances:
        video_id = inst['video_id']
        if video_id + '.mp4' in filenames:
            missing_ids.append(video_id)


with open('missing_other.txt', 'w') as f:
    f.write('\n'.join(missing_ids))

