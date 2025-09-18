import os
import json

filenames = set(os.listdir(r"C:\Users\Ken\Documents\GitHub\Echo-Me\start_kit\videos_new"))
content = json.load(open(r"C:\Users\Ken\Documents\GitHub\Echo-Me\start_kit\WLASL_v0.3.json"))

words = []
entries = []
tempEntries = []

combined = []

for entry in content:
    instances = entry['instances']

    for inst in instances:
        video_id = inst['video_id']

        if video_id + '.mp4' in filenames:
            if entry['gloss'] not in words:
                words.append(entry['gloss'])
                tempEntries.append(video_id + '.mp4')
            else:
                tempEntries.append(video_id + '.mp4')

    entries.append(tempEntries.copy())
    tempEntries.clear()

data = {key: value for key, value in zip(words, entries)}

with open("test.json", "w") as f:
    json.dump(data, f, indent=4)


