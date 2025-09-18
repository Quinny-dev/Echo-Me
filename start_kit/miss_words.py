import os
import json


filenames = set(os.listdir(r"C:\Users\Ken\Documents\GitHub\Echo-Me\start_kit\videos_other"))

content = json.load(open(r"C:\Users\Ken\Documents\GitHub\Echo-Me\start_kit\WLASL_v0.3.json"))

words = []

for entry in content:
    instances = entry['instances']

    for inst in instances:
        video_id = inst['video_id']
        if video_id + '.mp4' in filenames:
            words.append(entry['gloss'])
            break

missing_words = []

for entry in content:

    check_word = entry['gloss']

    if check_word not in words:
        missing_words.append(entry['gloss'])


missing_words.sort()

with open('missing_words.txt', 'w') as f:
    f.write('\n'.join(missing_words))

