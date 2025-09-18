import os
import json


filenames = set(os.listdir(r"C:\Users\Ken\Documents\GitHub\Echo-Me\start_kit\videos_new"))

content = json.load(open(r"C:\Users\Ken\Documents\GitHub\Echo-Me\start_kit\WLASL_v0.3.json"))

words = []

for entry in content:
    instances = entry['instances']

    for inst in instances:
        video_id = inst['video_id']
        if video_id + '.mp4' in filenames:
            words.append(entry['gloss'])
            break


words.sort()

with open('all_words_new.txt', 'w') as f:
    f.write('\n'.join(words))

