import json

content = json.load(open(r"C:\Users\Ken\Documents\GitHub\Echo-Me\start_kit\WLASL_v0.3.json"))

words = []

for entry in content:
    words.append(entry['gloss'])


words.sort()

with open('json_words.txt', 'w') as f:
    f.write('\n'.join(words))

