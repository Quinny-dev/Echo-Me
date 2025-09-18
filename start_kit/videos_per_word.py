import os
import json

filenames = set(os.listdir(r"C:\Users\Ken\Documents\GitHub\Echo-Me\start_kit\videos_new"))

content = json.load(open(r"C:\Users\Ken\Documents\GitHub\Echo-Me\start_kit\WLASL_v0.3.json"))

words = []

wordWW = []

numbers = []

combined = []

muchNum = []

muchOr= []

less=[]

lessWord = []

lessCombined = []

for entry in content:
    instances = entry['instances']

    counter = 0

    for inst in instances:
        video_id = inst['video_id']

        if video_id + '.mp4' in filenames:

            if entry['gloss'] not in words:
                words.append(entry['gloss'])
           
            counter += 1

            if entry['gloss'] == "wash face":
                print(video_id)

            word = entry['gloss']

    if counter < 3:
        less.append(counter)
        lessWord.append(word)

    numbers.append(counter)
    wordWW.append(word)

for i in range(21):

    counter = 0
    
    for number in numbers:
        if i == number:
            counter += 1
        if number not in muchNum:
            muchNum.append(number)

    muchNum.reverse()
    tempT = muchNum[0]
    muchNum.reverse()
    muchOr.append(f"{i} : {counter}")


for i in range(len(words)):
    tempString = (f"{words[i]} : {numbers[i]}")
    combined.append(tempString)

for i in range(len(lessWord)):
    tempString = (f"{lessWord[i]} : {less[i]}")
    lessCombined.append(tempString)


word_number_pairs = list(zip(wordWW, numbers))
word_number_pairs.sort(key=lambda x: x[1], reverse=True)

# Then create the combined strings
combined = [f"{wordWW} : {number}" for wordWW, number in word_number_pairs]

words.sort()

with open('video_per_word_NEW.txt', 'w') as f:
    f.write('\n'.join(combined))

with open('much_NEW.txt', 'w') as f:
    f.write('\n'.join(muchOr))

with open('need_NEW.txt', 'w') as f:
    f.write('\n'.join(lessCombined))

