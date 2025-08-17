from openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

OpenAIAPIkey = os.getenv("OPENAIAPIKEY")

client = OpenAI(
  api_key=OpenAIAPIkey
)

df = pd.read_csv("gloss.csv")

prompt = """
You are an expert in various sign languages like American Sign Language (ASL), South African Sign Language (SASL) and English translation. 
Your task is to translate a sign language gloss into a **natural, grammatically correct English sentence**. 

Guidelines:
- Keep the original meaning and order of events. 
- Convert numbers and times into standard English (e.g., 9-AM → “nine in the morning”).
- Include subjects, objects, and verbs explicitly.
- Use proper articles (“a”, “an”, “the”) and prepositions as needed.
- Combine short phrases into natural, flowing sentences when appropriate.
- Maintain clarity and readability for a general audience.

Examples:

Gloss: ME EAT APPLE
English: I eat an apple.

Gloss: STORE I GO 9-AM MORNING
English: I went to the store at nine in the morning.

Gloss: YESTERDAY PARK WE WALK SIT TALK LAUGH SUNSET BEAUTIFUL
English: Yesterday, we walked to the park, sat, talked, laughed, and enjoyed the beautiful sunset.

Translate the following gloss:
"""

for idx, row in df.iterrows():
    gloss_num = idx + 1 
    gloss = row['gloss']
    translate = prompt + gloss
    
    response = client.responses.create(
        model="gpt-4o-mini",
        input=translate,
        store=True,
    )
    
    print(f"Gloss {gloss_num}: {gloss}")
    print(f"Translation {gloss_num}: {response.output_text}")
    print("-" * 50)

