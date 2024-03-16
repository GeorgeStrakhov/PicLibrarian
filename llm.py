from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
import numpy as np
client = OpenAI()


# humanize a photo caption
def humanize(caption):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a system that helps turn automatically generated image captions into much more human descriptions. Imagine how a human would describe such a picture (if they were trying to prompt for it) and then reply with such a human-like text. Only reply with transformed text and nothing else."
            },
            {
                "role": "user",
                "content": caption
            }
        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response.choices[0].message.content


# generate embeddings from text
def generate_embedding(text):

    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )

    return response.data[0].embedding

# calculate similarity between two embeddings
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
