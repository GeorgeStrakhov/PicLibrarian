from dotenv import load_dotenv
load_dotenv()

from pprint import pprint

import argparse
import os
import sys
import ast
import pandas as pd
import numpy as np

from llm import generate_embedding, cosine_similarity

# Load the .env file
load_dotenv()

DATA_FILE_PATH = 'image_data.csv'

# Create the parser
parser = argparse.ArgumentParser(description="Process some inputs.")

# Add the '-q' argument
parser.add_argument('-q', type=str, help='A string parameter')

# Parse the arguments
args = parser.parse_args()

user_query = args.q

if not user_query:
    print("No query provided. Please use -q 'your query' to provide a query")
    sys.exit()

if len(user_query) < 3:
    print("Query too short.")
    sys.exit()

# check that image_data.csv exists and is not empty
if not os.path.exists(DATA_FILE_PATH):
    print("No image data available. Please first put your images into 'images' directory and run 'python3 piclibrarian.py' to generate image descriptions and embeddings.")
    sys.exit()

# load the image data into a dataframe
image_data = pd.read_csv(DATA_FILE_PATH)

# convert the string representation of the embedding back to a numpy array
image_data["Embedding"] = image_data.Embedding.apply(ast.literal_eval).apply(np.array)

# calculate the embeddings for the user query
user_query_embedding = generate_embedding(user_query)

# find the match
image_data['similarity'] = image_data.Embedding.apply(lambda x: cosine_similarity(x, user_query_embedding))
top_match = image_data.sort_values('similarity', ascending=False).iloc[0]

print(f"Top match: {top_match['Filename']} with similarity {top_match['similarity']}")
