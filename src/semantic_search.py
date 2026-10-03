import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent.parent

EMBEDDINGS_PATH = (
    BASE_DIR
    / "data"
    / "listing_embeddings.json"
)

load_dotenv()

client = OpenAI()

EMBEDDING_MODEL = "text-embedding-3-small"

def load_listing_embeddings() -> dict:

    with open(EMBEDDINGS_PATH, "r") as file:
        stored = json.load(file)

    return {
        item["listing_id"]: item["embedding"]
        for item in stored
    }



def get_embedding(text: str) -> list[float]:

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )

    return response.data[0].embedding

# cosine similarity
def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float]
) -> float:

    a = np.array(vector_a)
    b = np.array(vector_b)

    return np.dot(a, b) / (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

def semantic_search(
    listings: pd.DataFrame,
    query: str
) -> pd.DataFrame:

    results = listings.copy()

    # Only the user's query needs a new embedding
    query_embedding = get_embedding(query)

    stored_embeddings = load_listing_embeddings()

    similarities = []

    for _, listing in results.iterrows():

        listing_embedding = stored_embeddings[
            int(listing["id"])
        ]

        similarity = cosine_similarity(
            query_embedding,
            listing_embedding
        )

        similarities.append(similarity)

    results["semantic_score"] = similarities

    return results.sort_values(
        "semantic_score",
        ascending=False
    )