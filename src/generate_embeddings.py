import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI()

EMBEDDING_MODEL = "text-embedding-3-small"


BASE_DIR = Path(__file__).resolve().parent.parent

LISTINGS_PATH = BASE_DIR / "data" / "listings.csv"

EMBEDDINGS_PATH = (
    BASE_DIR
    / "data"
    / "listing_embeddings.json"
)


def generate_listing_embeddings():

    listings = pd.read_csv(LISTINGS_PATH)

    descriptions = listings["description"].tolist()

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=descriptions
    )

    stored_embeddings = []

    for listing_id, embedding_result in zip(
        listings["id"],
        response.data
    ):

        stored_embeddings.append(
            {
                "listing_id": int(listing_id),
                "embedding": embedding_result.embedding
            }
        )

    with open(
        EMBEDDINGS_PATH,
        "w"
    ) as file:

        json.dump(
            stored_embeddings,
            file
        )

    print(
        f"Stored embeddings for "
        f"{len(stored_embeddings)} listings."
    )


if __name__ == "__main__":
    generate_listing_embeddings()
