from pathlib import Path

import pandas as pd

from embedding_service import get_embedding
from vector_store import collection


BASE_DIR = Path(__file__).resolve().parent.parent

LISTINGS_PATH = BASE_DIR / "data" / "listings.csv"


def index_listings():

    listings = pd.read_csv(LISTINGS_PATH)

    ids = []
    documents = []
    embeddings = []
    metadatas = []

    for _, listing in listings.iterrows():

        description = listing["description"]

        embedding = get_embedding(
            description
        )

        ids.append(
            str(int(listing["id"]))
        )

        documents.append(
            description
        )

        embeddings.append(
            embedding
        )

        metadatas.append(
            {
                "title": str(listing["title"]),
                "area": str(listing["area"]),
                "rent": int(listing["rent"]),
                "room_type": str(listing["room_type"]),
                "cooking_allowed": bool(
                    listing["cooking_allowed"]
                ),
                "live_in_landlord": bool(
                    listing["live_in_landlord"]
                )
            }
        )

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(
        f"Indexed {len(ids)} listings into Chroma."
    )


if __name__ == "__main__":
    index_listings()