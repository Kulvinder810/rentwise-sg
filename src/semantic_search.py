import pandas as pd

from embedding_service import get_embedding
from vector_store import collection


def semantic_search(
    listings: pd.DataFrame,
    query: str
) -> pd.DataFrame:

    results = listings.copy()

    # IDs surviving our hard filters
    allowed_ids = {
        str(int(listing_id))
        for listing_id in results["id"]
    }

    # Only embed the user's search query
    query_embedding = get_embedding(query)

    # Vector database performs similarity search
    vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=collection.count(),
        include=[
            "distances",
            "documents",
            "metadatas"
        ]
    )

    distance_by_id = {}

    returned_ids = vector_results["ids"][0]
    returned_distances = vector_results["distances"][0]

    for listing_id, distance in zip(
        returned_ids,
        returned_distances
    ):

        if listing_id in allowed_ids:

            distance_by_id[int(listing_id)] = float(
                distance
            )

    results["semantic_distance"] = (
        results["id"]
        .map(distance_by_id)
    )

    return results.sort_values(
        "semantic_distance",
        ascending=True
    )