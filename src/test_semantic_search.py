from pathlib import Path

import pandas as pd

from semantic_search import semantic_search


BASE_DIR = Path(__file__).resolve().parent.parent

listings = pd.read_csv(
    BASE_DIR / "data" / "listings.csv"
)


query = """
I want a peaceful place with flexible cooking
and convenient public transport.
"""


results = semantic_search(
    listings,
    query
)


print(
    results[
        [
            "title",
            "description",
            "semantic_score"
        ]
    ].head(5)
)