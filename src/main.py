#steps to check the schema  can uncomment if want to check schema
# from models import RentalPreferences
# import json 

# print(
#     json.dumps(
#         RentalPreferences.model_json_schema(),
#         indent=2
#     )
# )

import pandas as pd
from rag_recommender import generate_rag_recommendation
from recommendation_explainer import explain_recommendations
from pathlib import Path
from preference_parser import parse_preferences
from ranking import rank_listings
from semantic_search import semantic_search
from hybrid_ranking import combine_scores

BASE_DIR = Path(__file__).resolve().parent.parent
data_path = BASE_DIR / "data" / "listings.csv"

query = """
I just moved to Singapore.

I work near Labrador Park and I'm looking for a room
for at most $1,700 per month.

I'd rather keep my commute below 35 minutes.

Cooking is important to me and I don't want to stay
with a live-in landlord.

Being close to MRT would also be nice.
"""
# query="""
# Master room under S$500, private bathroom, 5-minute commute, no landlord, cooking allowed.
# """
# 1. Understand user
preferences = parse_preferences(query)


# 2. Load listings
listings = pd.read_csv(
    BASE_DIR / "data" / "listings.csv"
)


# 3. Hard filtering + structured ranking
ranked = rank_listings(
    listings,
    preferences
)


if ranked.empty:

    print(
        "\nNo properties matched "
        "your hard requirements."
    )

else:

    # 4. Semantic search ONLY over valid listings
    semantic_results = semantic_search(
        ranked,
        query
    )


    # 5. Combine both ranking signals
    final_results = combine_scores(
        semantic_results
    )


    print("\nFinal RentWise Ranking:\n")

    print(
        final_results[
            [
                "title",
                "match_percentage",
                "semantic_score",
                "semantic_percentage",
                "final_score"
            ]
        ].head(5)
    )

    rag_answer = generate_rag_recommendation(
        query=query,
        ranked_listings=final_results,
        top_k=3
    )

    print("\nRentWise RAG Recommendation:\n")
    print(rag_answer)