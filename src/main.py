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
from recommendation_explainer import explain_recommendations
from pathlib import Path
from preference_parser import parse_preferences
from ranking import rank_listings

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
query="""
Master room under S$500, private bathroom, 5-minute commute, no landlord, cooking allowed.
"""

# STEP 1:
# Let the LLM understand the user
preferences = parse_preferences(query)

# print("\nExtracted preferences:")
# print(preferences.model_dump_json(indent=2))


# STEP 2:
# Load structured rental data
listings = pd.read_csv(data_path)


# STEP 3:
# Use deterministic Python logic
# print("\nExtracted Preferences:")
# print(preferences.model_dump_json(indent=2))

ranked = rank_listings(
    listings,
    preferences
)

print("\nRanked rows:", len(ranked))
print(ranked)

if ranked.empty:
    print("\nNo properties matched all of your hard requirements.")
    print(
        "Try relaxing one or more constraints such as "
        "budget, commute time, cooking requirements, "
        "or landlord preference."
    )

else:
    print("\nRanked Listings:\n")

    print(
        ranked[
            [
                "title",
                "rent",
                "mrt_walk_minutes",
                "commute_minutes",
                "match_percentage"
            ]
        ].head(3)
    )

    recommendation = explain_recommendations(
        ranked,
        preferences
    )

    print("\nRentWise Recommendation:\n")
    print(recommendation)