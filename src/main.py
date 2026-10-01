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


# STEP 1:
# Let the LLM understand the user
preferences = parse_preferences(query)

print("\nExtracted preferences:")
print(preferences.model_dump_json(indent=2))


# STEP 2:
# Load structured rental data
listings = pd.read_csv(data_path)


# STEP 3:
# Use deterministic Python logic
ranked = rank_listings(
    listings,
    preferences
)


print("\nBest matching rentals:\n")

print(
    ranked[
        [
            "title",
            "area",
            "rent",
            "mrt_walk_minutes",
            "commute_minutes",
            "score"
        ]
    ]
)

