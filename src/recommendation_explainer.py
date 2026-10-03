import json
import os

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from models import RentalPreferences


load_dotenv()

client = OpenAI()


def explain_recommendations(
    ranked_listings: pd.DataFrame,
    preferences: RentalPreferences,
    top_n: int = 3
) -> str:

    top_listings = ranked_listings.head(top_n)

    listings_for_llm = []

    for _, listing in top_listings.iterrows():

        listings_for_llm.append(
            {
                "title": listing["title"],
                "area": listing["area"],
                "rent": int(listing["rent"]),
                "mrt": listing["mrt"],
                "mrt_walk_minutes": int(
                    listing["mrt_walk_minutes"]
                ),
                "commute_minutes": int(
                    listing["commute_minutes"]
                ),
                "room_type": listing["room_type"],
                "private_bathroom": bool(
                    listing["private_bathroom"]
                ),
                "match_percentage": float(
                    listing["match_percentage"]
                ),
                "reasons": listing["reasons"],
                "tradeoffs": listing["tradeoffs"]
            }
        )

    payload = {
        "preferences": preferences.model_dump(),
        "ranked_listings": listings_for_llm
    }

    response = client.responses.create(
        model=os.environ["OPENAI_MODEL"],

        instructions="""
You are RentWise SG, a Singapore rental recommendation assistant.

The ranking and calculations have already been performed by the
application.

Your job is ONLY to explain the results clearly.

Rules:

- Do not change the ranking order.
- Do not recalculate the scores.
- Do not invent property details.
- Only use facts provided in the input.
- Clearly mention important trade-offs.
- Be concise and practical.
- Explain why the highest-ranked property is ranked first.
- Use SGD for rental prices.
""",

        input=json.dumps(
            payload,
            indent=2
        )
    )

    return response.output_text