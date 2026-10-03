import json
import os

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI()


def generate_rag_recommendation(
    query: str,
    ranked_listings: pd.DataFrame,
    top_k: int = 3
) -> str:

    # -----------------------------
    # RETRIEVAL
    # -----------------------------

    retrieved = ranked_listings.head(top_k)

    context = []

    for _, listing in retrieved.iterrows():

        context.append(
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
                "description": listing["description"],
                "match_percentage": float(
                    listing["match_percentage"]
                ),
                "semantic_score": float(
                    listing["semantic_score"]
                ),
                "final_score": float(
                    listing["final_score"]
                ),
                "reasons": listing["reasons"],
                "tradeoffs": listing["tradeoffs"]
            }
        )

    # -----------------------------
    # AUGMENTATION
    # -----------------------------

    prompt = {
        "user_query": query,
        "retrieved_listings": context
    }

    # -----------------------------
    # GENERATION
    # -----------------------------

    response = client.responses.create(

        model=os.environ["OPENAI_MODEL"],

        instructions="""
            You are RentWise SG, an AI rental assistant.

            You are given:
            1. The user's rental request.
            2. A small set of listings retrieved and ranked by the RentWise
            recommendation engine.

            Your job is to explain the retrieved results.

            Rules:
            - Use ONLY information contained in retrieved_listings.
            - Do not invent amenities, locations, travel times, prices,
            property features, or landlord conditions.
            - Do not change the ranking order.
            - Clearly explain why the first listing is ranked highest.
            - Mention meaningful trade-offs.
            - If information is unavailable, say that it is unavailable.
            - Keep the recommendation concise and practical.
            """,

        input=json.dumps(
            prompt,
            indent=2
        )
    )

    return response.output_text