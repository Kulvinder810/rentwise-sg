# src/ranking.py

import pandas as pd

from models import RentalPreferences


def rank_listings(
    listings: pd.DataFrame,
    preferences: RentalPreferences
) -> pd.DataFrame:
    # --------------------------
    # SOFT PREFERENCE SCORE
    # --------------------------

    results = listings.copy()

    results["raw_score"] = 0.0

    possible_score = 0


    # MRT proximity — max 30 points
    if preferences.near_mrt_preferred is True:

        possible_score += 30

        results["raw_score"] += (
            (15 - results["mrt_walk_minutes"])
            .clip(lower=0)
            / 15
            * 30
        )


    # Shorter commute — max 30 points
    if preferences.max_commute_minutes is not None:

        possible_score += 30

        results["raw_score"] += (
            (
                preferences.max_commute_minutes
                - results["commute_minutes"]
            )
            .clip(lower=0)
            / preferences.max_commute_minutes
            * 30
        )


    # Preferred room type — max 20 points
    if preferences.preferred_room_type is not None:

        possible_score += 20

        results.loc[
            results["room_type"].str.lower()
            == preferences.preferred_room_type.lower(),
            "raw_score"
        ] += 20


    # Private bathroom — max 20 points
    if preferences.private_bathroom_preferred is True:

        possible_score += 20

        results.loc[
            results["private_bathroom"] == True,
            "raw_score"
        ] += 20


    # Convert to percentage
    if possible_score > 0:

        results["match_percentage"] = (
            results["raw_score"]
            / possible_score
            * 100
        ).round(1)

    else:
        # User gave only hard constraints.
        # Every surviving listing satisfies them.
        results["match_percentage"] = 100.0

    results["reasons"] = None
    results["tradeoffs"] = None


    for index, listing in results.iterrows():

        reasons, tradeoffs = generate_match_reasons(
            listing,
            preferences
        )

        results.at[index, "reasons"] = reasons
        results.at[index, "tradeoffs"] = tradeoffs


    return results.sort_values(
        "match_percentage",
        ascending=False
    )

def generate_match_reasons(
    listing,
    preferences: RentalPreferences
):

    reasons = []
    tradeoffs = []

    # Budget
    if preferences.max_rent is not None:

        savings = preferences.max_rent - listing["rent"]

        reasons.append(
            f"Rent is S${listing['rent']}, "
            f"S${savings} below your maximum budget."
        )

    # Commute
    if preferences.max_commute_minutes is not None:

        reasons.append(
            f"Commute is approximately "
            f"{listing['commute_minutes']} minutes."
        )

    # Cooking
    if preferences.cooking_required is True:

        reasons.append(
            "Cooking is allowed."
        )

    # Landlord
    if preferences.no_live_in_landlord is True:

        reasons.append(
            "No live-in landlord."
        )

    # MRT
    if preferences.near_mrt_preferred is True:

        walk = listing["mrt_walk_minutes"]

        if walk <= 5:
            reasons.append(
                f"Very close to MRT: about {walk} minutes walking."
            )

        elif walk <= 10:
            reasons.append(
                f"Reasonably close to MRT: about {walk} minutes walking."
            )

        else:
            tradeoffs.append(
                f"MRT is around {walk} minutes walking."
            )

    # Room type
    if preferences.preferred_room_type is not None:

        if (
            listing["room_type"].lower()
            == preferences.preferred_room_type.lower()
        ):
            reasons.append(
                f"Matches your preferred "
                f"{preferences.preferred_room_type} room type."
            )

        else:
            tradeoffs.append(
                f"This is a {listing['room_type']} room "
                f"instead of your preferred "
                f"{preferences.preferred_room_type} room."
            )

    # Bathroom
    if preferences.private_bathroom_preferred is True:

        if listing["private_bathroom"]:
            reasons.append(
                "Includes a private bathroom."
            )

        else:
            tradeoffs.append(
                "Does not include a private bathroom."
            )

    return reasons, tradeoffs