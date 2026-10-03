import pandas as pd
from models import RentalPreferences

def generate_match_reasons(
    listing,
    preferences: RentalPreferences
):

    reasons = []
    tradeoffs = []

    # Budget
    if preferences.max_rent is not None:

        savings = preferences.max_rent - listing["rent"]

        if savings >= 0:
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

    # Preferred room type
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

    # Private bathroom
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


def rank_listings(
    listings: pd.DataFrame,
    preferences: RentalPreferences
) -> pd.DataFrame:

    results = listings.copy()

    # ==========================================
    # 1. HARD CONSTRAINTS
    # ==========================================

    if preferences.max_rent is not None:
        results = results[
            results["rent"] <= preferences.max_rent
        ]

    if preferences.max_commute_minutes is not None:
        results = results[
            results["commute_minutes"]
            <= preferences.max_commute_minutes
        ]

    if preferences.cooking_required is True:
        results = results[
            results["cooking_allowed"] == True
        ]

    if preferences.no_live_in_landlord is True:
        results = results[
            results["live_in_landlord"] == False
        ]

    # IMPORTANT:
    # If hard filtering removed everything,
    # stop immediately.
    if results.empty:
        return results

    # ==========================================
    # 2. SOFT-PREFERENCE SCORING
    # ==========================================

    results = results.copy()

    results["raw_score"] = 0.0
    possible_score = 0

    # MRT proximity
    if preferences.near_mrt_preferred is True:

        possible_score += 30

        results["raw_score"] += (
            (15 - results["mrt_walk_minutes"])
            .clip(lower=0)
            / 15
            * 30
        )

    # Shorter commute
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

    # Preferred room type
    if preferences.preferred_room_type is not None:

        possible_score += 20

        results.loc[
            results["room_type"].str.lower()
            == preferences.preferred_room_type.lower(),
            "raw_score"
        ] += 20

    # Private bathroom
    if preferences.private_bathroom_preferred is True:

        possible_score += 20

        results.loc[
            results["private_bathroom"] == True,
            "raw_score"
        ] += 20

    # ==========================================
    # 3. NORMALIZE SCORE
    # ==========================================

    if possible_score > 0:

        results["match_percentage"] = (
            results["raw_score"]
            / possible_score
            * 100
        ).round(1)

    else:
        results["match_percentage"] = 100.0

    # ==========================================
    # 4. GENERATE EXPLANATIONS
    # ==========================================

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