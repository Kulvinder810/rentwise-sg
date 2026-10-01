# src/ranking.py

import pandas as pd

from models import RentalPreferences


def rank_listings(
    listings: pd.DataFrame,
    preferences: RentalPreferences
) -> pd.DataFrame:

    results = listings.copy()

    # --------------------------
    # HARD CONSTRAINTS
    # --------------------------

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

    # --------------------------
    # SOFT-PREFERENCE SCORE
    # --------------------------

    results = results.copy()

    results["score"] = 0.0

    # Near MRT preference
    if preferences.near_mrt_preferred is True:

        results["score"] += (
            (15 - results["mrt_walk_minutes"])
            .clip(lower=0)
            / 15
            * 40
        )

    # Shorter commute is better
    if preferences.max_commute_minutes is not None:

        results["score"] += (
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

        results.loc[
            results["room_type"].str.lower()
            == preferences.preferred_room_type.lower(),
            "score"
        ] += 15

    # Private bathroom
    if preferences.private_bathroom_preferred is True:

        results.loc[
            results["private_bathroom"] == True,
            "score"
        ] += 15

    return results.sort_values(
        "score",
        ascending=False
    )