import pandas as pd


def combine_scores(
    listings: pd.DataFrame,
    structured_weight: float = 0.7,
    semantic_weight: float = 0.3
) -> pd.DataFrame:

    results = listings.copy()

    min_distance = results["semantic_distance"].min()
    max_distance = results["semantic_distance"].max()

    # -----------------------------
    # Normalize semantic distance
    # -----------------------------

    if len(results) == 1:

        results["semantic_percentage"] = 100.0

    elif max_distance == min_distance:

        results["semantic_percentage"] = 50.0

    else:

        results["semantic_percentage"] = (
            (
                max_distance
                - results["semantic_distance"]
            )
            /
            (
                max_distance
                - min_distance
            )
            * 100
        )

    # -----------------------------
    # Hybrid score
    # -----------------------------

    results["final_score"] = (
        results["match_percentage"]
        * structured_weight
        +
        results["semantic_percentage"]
        * semantic_weight
    ).round(1)

    # VERY IMPORTANT
    return results.sort_values(
        "final_score",
        ascending=False
    )