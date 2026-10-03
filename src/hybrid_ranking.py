import pandas as pd


def combine_scores(
    listings: pd.DataFrame,
    structured_weight: float = 0.7,
    semantic_weight: float = 0.3
) -> pd.DataFrame:

    results = listings.copy()

    min_score = results["semantic_score"].min()
    max_score = results["semantic_score"].max()

    # --------------------------------
    # Normalize semantic score to 0-100
    # --------------------------------

    if len(results) == 1:

        results["semantic_percentage"] = 100.0

    elif max_score == min_score:

        results["semantic_percentage"] = 50.0

    else:

        results["semantic_percentage"] = (
            (
                results["semantic_score"]
                - min_score
            )
            /
            (
                max_score
                - min_score
            )
            * 100
        )

    # --------------------------------
    # Final hybrid score
    # --------------------------------

    results["final_score"] = (
        results["match_percentage"]
        * structured_weight
        +
        results["semantic_percentage"]
        * semantic_weight
    ).round(1)

    return results.sort_values(
        "final_score",
        ascending=False
    )