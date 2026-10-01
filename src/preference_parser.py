# src/preference_parser.py

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from models import RentalPreferences


load_dotenv()

client = OpenAI()


def parse_preferences(user_query: str) -> RentalPreferences:

    schema = RentalPreferences.model_json_schema()

    response = client.responses.create(
        model=os.environ["OPENAI_MODEL"],

        instructions="""
        You extract Singapore rental-search preferences.

        Convert the user's natural-language request into the provided schema.

        Important rules:
        - Do not invent requirements.
        - If the user did not specify something, return null.
        - Monetary values are monthly SGD.
        - Distinguish mandatory requirements from preferences where possible.
        """,

        input=user_query,

        text={
            "format": {
                "type": "json_schema",
                "name": "rental_preferences",
                "schema": schema,
                "strict": True
            }
        }
    )

    return RentalPreferences.model_validate_json(
        response.output_text
    )