from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class RentalPreferences(BaseModel):

    # IMPORTANT:
    # Don't allow the LLM to produce fields that we haven't defined.
    model_config = ConfigDict(extra="forbid")

    max_rent: Optional[int] = Field(
        description="Maximum monthly rent in SGD"
    )

    work_location: Optional[str] = Field(
        description="Location where the user works or regularly commutes to"
    )

    max_commute_minutes: Optional[int] = Field(
        description="Maximum acceptable one-way commute in minutes"
    )

    cooking_required: Optional[bool] = Field(
        description="Whether cooking being allowed is mandatory"
    )

    max_mrt_walk_minutes: Optional[int] = Field(
        description="Maximum preferred walking time to MRT"
    )

    no_live_in_landlord: Optional[bool] = Field(
        description="Whether the user wants to avoid living with the landlord"
    )

    preferred_room_type: Optional[str] = Field(
        description="Preferred room type such as common or master"
    )

    private_bathroom_preferred: Optional[bool] = Field(
        description="Whether a private bathroom is preferred"
    )

    near_mrt_preferred: Optional[bool] = Field(
    description="Whether being close to an MRT station is a preference"
    )