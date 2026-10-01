# RentWise SG

An AI-powered Singapore rental discovery assistant that converts
natural-language housing requirements into structured preferences
and ranks suitable rental listings.

## Current Features

- Natural-language rental preference extraction using an LLM
- Structured outputs using Pydantic
- Hard-constraint filtering
- Soft-preference scoring
- Synthetic Singapore rental dataset

## Current Architecture

User Query
    ↓
LLM Preference Parser
    ↓
Structured RentalPreferences
    ↓
Python Filtering & Ranking
    ↓
Ranked Rental Listings

## Planned Features

- Semantic search and embeddings
- RAG over listing descriptions
- Real rental data ingestion
- Commute-time APIs
- LangChain / LangGraph
- Agentic tool calling
- FastAPI backend
- Evaluation and observability
- Docker deployment