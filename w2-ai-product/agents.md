# AI Agent Context — LLM Evaluation Assignment

This document provides context for any AI coding assistant (Claude, Cursor, Windsurf, etc.) working on this project.

## Project Goal
This is Assignment 1 for the Nebius AI Engineering course. The goal is to build an LLM evaluation pipeline demonstrating both manual evaluation and "LLM-as-a-judge" evaluation of e-commerce product descriptions.

## Tech Stack & Architecture
- **Language**: Python >= 3.11
- **Package Manager**: `uv` (global `pyproject.toml` in the repo root)
- **Notebooks**: `marimo` (Reactive notebooks, stored as pure Python files in `notebooks/`)
- **API Client**: `litellm` (handles routing for Nebius and NVIDIA NIM)
- **Structured Output**: `pydantic` (for Judge schemas)
- **Experiment Tracking**: `mlflow` (using local SQLite `experiments.db` + `mlflow.litellm.autolog()`)
- **Data Handling**: `pandas`, `openpyxl`

## Directory Structure
- `src/` — Reusable logic imported by notebooks (`rubric.py`, `schemas.py`, `utils.py`). Keeps notebooks focused on orchestration and analysis.
- `notebooks/` — The actual deliverables (6 `marimo` notebooks, one for each task).
- `prompts/` — Prompt templates stored as `.txt` files.
- `data/` — Static datasets.

## Key Design Constraints & Rules
1. **Rubric is Code**: `src/rubric.py` is the single source of truth for evaluation criteria. The judge prompt and scoring logic both read from here.
2. **Judge Schema Ordering**: In `src/schemas.py`, `explanation` MUST come before `verdict`. This forces chain-of-thought reasoning and prevents anchoring bias. Do not change this order.
3. **MLflow Boundary**: We use `mlflow` for tracing (`autolog()`) and experiment tracking for Task 4. We DO NOT use `mlflow.genai.evaluate()` because building the judge manually is part of the assignment grading criteria.
4. **Resiliency**: The Judge model (Task 5/6) relies on `tenacity` retries and strict JSON schemas to handle parsing failures.
5. **No `jupyter`/`.ipynb`**: Use `marimo` UI (`marimo edit ...`) or standard text editors. All notebooks are plain `.py` scripts.

## How to run
1. Set `NEBIUS_API_KEY` (and optionally `NVIDIA_API_KEY`) in `.env` inside `w2-ai-product/`.
2. Run notebooks sequentially: `marimo edit notebooks/01_rubric.py`
3. Check traces: `mlflow ui --backend-store-uri sqlite:///experiments.db`
