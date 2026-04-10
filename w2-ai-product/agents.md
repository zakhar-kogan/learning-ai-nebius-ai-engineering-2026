# AI Agent Context — LLM Evaluation Assignment

This document provides context for any AI coding assistant (Claude, Cursor, Windsurf, etc.) working on this project.

## Project Goal
This is Assignment 1 for the Nebius AI Engineering course. The goal is to build an LLM evaluation pipeline demonstrating both manual evaluation and "LLM-as-a-judge" evaluation of e-commerce product descriptions.

## Tech Stack & Architecture
- **Language**: Python >= 3.11
- **Package Manager**: `uv` (global `pyproject.toml` in the repo root)
- **Notebook**: `marimo` (reactive notebook, stored as the pure Python file `assignment_colab.py`)
- **API Client**: `litellm` (handles routing for Nebius and NVIDIA NIM)
- **Structured Output**: `pydantic` (for Judge schemas)
- **Experiment Tracking**: `mlflow` (runtime artifact only; not required for the committed workflow)
- **Data Handling**: `pandas`, `openpyxl`

## Directory Structure
- `assignment_colab.py` — The canonical self-contained marimo notebook.
- `assignment_colab.ipynb` — Exported notebook with preserved outputs for submission/sharing.
- `data/` — Static dataset used by the notebook.
- `outputs/` — Current deliverables and cached experiment artifacts.
- `legacy/modular_workflow/` — Archived pre-consolidation notebooks/src/prompts workflow kept only for reference.

## Key Design Constraints & Rules
1. **Single Source of Truth**: `assignment_colab.py` is the canonical workflow. Do not split logic back across `src/`, `notebooks/`, or external prompt files unless explicitly requested.
2. **Judge Schema Ordering**: `explanation` must come before `verdict` in the judge schema and prompts to reduce anchoring bias.
3. **Packaged Artifact Reuse**: The notebook should reuse existing deliverables from `outputs/` and only run live calls when required artifacts are missing or `FORCE_RERUN=1`.
4. **Resiliency**: Judge calls rely on retries, bounded timeouts, and structured output validation.
5. **Exports**: Keep the marimo source notebook and the exported `.ipynb` with outputs aligned.

## How to run
1. Set `NEBIUS_API_KEY` (and optionally `NVIDIA_API_KEY`) in `.env` inside `w2-ai-product/`.
2. Open the canonical notebook: `marimo edit w2-ai-product/assignment_colab.py`
3. Export when needed: `marimo export ipynb w2-ai-product/assignment_colab.py -o w2-ai-product/assignment_colab.ipynb --include-outputs`
