# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.2",
# ]
# ///

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    title_output = mo.md(f"""
    # Assignment 2 - RAG

    This notebook contains the code, tables, and write-ups for the FinanceBench RAG assignment.

    Runtime policy:
    - expensive model, embedding, and evaluation calls are cached per row;
    - cache rows are the source of truth for computed results;
    - required table deliverables are regenerated from DataFrames;
    """)

    title_output
    return


@app.cell
def _():
    import ast
    import hashlib
    import importlib.util as importlib_util
    import json
    import os
    import subprocess
    import sys
    import time
    import urllib.request
    from collections.abc import Callable, Iterable
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from dataclasses import dataclass
    from datetime import UTC, datetime
    from pathlib import Path
    from threading import Lock
    from typing import Any


    return (
        Any,
        Callable,
        Iterable,
        Lock,
        Path,
        ThreadPoolExecutor,
        UTC,
        as_completed,
        ast,
        dataclass,
        datetime,
        hashlib,
        importlib_util,
        json,
        os,
        subprocess,
        sys,
        time,
        urllib,
    )


@app.cell
def _(importlib_util, mo, os, subprocess, sys):
    NOTEBOOK_REQUIRED_PACKAGES = [
        "pandas",
        "openpyxl",
        "python-dotenv",
        "litellm",
        "openai",
        "langchain>=1.2.15",
        "langchain-community",
        "langchain-huggingface",
        "langchain-text-splitters",
        "faiss-cpu",
        "sentence-transformers",
        "pypdf",
        "ragas",
        "datasets",
        "tqdm",
        "ipython",
    ]

    PACKAGE_MODULE_OVERRIDES = {
        "python-dotenv": "dotenv",
        "faiss-cpu": "faiss",
    }

    def package_module_name(package_spec: str) -> str:
        package_name = package_spec
        for separator in ["==", ">=", "<=", "~=", "!=", ">", "<"]:
            package_name = package_name.split(separator, maxsplit=1)[0]
        return PACKAGE_MODULE_OVERRIDES.get(
            package_name, package_name.replace("-", "_")
        )

    NOTEBOOK_MISSING_PACKAGES = [
        package_spec
        for package_spec in NOTEBOOK_REQUIRED_PACKAGES
        if importlib_util.find_spec(package_module_name(package_spec)) is None
    ]

    INSTALL_MISSING_PACKAGES = os.getenv("INSTALL_MISSING_PACKAGES", "1") == "1"

    if NOTEBOOK_MISSING_PACKAGES:
        NOTEBOOK_INSTALL_COMMAND = (
            "uv add " + " ".join(NOTEBOOK_MISSING_PACKAGES)
        )
        if INSTALL_MISSING_PACKAGES:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", *NOTEBOOK_MISSING_PACKAGES]
            )
            setup_check_output = mo.md(
                "## Setup Check\n\nInstalled missing packages with pip. Restart the notebook kernel before continuing."
            )
        else:
            setup_check_output = mo.md(
                f"""
                ## Setup Check

                Missing required packages:

                `{NOTEBOOK_INSTALL_COMMAND}`

                To install from inside the notebook instead, restart with `INSTALL_MISSING_PACKAGES=1`.
                """
            )
            mo.stop(True, setup_check_output)
    else:
        NOTEBOOK_INSTALL_COMMAND = ""
        setup_check_output = mo.md(
            "## Setup Check\n\nAll required notebook packages are importable."
        )

    setup_check_output
    return


@app.cell
def _():
    import pandas as pd
    from dotenv import load_dotenv
    from tqdm.auto import tqdm

    return load_dotenv, pd, tqdm


@app.cell
def _(Path, load_dotenv, os):
    if "__file__" in globals():
        NOTEBOOK_PATH = Path(__file__).resolve()
        NOTEBOOK_DIR = NOTEBOOK_PATH.parent
    else:
        NOTEBOOK_DIR = Path.cwd().resolve() / "w2-ai-product"
        NOTEBOOK_PATH = NOTEBOOK_DIR / "ai-product-2.py"

    REPO_ROOT = NOTEBOOK_DIR.parent
    ASSIGNMENT_SOURCE_DIR = REPO_ROOT / "ai-product-2"
    ASSIGNMENT2_DATA_DIR = NOTEBOOK_DIR / "data" / "assignment2"
    ASSIGNMENT2_OUTPUTS_DIR = NOTEBOOK_DIR / "outputs" / "assignment2"
    ASSIGNMENT2_CACHE_DIR = ASSIGNMENT2_OUTPUTS_DIR / "cache"
    ASSIGNMENT2_VECTORSTORE_DIR = ASSIGNMENT2_OUTPUTS_DIR / "vectorstores"
    FINANCEBENCH_PDF_DIR = ASSIGNMENT2_DATA_DIR / "financebench_pdfs"

    for notebook_directory in [
        ASSIGNMENT2_DATA_DIR,
        ASSIGNMENT2_OUTPUTS_DIR,
        ASSIGNMENT2_CACHE_DIR,
        ASSIGNMENT2_VECTORSTORE_DIR,
        FINANCEBENCH_PDF_DIR,
    ]:
        notebook_directory.mkdir(parents=True, exist_ok=True)

    load_dotenv(NOTEBOOK_DIR / ".env")

    # ── Runtime flags ────────────────────────────────────────────────────
    # Set to True to enable; loaded from .env when available.
    FORCE_RERUN = os.getenv("FORCE_RERUN", "0") == "1"
    RETRY_ERRORS = os.getenv("RETRY_ERRORS", "1") == "1"
    RUN_MODEL_CALLS = os.getenv("RUN_MODEL_CALLS", "1") == "1"
    RUN_EXPENSIVE_EVALS = os.getenv("RUN_EXPENSIVE_EVALS", "1") == "1"
    RUN_FAITHFULNESS = os.getenv("RUN_FAITHFULNESS", "1") == "1"
    ALLOW_DOWNLOADS = os.getenv("ALLOW_DOWNLOADS", "1") == "1"
    OUTPUT_FORMAT = os.getenv("OUTPUT_FORMAT", "both").lower()
    if OUTPUT_FORMAT not in {"xlsx", "csv", "both"}:
        raise ValueError("OUTPUT_FORMAT must be one of: xlsx, csv, both")

    # ── API keys & endpoints ─────────────────────────────────────────────
    # Replace the default strings below if running without a .env file.
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
    NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY", default="insert-your-nebius-api-key-here").strip()
    NEBIUS_BASE_URL = os.getenv("NEBIUS_BASE_URL", default="https://api.tokenfactory.nebius.com/v1/").strip()
    NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "").strip()
    NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", default="https://integrate.api.nvidia.com/v1").strip()

    # ── Model selection ──────────────────────────────────────────────────
    GENERATION_MODEL = os.getenv("ASSIGNMENT2_GENERATION_MODEL", default="meta-llama/Llama-3.3-70B-Instruct").strip()
    JUDGE_MODEL = os.getenv("ASSIGNMENT2_JUDGE_MODEL", default="deepseek-ai/DeepSeek-V3.2").strip()
    EMBEDDING_MODEL = os.getenv("ASSIGNMENT2_EMBEDDING_MODEL", default="BAAI/bge-small-en-v1.5").strip()
    RERANKER_MODEL = os.getenv("ASSIGNMENT2_RERANKER_MODEL", default="BAAI/bge-reranker-base").strip()
    QUERY_ROUTER_MODEL = os.getenv("ASSIGNMENT2_QUERY_ROUTER_MODEL", default="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning").strip()
    QUERY_ROUTER_MAX_TOKENS = int(os.getenv("ASSIGNMENT2_QUERY_ROUTER_MAX_TOKENS", "512"))

    # ── Evaluation parameters ────────────────────────────────────────────
    FAITHFULNESS_LIMIT = int(os.getenv("FAITHFULNESS_LIMIT", "20"))
    DEFAULT_RETRIEVAL_K = int(os.getenv("DEFAULT_RETRIEVAL_K", "4"))
    PAGE_HIT_K_VALUES = [1, 3, 5]

    FINANCEBENCH_LOCAL_JSON = ASSIGNMENT2_DATA_DIR / "financebench.json"
    FINANCEBENCH_LOCAL_CSV = ASSIGNMENT2_DATA_DIR / "financebench.csv"
    FINANCEBENCH_LOCAL_PARQUET = ASSIGNMENT2_DATA_DIR / "financebench.parquet"

    ASSIGNMENT2_NAIVE_BASE = ASSIGNMENT2_OUTPUTS_DIR / "assignment2_naive_generation"
    ASSIGNMENT2_COMPARE_BASE = ASSIGNMENT2_OUTPUTS_DIR / "assignment2_run_and_compare"
    ASSIGNMENT2_EVAL_BASE = ASSIGNMENT2_OUTPUTS_DIR / "assignment2_evaluation"
    ASSIGNMENT2_IMPROVEMENT_BASE = (
        ASSIGNMENT2_OUTPUTS_DIR / "assignment2_improvement_cycles"
    )

    TASK1_CACHE_PATH = ASSIGNMENT2_CACHE_DIR / "task1_naive_generation.jsonl"
    TASK4_CACHE_PATH = ASSIGNMENT2_CACHE_DIR / "task4_rag_answers.jsonl"
    TASK6_CORRECTNESS_CACHE_PATH = ASSIGNMENT2_CACHE_DIR / "task6_correctness.jsonl"
    TASK6_FAITHFULNESS_CACHE_PATH = ASSIGNMENT2_CACHE_DIR / "task6_faithfulness.jsonl"
    TASK7_EXPERIMENT_CACHE_PATH = ASSIGNMENT2_CACHE_DIR / "task7_experiments.jsonl"
    return (
        ALLOW_DOWNLOADS,
        ASSIGNMENT2_CACHE_DIR,
        ASSIGNMENT2_COMPARE_BASE,
        ASSIGNMENT2_DATA_DIR,
        ASSIGNMENT2_EVAL_BASE,
        ASSIGNMENT2_IMPROVEMENT_BASE,
        ASSIGNMENT2_NAIVE_BASE,
        ASSIGNMENT2_VECTORSTORE_DIR,
        DEFAULT_RETRIEVAL_K,
        EMBEDDING_MODEL,
        FAITHFULNESS_LIMIT,
        FINANCEBENCH_LOCAL_CSV,
        FINANCEBENCH_LOCAL_JSON,
        FINANCEBENCH_LOCAL_PARQUET,
        FINANCEBENCH_PDF_DIR,
        FORCE_RERUN,
        GEMINI_API_KEY,
        GENERATION_MODEL,
        JUDGE_MODEL,
        NEBIUS_API_KEY,
        NEBIUS_BASE_URL,
        NVIDIA_API_KEY,
        NVIDIA_BASE_URL,
        OUTPUT_FORMAT,
        PAGE_HIT_K_VALUES,
        QUERY_ROUTER_MODEL,
        RERANKER_MODEL,
        RETRY_ERRORS,
        RUN_EXPENSIVE_EVALS,
        RUN_FAITHFULNESS,
        RUN_MODEL_CALLS,
        TASK1_CACHE_PATH,
        TASK4_CACHE_PATH,
        TASK6_CORRECTNESS_CACHE_PATH,
        TASK6_FAITHFULNESS_CACHE_PATH,
        TASK7_EXPERIMENT_CACHE_PATH,
    )


@app.cell(hide_code=True)
def _(
    ALLOW_DOWNLOADS,
    DEFAULT_RETRIEVAL_K,
    EMBEDDING_MODEL,
    FAITHFULNESS_LIMIT,
    FORCE_RERUN,
    GENERATION_MODEL,
    JUDGE_MODEL,
    NEBIUS_API_KEY,
    NEBIUS_BASE_URL,
    OUTPUT_FORMAT,
    RETRY_ERRORS,
    RUN_EXPENSIVE_EVALS,
    RUN_FAITHFULNESS,
    RUN_MODEL_CALLS,
    mo,
):
    runtime_flags_output = mo.md(
        f"""
        ## Runtime Flags

        - `RUN_MODEL_CALLS`: `{RUN_MODEL_CALLS}`
        - `RUN_EXPENSIVE_EVALS`: `{RUN_EXPENSIVE_EVALS}`
        - `RUN_FAITHFULNESS`: `{RUN_FAITHFULNESS}`
        - `FORCE_RERUN`: `{FORCE_RERUN}`
        - `RETRY_ERRORS`: `{RETRY_ERRORS}`
        - `ALLOW_DOWNLOADS`: `{ALLOW_DOWNLOADS}`
        - `GENERATION_MODEL`: `{GENERATION_MODEL}`
        - `JUDGE_MODEL`: `{JUDGE_MODEL}`
        - `EMBEDDING_MODEL`: `{EMBEDDING_MODEL}`
        - `DEFAULT_RETRIEVAL_K`: `{DEFAULT_RETRIEVAL_K}`
        - `FAITHFULNESS_LIMIT`: `{FAITHFULNESS_LIMIT}`
        - `NEBIUS_BASE_URL`: `{NEBIUS_BASE_URL}`
        - `NEBIUS_API_KEY` present: `{bool(NEBIUS_API_KEY)}`
        - `OUTPUT_FORMAT`: `{OUTPUT_FORMAT}`
        """
    )
    runtime_flags_output
    return


@app.cell
def _(
    Any,
    Callable,
    Iterable,
    Lock,
    Path,
    ThreadPoolExecutor,
    UTC,
    as_completed,
    ast,
    dataclass,
    datetime,
    hashlib,
    json,
    pd,
    sys,
    time,
    tqdm,
    urllib,
):
    def utc_now_iso() -> str:
        return datetime.now(UTC).isoformat()


    def stable_json(value: Any) -> str:
        return json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)


    def stable_hash(value: Any) -> str:
        return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


    def text_hash(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()


    def package_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        return [dict(row) for row in rows]


    def round_metric_columns(
        df: pd.DataFrame,
        columns: Iterable[str],
        digits: int = 3,
    ) -> pd.DataFrame:
        rounded_df = df.copy()
        for _metric_column in columns:
            if _metric_column in rounded_df.columns:
                rounded_df[_metric_column] = pd.to_numeric(
                    rounded_df[_metric_column],
                    errors="coerce",
                ).round(digits)
        return rounded_df


    def read_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        rows = []
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                stripped = line.strip()
                if stripped:
                    try:
                        rows.append(json.loads(stripped))
                    except json.JSONDecodeError as exc:
                        print(
                            f"[cache] Skipping malformed JSONL row "
                            f"{path}:{line_number}: {exc}"
                        )
        return rows


    def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
        with tmp_path.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(stable_json(row) + "\n")
        tmp_path.replace(path)


    def load_cache_df(path: Path) -> pd.DataFrame:
        rows = read_jsonl(path)
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)


    def cache_lookup(
        cache_df: pd.DataFrame,
        *,
        task: str,
        financebench_id: str,
        input_hash: str,
        config_hash: str,
    ) -> dict[str, Any] | None:
        if cache_df.empty:
            return None
        required_columns = {
            "task",
            "financebench_id",
            "input_hash",
            "config_hash",
            "status",
        }
        if not required_columns.issubset(cache_df.columns):
            return None
        matches = cache_df[
            (cache_df["task"] == task)
            & (cache_df["financebench_id"] == financebench_id)
            & (cache_df["input_hash"] == input_hash)
            & (cache_df["config_hash"] == config_hash)
        ]
        if matches.empty:
            return None
        return matches.iloc[-1].to_dict()


    def upsert_cache_row(path: Path, row: dict[str, Any]) -> None:
        rows = read_jsonl(path)
        key_fields = ["task", "financebench_id", "input_hash", "config_hash"]
        filtered_rows = [
            existing
            for existing in rows
            if any(existing.get(field) != row.get(field) for field in key_fields)
        ]
        filtered_rows.append(row)
        write_jsonl(path, filtered_rows)


    def make_status_row(
        *,
        task: str,
        financebench_id: str,
        input_payload: dict[str, Any],
        config_payload: dict[str, Any],
        status: str,
        result_payload: dict[str, Any] | None = None,
        attempts: int = 0,
        error_message: str = "",
    ) -> dict[str, Any]:
        return {
            "task": task,
            "financebench_id": financebench_id,
            "status": status,
            "input_hash": stable_hash(input_payload),
            "config_hash": stable_hash(config_payload),
            "attempts": attempts,
            "error_message": error_message,
            "completed_at": utc_now_iso() if status == "done" else "",
            "created_at": utc_now_iso(),
            **(result_payload or {}),
        }


    def run_cached_rows(
        *,
        task: str,
        rows: list[dict[str, Any]],
        cache_path: Path,
        config_payload: dict[str, Any],
        input_payload_fn: Callable[[dict[str, Any]], dict[str, Any]],
        run_one_fn: Callable[[dict[str, Any]], dict[str, Any]],
        force_rerun: bool,
        retry_errors: bool,
        allow_run: bool,
        max_workers: int = 4,
    ) -> pd.DataFrame:
        cache_df = load_cache_df(cache_path)
        config_hash = stable_hash(config_payload)
        cache_lock = Lock()

        # First pass: separate cached hits from rows that need execution
        output_rows: dict[int, dict[str, Any]] = {}
        to_run: list[tuple[int, dict[str, Any], dict[str, Any], dict | None]] = []

        for idx, source_row in enumerate(rows):
            financebench_id = str(source_row["financebench_id"])
            input_payload = input_payload_fn(source_row)
            input_hash = stable_hash(input_payload)
            cached = cache_lookup(
                cache_df,
                task=task,
                financebench_id=financebench_id,
                input_hash=input_hash,
                config_hash=config_hash,
            )
            if (
                cached
                and cached.get("status") == "done"
                and not force_rerun
            ):
                output_rows[idx] = cached
                continue
            if (
                cached
                and cached.get("status") == "error"
                and not force_rerun
                and not retry_errors
            ):
                output_rows[idx] = cached
                continue
            if not allow_run:
                pending_row = make_status_row(
                    task=task,
                    financebench_id=financebench_id,
                    input_payload=input_payload,
                    config_payload=config_payload,
                    status="pending",
                    result_payload={"source_row": stable_json(source_row)},
                )
                output_rows[idx] = pending_row
                continue
            to_run.append((idx, source_row, input_payload, cached))

        # Second pass: run pending rows concurrently
        def _execute_one(item: tuple[int, dict, dict, dict | None]) -> tuple[int, dict[str, Any]]:
            idx, source_row, input_payload, cached = item
            financebench_id = str(source_row["financebench_id"])
            try:
                result_payload = run_one_fn(source_row)
                row = make_status_row(
                    task=task,
                    financebench_id=financebench_id,
                    input_payload=input_payload,
                    config_payload=config_payload,
                    status="done",
                    result_payload=result_payload,
                    attempts=int((cached or {}).get("attempts", 0)) + 1,
                )
            except Exception as exc:
                row = make_status_row(
                    task=task,
                    financebench_id=financebench_id,
                    input_payload=input_payload,
                    config_payload=config_payload,
                    status="error",
                    result_payload={"source_row": stable_json(source_row)},
                    attempts=int((cached or {}).get("attempts", 0)) + 1,
                    error_message=repr(exc),
                )
            with cache_lock:
                upsert_cache_row(cache_path, row)
            return idx, row

        if to_run:
            progress = tqdm(
                total=len(rows),
                desc=f"[{task}]",
                unit="row",
                file=sys.stdout,
                mininterval=0.5,
            )
            progress.update(len(output_rows))  # count cache hits
            progress.refresh()
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_execute_one, item): item for item in to_run}
                for future in as_completed(futures):
                    idx, row = future.result()
                    output_rows[idx] = row
                    progress.update(1)
            progress.close()
        else:
            # All from cache, still show progress
            for _ in tqdm(
                rows,
                desc=f"[{task}] (cached)",
                unit="row",
                file=sys.stdout,
                mininterval=0.5,
            ):
                pass

        return pd.DataFrame([output_rows[i] for i in sorted(output_rows)])


    def export_table(
        df: pd.DataFrame,
        base_path: Path,
        required_columns: list[str],
        output_format: str,
    ) -> list[Path]:
        export_df = df.copy()
        for column in required_columns:
            if column not in export_df.columns:
                export_df[column] = ""
        export_df = export_df[required_columns]
        base_path.parent.mkdir(parents=True, exist_ok=True)
        written_paths = []
        if output_format in {"xlsx", "both"}:
            xlsx_path = base_path.with_suffix(".xlsx")
            export_df.to_excel(xlsx_path, index=False)
            written_paths.append(xlsx_path)
        if output_format in {"csv", "both"}:
            csv_path = base_path.with_suffix(".csv")
            export_df.to_csv(csv_path, index=False)
            written_paths.append(csv_path)
        return written_paths


    def dataframe_to_markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
        if df.empty:
            return "_No rows available._"
        return df.head(max_rows).to_markdown(index=False)


    def display_table(df: pd.DataFrame, mo_module, max_rows: int = 20):
        if df.empty:
            return mo_module.md("_No rows available._")
        return mo_module.ui.table(df.head(max_rows), pagination=True)


    def parse_evidence_items(evidence: Any) -> list[dict[str, Any]]:
        if isinstance(evidence, list):
            return [
                item if isinstance(item, dict) else {"value": item}
                for item in evidence
            ]
        if isinstance(evidence, dict):
            return [evidence]
        if pd.isna(evidence):
            return []
        if isinstance(evidence, str):
            stripped = evidence.strip()
            if not stripped:
                return []
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(stripped)
                    return parse_evidence_items(parsed)
                except Exception:
                    continue
            return [{"evidence_text": stripped}]
        return []


    def evidence_page_numbers(evidence: Any) -> list[int]:
        page_numbers = []
        for item in parse_evidence_items(evidence):
            for key in ["evidence_page_num", "page_number", "page", "page_num"]:
                if key in item and item[key] not in (None, ""):
                    try:
                        page_numbers.append(int(item[key]))
                    except (TypeError, ValueError):
                        pass
        return sorted(set(page_numbers))


    def evidence_texts(evidence: Any) -> list[str]:
        texts = []
        for item in parse_evidence_items(evidence):
            text = item.get("evidence_text") or item.get("text") or item.get("value")
            if text:
                texts.append(str(text))
        return texts


    def source_fingerprint(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {"path": str(path), "exists": False}
        stat = path.stat()
        return {
            "path": str(path),
            "exists": True,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }


    def write_manifest(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(stable_json(payload), encoding="utf-8")


    def read_manifest(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))


    def manifest_matches(path: Path, payload: dict[str, Any]) -> bool:
        return read_manifest(path) == payload


    def download_file(url: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as response:
            destination.write_bytes(response.read())
        return destination


    def require_api_key(api_key: str) -> None:
        if not api_key:
            raise RuntimeError("NEBIUS_API_KEY is required for this call.")


    def call_chat_model(
        *,
        model: str,
        messages: list[dict[str, str]],
        api_key: str,
        base_url: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: float = 120.0,
    ) -> str:
        require_api_key(api_key)
        import litellm

        response = litellm.completion(
            model=f"openai/{model}",
            messages=messages,
            api_key=api_key,
            api_base=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        return response.choices[0].message.content or ""


    def retry_call(
        fn: Callable[[], Any],
        *,
        attempts: int = 3,
        sleep_seconds: float = 2.0,
    ) -> Any:
        last_exc = None
        for attempt_index in range(attempts):
            try:
                return fn()
            except Exception as exc:
                last_exc = exc
                if attempt_index + 1 < attempts:
                    time.sleep(sleep_seconds * (attempt_index + 1))
        raise last_exc


    @dataclass(frozen=True)
    class RagExperimentConfig:
        experiment: str
        change: str
        vectorstore_name: str
        chunk_size: int
        chunk_overlap: int
        generator_model: str
        generation_prompt_version: str
        k_for_generation: int
        use_reranker: bool = False
        rerank_fetch_k: int = 20
        hypothesis: str = ""

        def to_payload(self) -> dict[str, Any]:
            return {
                "experiment": self.experiment,
                "change": self.change,
                "vectorstore_name": self.vectorstore_name,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "generator_model": self.generator_model,
                "generation_prompt_version": self.generation_prompt_version,
                "k_for_generation": self.k_for_generation,
                "use_reranker": self.use_reranker,
                "rerank_fetch_k": self.rerank_fetch_k,
                "hypothesis": self.hypothesis,
            }


    return (
        RagExperimentConfig,
        call_chat_model,
        display_table,
        evidence_page_numbers,
        evidence_texts,
        export_table,
        manifest_matches,
        package_rows,
        retry_call,
        round_metric_columns,
        run_cached_rows,
        source_fingerprint,
        write_manifest,
    )


@app.cell(hide_code=True)
def _(mo):
    dataset_prep_output = mo.md(f"""
    ## Dataset Preparation

    FinanceBench is filtered to remove `metrics-generated` questions. The remaining rows are sorted by `financebench_id`.
    """)
    dataset_prep_output
    return


@app.cell
def _(
    ALLOW_DOWNLOADS,
    ASSIGNMENT2_DATA_DIR,
    FINANCEBENCH_LOCAL_CSV,
    FINANCEBENCH_LOCAL_JSON,
    FINANCEBENCH_LOCAL_PARQUET,
    FINANCEBENCH_PDF_DIR,
    pd,
    tqdm,
):
    def load_financebench_raw() -> pd.DataFrame:
        if FINANCEBENCH_LOCAL_JSON.exists():
            return pd.read_json(FINANCEBENCH_LOCAL_JSON)
        if FINANCEBENCH_LOCAL_CSV.exists():
            return pd.read_csv(FINANCEBENCH_LOCAL_CSV)
        if FINANCEBENCH_LOCAL_PARQUET.exists():
            return pd.read_parquet(FINANCEBENCH_LOCAL_PARQUET)
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "Install `datasets` or place financebench.csv/json/parquet under "
                f"{ASSIGNMENT2_DATA_DIR}."
            ) from exc
        dataset = load_dataset("PatronusAI/financebench", split="train")
        raw_df = dataset.to_pandas()
        FINANCEBENCH_LOCAL_JSON.parent.mkdir(parents=True, exist_ok=True)
        raw_df.to_json(FINANCEBENCH_LOCAL_JSON, orient="records", indent=2)
        return raw_df


    def load_filtered_financebench() -> pd.DataFrame:
        raw_df = load_financebench_raw()
        filtered_df = raw_df[raw_df["question_type"] != "metrics-generated"].copy()
        filtered_df = filtered_df.sort_values("financebench_id").reset_index(drop=True)
        return filtered_df


    def financebench_pdf_path(doc_name: str) -> str:
        return str(FINANCEBENCH_PDF_DIR / f"{doc_name}.pdf")


    def maybe_download_financebench_pdf(doc_name: str) -> str:
        destination = FINANCEBENCH_PDF_DIR / f"{doc_name}.pdf"
        if destination.exists():
            return str(destination)
        if not ALLOW_DOWNLOADS:
            return str(destination)
        from urllib.parse import quote
        import urllib.request

        encoded_name = quote(f"{doc_name}.pdf")
        url = (
            "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs/"
            + encoded_name
        )
        with urllib.request.urlopen(url, timeout=60) as response:
            destination.write_bytes(response.read())
        return str(destination)


    try:
        financebench_df = load_filtered_financebench()
        financebench_load_error = ""
    except Exception as financebench_exc:
        financebench_df = pd.DataFrame()
        financebench_load_error = repr(financebench_exc)

    if not financebench_df.empty:
        financebench_df["pdf_path"] = [
            maybe_download_financebench_pdf(doc_name)
            for doc_name in tqdm(
                financebench_df["doc_name"],
                desc="Resolving PDFs",
                unit="file",
            )
        ]

    financebench_summary_df = (
        financebench_df.groupby("question_type", dropna=False)
        .size()
        .reset_index(name="rows")
        if not financebench_df.empty
        else pd.DataFrame(columns=["question_type", "rows"])
    )
    return financebench_df, financebench_load_error, financebench_summary_df


@app.cell
def _(display_table, financebench_load_error, financebench_summary_df, mo):
    if financebench_load_error:
        dataset_status_output = mo.md(f"Dataset load status: `{financebench_load_error}`")
    else:
        dataset_status_output = display_table(financebench_summary_df, mo)
    dataset_status_output
    return


@app.cell(hide_code=True)
def _(financebench_df, mo):
    dataset_prep_qa_output = mo.md(f"""
    Q: How many rows remain and were document files already local or downloaded?

    A: {len(financebench_df)} rows; files were resolved locally or downloaded.
    """)
    dataset_prep_qa_output
    return


@app.cell(hide_code=True)
def _(mo):
    task1_prompt_output = mo.md(f"""
    ## Task 1 - Naive Generation

    Run the first 5 `domain-relevant` and first 5 `novel-generated` questions through the generation model without retrieval.
    """)
    task1_prompt_output
    return


@app.cell
def _(
    ASSIGNMENT2_NAIVE_BASE,
    FORCE_RERUN,
    GENERATION_MODEL,
    NEBIUS_API_KEY,
    NEBIUS_BASE_URL,
    OUTPUT_FORMAT,
    RETRY_ERRORS,
    RUN_MODEL_CALLS,
    TASK1_CACHE_PATH,
    call_chat_model,
    export_table,
    financebench_df,
    package_rows,
    pd,
    retry_call,
    run_cached_rows,
):
    # TASK1_SYSTEM_PROMPT = """You answer finance questions directly. If the question requires document-specific context that is not provided, say that the filing context is needed instead of guessing."""
    TASK1_SYSTEM_PROMPT = """"""

    def select_task1_questions(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        selected_parts = []
        for question_type in ["domain-relevant", "novel-generated"]:
            selected_parts.append(
                df[df["question_type"] == question_type]
                .sort_values("financebench_id")
                .head(5)
            )
        return pd.concat(selected_parts, ignore_index=True)


    task1_questions_df = select_task1_questions(financebench_df)


    def run_task1_one(row: dict) -> dict:
        answer = retry_call(
            lambda: call_chat_model(
                model=GENERATION_MODEL,
                api_key=NEBIUS_API_KEY,
                base_url=NEBIUS_BASE_URL,
                messages=[
                    {"role": "system", "content": TASK1_SYSTEM_PROMPT},
                    {"role": "user", "content": str(row["question"])},
                ],
                temperature=0.0,
                max_tokens=512,
            )
        )
        return {
            "question_type": row.get("question_type", ""),
            "question": row.get("question", ""),
            "naive_answer": answer,
            "ground_truth": row.get("answer", ""),
            "verdict": "",
            "model": GENERATION_MODEL,
        }


    task1_cache_df = run_cached_rows(
        task="task1_naive_generation",
        rows=package_rows(task1_questions_df.to_dict("records")),
        cache_path=TASK1_CACHE_PATH,
        config_payload={
            "model": GENERATION_MODEL,
            "prompt": TASK1_SYSTEM_PROMPT,
            "temperature": 0.0,
        },
        input_payload_fn=lambda row: {
            "financebench_id": row["financebench_id"],
            "question": row["question"],
        },
        run_one_fn=run_task1_one,
        force_rerun=FORCE_RERUN,
        retry_errors=RETRY_ERRORS,
        allow_run=RUN_MODEL_CALLS,
    )

    if task1_cache_df.empty:
        task1_naive_generation_df = pd.DataFrame(
            columns=[
                "financebench_id",
                "question_type",
                "question",
                "naive_answer",
                "ground_truth",
                "verdict",
            ]
        )
    else:
        task1_naive_generation_df = task1_cache_df.copy()
        if "ground_truth" not in task1_naive_generation_df.columns:
            task1_naive_generation_df = task1_naive_generation_df.merge(
                task1_questions_df[
                    ["financebench_id", "question_type", "question", "answer"]
                ].rename(columns={"answer": "ground_truth"}),
                on="financebench_id",
                how="left",
            )

    task1_export_paths = export_table(
        task1_naive_generation_df,
        ASSIGNMENT2_NAIVE_BASE,
        [
            "financebench_id",
            "question_type",
            "question",
            "naive_answer",
            "ground_truth",
            "verdict",
        ],
        OUTPUT_FORMAT,
    )
    return task1_export_paths, task1_naive_generation_df, task1_questions_df


@app.cell
def _(display_table, mo, task1_naive_generation_df):
    task1_table_output = display_table(task1_naive_generation_df, mo)
    task1_table_output
    return


@app.cell(hide_code=True)
def _(mo):
    task1_qa_output = mo.md(f"""
    Q: Where did the model refuse or ask for more information, and why?

    A: For some reason the model only refused to answer questions on global markets and data.
    Both requests did NOT name the company, so no 'start data' to confabulate.

    With another prompt like 'You answer finance questions directly. If the question requires document-specific context that is not provided, say that the filing context is needed instead of guessing.' it refused almost all the questions → which is great and expected with such a prompt.

    Q: Where did the model answer confidently, and how did it compare with the ground truth?

    A: All the questions on particular companies and their filing data - wonderfully hallucinated!

    Q: What patterns appear by `question_type`?

    A: Domain-relevant questions seem to have more of a 'To determine' + hallucinations. Novel-generated questions are more varied in answer patterns.
    """)
    task1_qa_output
    return


@app.cell(hide_code=True)
def _(mo):
    task2_output = mo.md(f"""
    ## Task 2 - RAG Reminder

    **Indexing**

    Q: Explain how documents are loaded, split, embedded, and saved. Include where indexing can fail and whether it is offline or per query.

    A: Documents are loaded from local PDF files using LangChain's `PyPDFLoader`, which extracts one `Document` object per page. Each page is enriched with metadata (`doc_name`, `company`, `doc_period`, `page_number`). Pages are then split into chunks using `RecursiveCharacterTextSplitter` with `chunk_size=1000` and `chunk_overlap=150`. Chunks are embedded using `BAAI/bge-small-en-v1.5` via `HuggingFaceEmbeddings` and stored in a FAISS vector index saved to disk alongside a JSON manifest that records document fingerprints and embedding config. Indexing is **offline** — it runs once and is reused across queries. It can fail if a PDF file is missing (`FileNotFoundError`), if the embedding model fails to load (network or disk issue), or if FAISS serialization fails (disk space).

    **Retrieval**

    Q: Explain how the user query is embedded and matched to stored chunks. Include concrete retrieval failure examples and whether it happens per query.

    A: The user query is embedded with the same `BAAI/bge-small-en-v1.5` model used at index time. FAISS performs approximate nearest-neighbor search over the stored vectors and returns the top-k chunks (default k=4). This happens **per query**. Retrieval can fail if the vectorstore was not built or loaded (returns empty results), if the query is semantically distant from all indexed content (returns irrelevant chunks), or if the relevant information spans multiple pages and the chunk boundaries split it (partial or misleading context). A concrete example: a question about a specific financial metric may retrieve chunks from the right company but wrong fiscal year if the embedding similarity is close.

    **Generation**

    Q: Explain how retrieved chunks are put into the prompt. Include where generation can fail and whether it happens per query.

    A: Retrieved chunks are formatted into a numbered list (e.g., `[Chunk 1]`) with their `doc_name`, `page_number`, and full text content, separated by `---` dividers. This formatted context is appended to the user's question in the user message, while a system prompt instructs the model to answer only from the provided context and cite sources. The full prompt is sent to `meta-llama/Llama-3.3-70B-Instruct` via the Nebius API through LiteLLM. This happens **per query**. Generation can fail if the API key is missing or invalid (`RuntimeError`), if the API times out (120s default), if the model is overloaded (rate limiting / 5xx errors — mitigated by `retry_call` with 3 attempts), or if the retrieved context is irrelevant causing the model to hallucinate or refuse.
    """)
    task2_output
    return


@app.cell(hide_code=True)
def _(mo):
    task3_output = mo.md(f"""
    ## Task 3
    ### Embed Documents

    Baseline index:
    - loader: `PyPDFLoader`, one document per page;
    - metadata: `doc_name`, `company`, `doc_period`, `page_number`;
    - splitter: `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)`;
    - embeddings: `BAAI/bge-small-en-v1.5`;
    - vector store: LangChain FAISS, saved to disk with a manifest.
    """)
    task3_output
    return


@app.cell
def _(
    ASSIGNMENT2_VECTORSTORE_DIR,
    EMBEDDING_MODEL,
    FINANCEBENCH_PDF_DIR,
    GEMINI_API_KEY,
    Path,
    financebench_df,
    manifest_matches,
    pd,
    source_fingerprint,
    tqdm,
    write_manifest,
):
    BASELINE_CHUNK_SIZE = 1000
    BASELINE_CHUNK_OVERLAP = 150
    BASELINE_VECTORSTORE_NAME = "faiss_chunk1000"
    BASELINE_VECTORSTORE_PATH = ASSIGNMENT2_VECTORSTORE_DIR / BASELINE_VECTORSTORE_NAME
    BASELINE_MANIFEST_PATH = BASELINE_VECTORSTORE_PATH / "manifest.json"


    def referenced_documents(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(
                columns=["doc_name", "company", "doc_period", "pdf_path"]
            )
        return (
            df[["doc_name", "company", "doc_period", "pdf_path"]]
            .drop_duplicates()
            .sort_values("doc_name")
            .reset_index(drop=True)
        )


    referenced_documents_df = referenced_documents(financebench_df)


    def vectorstore_manifest(
        *,
        documents_df: pd.DataFrame,
        embedding_model: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> dict:
        document_fingerprints = []
        for row in documents_df.to_dict("records"):
            path = row.get("pdf_path") or str(FINANCEBENCH_PDF_DIR / f"{row['doc_name']}.pdf")
            document_fingerprints.append(
                {
                    "doc_name": row["doc_name"],
                    "company": row.get("company", ""),
                    "doc_period": row.get("doc_period", ""),
                    "fingerprint": source_fingerprint(Path(path)),
                }
            )
        return {
            "embedding_model": embedding_model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "documents": document_fingerprints,
        }


    baseline_vectorstore_manifest = vectorstore_manifest(
        documents_df=referenced_documents_df,
        embedding_model=EMBEDDING_MODEL,
        chunk_size=BASELINE_CHUNK_SIZE,
        chunk_overlap=BASELINE_CHUNK_OVERLAP,
    )


    def load_pdf_pages_with_metadata(documents_df: pd.DataFrame) -> list:
        from langchain_community.document_loaders import PyPDFLoader

        all_pages = []
        for row in tqdm(documents_df.to_dict("records"), desc="Loading PDFs", unit="doc"):
            pdf_path = Path(row["pdf_path"])
            if not pdf_path.exists():
                raise FileNotFoundError(
                    f"Missing PDF for {row['doc_name']}: {pdf_path}"
                )
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            for page in pages:
                default_page = page.metadata.get("page", 0)
                page.metadata.update(
                    {
                        "doc_name": row["doc_name"],
                        "company": row.get("company", ""),
                        "doc_period": row.get("doc_period", ""),
                        "page_number": int(default_page),
                    }
                )
                all_pages.append(page)
        return all_pages


    def build_or_load_vectorstore(
        *,
        documents_df: pd.DataFrame,
        vectorstore_path: Path,
        manifest_path: Path,
        manifest_payload: dict,
        embedding_model: str,
        chunk_size: int,
        chunk_overlap: int,
        force_rebuild: bool = False,
    ):
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        class LiteLLMEmbeddings:
            def __init__(
                self,
                model: str,
                api_key: str,
                min_request_interval_seconds: float = 0.75,
                max_retries: int = 8,
            ):
                from threading import Lock

                self.model = model
                self.api_key = api_key
                self.min_request_interval_seconds = min_request_interval_seconds
                self.max_retries = max_retries
                self._lock = Lock()
                self._last_request_at = 0.0

            def _wait_for_rate_limit(self) -> None:
                import time

                with self._lock:
                    elapsed = time.monotonic() - self._last_request_at
                    sleep_seconds = self.min_request_interval_seconds - elapsed
                    if sleep_seconds > 0:
                        time.sleep(sleep_seconds)
                    self._last_request_at = time.monotonic()

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                if not self.api_key:
                    raise RuntimeError("GEMINI_API_KEY is required for Gemini embeddings.")
                import time
                import litellm

                last_exc = None
                for attempt_index in range(self.max_retries):
                    try:
                        self._wait_for_rate_limit()
                        response = litellm.embedding(
                            model=self.model,
                            input=texts,
                            api_key=self.api_key,
                        )
                        break
                    except Exception as exc:
                        last_exc = exc
                        is_rate_limit_error = (
                            "RateLimit" in exc.__class__.__name__
                            or "RESOURCE_EXHAUSTED" in repr(exc)
                            or "Quota exceeded" in repr(exc)
                        )
                        if not is_rate_limit_error:
                            raise
                        sleep_seconds = max(1.5, 2**attempt_index)
                        print(
                            "[vectorstore] Gemini embedding rate-limited; "
                            f"sleeping {sleep_seconds:.1f}s before retry "
                            f"{attempt_index + 1}/{self.max_retries}"
                        )
                        time.sleep(sleep_seconds)
                else:
                    raise last_exc
                embeddings = []
                for item in response.data:
                    embedding = item["embedding"] if isinstance(item, dict) else item.embedding
                    embeddings.append(list(embedding))
                return embeddings

            def embed_query(self, text: str) -> list[float]:
                return self.embed_documents([text])[0]

            def __call__(self, text: str) -> list[float]:
                return self.embed_query(text)

        def make_embeddings(model_name: str):
            if model_name.startswith("gemini/"):
                return LiteLLMEmbeddings(model=model_name, api_key=GEMINI_API_KEY)
            return HuggingFaceEmbeddings(model_name=model_name)

        print(f"[vectorstore] Initializing embedding provider: {embedding_model}")
        embeddings = make_embeddings(embedding_model)
        index_file = vectorstore_path / "index.faiss"
        if (
            index_file.exists()
            and manifest_matches(manifest_path, manifest_payload)
            and not force_rebuild
        ):
            print(f"[vectorstore] Loading existing index from {vectorstore_path}")
            return FAISS.load_local(
                str(vectorstore_path),
                embeddings,
                allow_dangerous_deserialization=True,
            )
        print(f"[vectorstore] Building new index ({len(documents_df)} documents)")
        pages = load_pdf_pages_with_metadata(documents_df)
        print(f"[vectorstore] Loaded {len(pages)} pages, splitting into chunks (size={chunk_size}, overlap={chunk_overlap})")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = splitter.split_documents(pages)
        print(f"[vectorstore] Embedding {len(chunks)} chunks into FAISS index")
        # Embed in batches with progress tracking
        batch_size = 100 if embedding_model.startswith("gemini/") else 256
        all_texts = [chunk.page_content for chunk in chunks]
        all_metadatas = [chunk.metadata for chunk in chunks]
        all_embeddings = []
        print(f"[vectorstore] Embedding batch size: {batch_size}")
        if embedding_model.startswith("gemini/"):
            print("[vectorstore] Gemini embeddings are throttled to avoid free-tier request limits.")
        for i in tqdm(range(0, len(all_texts), batch_size), desc="Embedding chunks", unit="batch"):
            batch = all_texts[i : i + batch_size]
            all_embeddings.extend(embeddings.embed_documents(batch))
        text_embedding_pairs = list(zip(all_texts, all_embeddings))
        vectorstore = FAISS.from_embeddings(text_embedding_pairs, embeddings, metadatas=all_metadatas)
        vectorstore_path.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(vectorstore_path))
        write_manifest(manifest_path, manifest_payload)
        print(f"[vectorstore] Index saved to {vectorstore_path}")
        return vectorstore


    return (
        BASELINE_CHUNK_OVERLAP,
        BASELINE_CHUNK_SIZE,
        BASELINE_MANIFEST_PATH,
        BASELINE_VECTORSTORE_NAME,
        BASELINE_VECTORSTORE_PATH,
        baseline_vectorstore_manifest,
        build_or_load_vectorstore,
        referenced_documents_df,
    )


@app.cell
def _(
    BASELINE_MANIFEST_PATH,
    BASELINE_VECTORSTORE_PATH,
    Path,
    display_table,
    mo,
    referenced_documents_df,
):
    baseline_index_status_df = referenced_documents_df.copy()
    if not baseline_index_status_df.empty:
        baseline_index_status_df["pdf_exists"] = baseline_index_status_df["pdf_path"].map(
            lambda path: bool(path) and Path(path).exists()
        )
    baseline_index_output = mo.vstack([
        mo.md(
        f"""
        Baseline vectorstore path: `{BASELINE_VECTORSTORE_PATH}`

        Baseline manifest path: `{BASELINE_MANIFEST_PATH}`
        """
        ),
        display_table(referenced_documents_df, mo),
    ])
    baseline_index_output
    return


@app.cell
def _(
    BASELINE_MANIFEST_PATH,
    BASELINE_VECTORSTORE_PATH,
    FORCE_RERUN,
    baseline_vectorstore_manifest,
    build_or_load_vectorstore,
    referenced_documents_df,
):
    if not referenced_documents_df.empty:
        try:
            baseline_vectorstore = build_or_load_vectorstore(
                documents_df=referenced_documents_df,
                vectorstore_path=BASELINE_VECTORSTORE_PATH,
                manifest_path=BASELINE_MANIFEST_PATH,
                manifest_payload=baseline_vectorstore_manifest,
                embedding_model=baseline_vectorstore_manifest["embedding_model"],
                chunk_size=baseline_vectorstore_manifest["chunk_size"],
                chunk_overlap=baseline_vectorstore_manifest["chunk_overlap"],
                force_rebuild=FORCE_RERUN,
            )
            baseline_vectorstore_error = ""
        except Exception as vectorstore_exc:
            baseline_vectorstore = None
            baseline_vectorstore_error = repr(vectorstore_exc)
    else:
        baseline_vectorstore = None
        baseline_vectorstore_error = "No referenced documents loaded yet."
    return baseline_vectorstore, baseline_vectorstore_error


@app.cell
def _(baseline_vectorstore_error, mo):
    if baseline_vectorstore_error:
        vectorstore_status_output = mo.md(f"Vectorstore status: `{baseline_vectorstore_error}`")
    else:
        vectorstore_status_output = mo.md("Vectorstore status: baseline FAISS index loaded.")
    vectorstore_status_output
    return


@app.cell(hide_code=True)
def _(mo):
    retrieval_check_desc_output = mo.md(f"""
    ### Retrieval Sanity Check

    Pick 3 questions from the dataset and retrieve the top-k chunks for each. For each retrieval, check:
    - Did the retrieved chunks come from the right document (matching `doc_name`)?
    - Do they contain (or come close to) the evidence text from the dataset's `evidence` field?
    - Do they come from the right page (chunk `page_number` vs `evidence_page_num`)?
    """)
    retrieval_check_desc_output
    return


@app.cell
def _(
    baseline_vectorstore,
    evidence_page_numbers,
    evidence_texts,
    financebench_df,
    pd,
    retrieve_chunks,
):
    def retrieval_sanity_check(
        df: pd.DataFrame,
        *,
        vectorstore,
        sample_size: int = 3,
        k: int = 4,
    ) -> pd.DataFrame:
        if df.empty or vectorstore is None:
            return pd.DataFrame()
        rows = []
        sample_df = df.sort_values("financebench_id").head(sample_size)
        for source_row in sample_df.to_dict("records"):
            chunks = retrieve_chunks(vectorstore, source_row["question"], k=k)
            expected_pages = evidence_page_numbers(source_row.get("evidence"))
            evidence_snippets = evidence_texts(source_row.get("evidence"))
            retrieved_pages = [
                chunk.get("page_number")
                for chunk in chunks
                if chunk.get("doc_name") == source_row.get("doc_name")
            ]
            top_chunk_content = chunks[0].get("content", "")[:300] if chunks else ""
            rows.append(
                {
                    "financebench_id": source_row["financebench_id"],
                    "doc_name": source_row["doc_name"],
                    "expected_pages": expected_pages,
                    "retrieved_doc_names": [
                        chunk.get("doc_name", "") for chunk in chunks
                    ],
                    "retrieved_pages_for_doc": retrieved_pages,
                    "page_hit": bool(set(expected_pages) & set(retrieved_pages)),
                    "doc_name_hit": source_row.get("doc_name") in [
                        chunk.get("doc_name", "") for chunk in chunks
                    ],
                    "first_evidence_snippet": (
                        evidence_snippets[0][:300] if evidence_snippets else ""
                    ),
                    "top_retrieved_chunk": top_chunk_content,
                }
            )
        return pd.DataFrame(rows)


    retrieval_sanity_df = retrieval_sanity_check(
        financebench_df, vectorstore=baseline_vectorstore, sample_size=3, k=4
    )
    return (retrieval_sanity_df,)


@app.cell
def _(display_table, mo, retrieval_sanity_df):
    retrieval_sanity_output = display_table(retrieval_sanity_df, mo)
    retrieval_sanity_output
    return


@app.cell(hide_code=True)
def _(mo):
    task3_qa_output = mo.md(f"""
    ## Write-up - Retrieval Sanity Check
    Q: After running retrieval sanity checks, did chunks come from the right documents, right pages, and near the evidence text?

    C: CORNING_2022_10K
    - Did the retrieved chunks come from the right document (matching `doc_name`)?
    Doc names - yeah, with an additional 'false positive' of 3M documents.
    - Do they contain (or come close to) the evidence text from the dataset's `evidence` field?
    Not too close - it's just the same company.
    - Do they come from the right page (chunk `page_number` vs `evidence_page_num`)?
    Nope.
    C: AMERICANWATERWORKS_2022_10K
    - Did the retrieved chunks come from the right document (matching `doc_name`)?
    Yes. Again, with a sprinkle of 3M!
    - Do they contain (or come close to) the evidence text from the dataset's `evidence` field?
    Yeah, NOW it's at least close - balance sheet/assets.
    - Do they come from the right page (chunk `page_number` vs `evidence_page_num`)?
    One of the pages is, at least, correct!
    C: PAYPAL_2022_10K
    - Did the retrieved chunks come from the right document (matching `doc_name`)?
    Yeah, all correct.
    - Do they contain (or come close to) the evidence text from the dataset's `evidence` field?
    Totally not correct.
    - Do they come from the right page (chunk `page_number` vs `evidence_page_num`)?
    No.
    """)
    task3_qa_output
    return


@app.cell(hide_code=True)
def _(mo):
    task4_output = mo.md(f"""
    ## Task 4 - RAG Pipeline

    Required public function:

    ```python
    answer_with_rag(
        query: str,
        k: int = 4,
        include_programmatic_sources: bool = False,
    ) -> dict
    ```

    The return dictionary contains:
    - `answer`: final model answer;
    - `retrieved_chunks`: chunk records with `doc_name` and `page_number` metadata.
    """)
    task4_output
    return


@app.cell
def _(
    DEFAULT_RETRIEVAL_K,
    GENERATION_MODEL,
    NEBIUS_API_KEY,
    NEBIUS_BASE_URL,
    baseline_vectorstore,
    call_chat_model,
    retry_call,
):
    RAG_SYSTEM_PROMPT_V1 = """You answer finance questions using only the provided context.

    Rules:
    1. Answer only from the context.
    2. If the context does not contain the answer, say: "The provided context does not contain the answer."
    3. Keep the answer concise.
    4. Cite the document name and page number for each factual claim. Use citations like [doc_name p.page_number].
    """


    def retrieved_doc_to_record(doc) -> dict:
        return {
            "doc_name": doc.metadata.get("doc_name", ""),
            "page_number": doc.metadata.get("page_number", ""),
            "company": doc.metadata.get("company", ""),
            "doc_period": doc.metadata.get("doc_period", ""),
            "content": doc.page_content,
        }


    def retrieve_chunks(vectorstore, query: str, k: int = DEFAULT_RETRIEVAL_K) -> list[dict]:
        if vectorstore is None:
            return []
        docs = vectorstore.similarity_search(query, k=k)
        return [retrieved_doc_to_record(doc) for doc in docs]


    def format_retrieved_context(chunks: list[dict]) -> str:
        if not chunks:
            return "No relevant context was retrieved."
        formatted_chunks = []
        for index, chunk in enumerate(chunks, start=1):
            formatted_chunks.append(
                "\n".join(
                    [
                        f"[Chunk {index}]",
                        f"doc_name: {chunk.get('doc_name', '')}",
                        f"page_number: {chunk.get('page_number', '')}",
                        "content:",
                        str(chunk.get("content", "")),
                    ]
                )
            )
        return "\n\n---\n\n".join(formatted_chunks)


    def normalize_page_number(page_number) -> int | str:
        if page_number is None:
            return ""
        try:
            return int(page_number)
        except (TypeError, ValueError):
            return str(page_number).strip()


    def page_sort_key(page_number: int | str) -> tuple[int, int | str]:
        if isinstance(page_number, int):
            return (0, page_number)
        return (1, page_number)


    def programmatic_sources_from_chunks(chunks: list[dict]) -> list[dict]:
        source_pages_by_doc: dict[str, set[int | str]] = {}
        doc_order = []
        for chunk in chunks:
            doc_name = str(chunk.get("doc_name", "")).strip()
            if not doc_name:
                continue
            if doc_name not in source_pages_by_doc:
                source_pages_by_doc[doc_name] = set()
                doc_order.append(doc_name)
            page_number = normalize_page_number(chunk.get("page_number", ""))
            if page_number != "":
                source_pages_by_doc[doc_name].add(page_number)
        return [
            {
                "doc_name": doc_name,
                "page_numbers": sorted(
                    source_pages_by_doc[doc_name],
                    key=page_sort_key,
                ),
            }
            for doc_name in doc_order
        ]


    def format_programmatic_sources(sources: list[dict]) -> str:
        if not sources:
            return "Sources retrieved:\n- None"
        lines = ["Sources retrieved:"]
        for source in sources:
            lines.append(f"- {source['doc_name']}, {source['page_numbers']}")
        return "\n".join(lines)


    def answer_from_chunks(
        *,
        query: str,
        chunks: list[dict],
        model: str = GENERATION_MODEL,
        system_prompt: str = RAG_SYSTEM_PROMPT_V1,
        include_programmatic_sources: bool = False,
    ) -> dict:
        context = format_retrieved_context(chunks)
        user_prompt = f"""Question:
    {query}

    Retrieved context:
    {context}
    """
        answer = retry_call(
            lambda: call_chat_model(
                model=model,
                api_key=NEBIUS_API_KEY,
                base_url=NEBIUS_BASE_URL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=700,
            )
        )
        programmatic_sources = programmatic_sources_from_chunks(chunks)
        if include_programmatic_sources:
            answer = (
                f"{answer.rstrip()}\n\n"
                f"{format_programmatic_sources(programmatic_sources)}"
            )
        return {
            "answer": answer,
            "retrieved_chunks": chunks,
        }


    def answer_with_rag_from_vectorstore(
        *,
        vectorstore,
        query: str,
        k: int = DEFAULT_RETRIEVAL_K,
        model: str = GENERATION_MODEL,
        system_prompt: str = RAG_SYSTEM_PROMPT_V1,
        include_programmatic_sources: bool = False,
    ) -> dict:
        chunks = retrieve_chunks(vectorstore, query, k=k)
        return answer_from_chunks(
            query=query,
            chunks=chunks,
            model=model,
            system_prompt=system_prompt,
            include_programmatic_sources=include_programmatic_sources,
        )


    def answer_with_rag(
        query: str,
        k: int = DEFAULT_RETRIEVAL_K,
        include_programmatic_sources: bool = False,
    ) -> dict:
        return answer_with_rag_from_vectorstore(
            vectorstore=baseline_vectorstore,
            query=query,
            k=k,
            model=GENERATION_MODEL,
            system_prompt=RAG_SYSTEM_PROMPT_V1,
            include_programmatic_sources=include_programmatic_sources,
        )


    return (
        RAG_SYSTEM_PROMPT_V1,
        answer_from_chunks,
        answer_with_rag,
        retrieve_chunks,
    )


@app.cell(hide_code=True)
def _(mo):
    task5_output = mo.md(f"""
    ## Task 5 - Run And Compare
    """)
    task5_output
    return


@app.cell
def _(
    ASSIGNMENT2_COMPARE_BASE,
    FORCE_RERUN,
    GENERATION_MODEL,
    OUTPUT_FORMAT,
    RAG_SYSTEM_PROMPT_V1,
    RETRY_ERRORS,
    RUN_MODEL_CALLS,
    TASK4_CACHE_PATH,
    answer_with_rag,
    export_table,
    json,
    package_rows,
    pd,
    run_cached_rows,
    task1_naive_generation_df,
    task1_questions_df,
):
    def run_task5_rag_one(row: dict) -> dict:
        rag_result = answer_with_rag(
            str(row["question"]),
            include_programmatic_sources=True,
        )
        return {
            "question_type": row.get("question_type", ""),
            "question": row.get("question", ""),
            "RAG_answer": rag_result["answer"],
            "retrieved_chunks_json": json.dumps(
                rag_result["retrieved_chunks"], ensure_ascii=True
            ),
            "ground_truth": row.get("answer", ""),
            "model": GENERATION_MODEL,
        }


    task5_rag_cache_df = run_cached_rows(
        task="task5_rag_answers",
        rows=package_rows(task1_questions_df.to_dict("records")),
        cache_path=TASK4_CACHE_PATH,
        config_payload={
            "model": GENERATION_MODEL,
            "prompt": RAG_SYSTEM_PROMPT_V1,
            "k": 4,
            "include_programmatic_sources": True,
        },
        input_payload_fn=lambda row: {
            "financebench_id": row["financebench_id"],
            "question": row["question"],
        },
        run_one_fn=run_task5_rag_one,
        force_rerun=FORCE_RERUN,
        retry_errors=RETRY_ERRORS,
        allow_run=RUN_MODEL_CALLS,
    )

    if task5_rag_cache_df.empty:
        task5_run_and_compare_df = pd.DataFrame()
    else:
        task5_run_and_compare_df = task5_rag_cache_df.merge(
            task1_naive_generation_df[
                ["financebench_id", "naive_answer"]
            ],
            on="financebench_id",
            how="left",
        )

    task5_export_paths = export_table(
        task5_run_and_compare_df,
        ASSIGNMENT2_COMPARE_BASE,
        [
            "financebench_id",
            "question_type",
            "question",
            "naive_answer",
            "RAG_answer",
            "ground_truth",
        ],
        OUTPUT_FORMAT,
    )
    return task5_export_paths, task5_run_and_compare_df


@app.cell
def _(display_table, mo, task5_run_and_compare_df):
    task5_compare_output = display_table(task5_run_and_compare_df, mo)
    task5_compare_output
    return


@app.cell(hide_code=True)
def _(mo):
    task5_qa_output = mo.md(f"""
    ### Write-up - Task 5

    Q: Did RAG help? Cases where the naive model refused or hallucinated, and RAG
    produced a grounded answer.

    A: Yes, but only when retrieval found the right filing and page. The clearest
    examples are the Best Buy cash question and the MGM EBITDAR question. Otherwise, RAG had provided more refusals at the moment.

    Q: Did RAG hurt? Cases where the naive model happened to be right (from memorization) and RAG made it worse - e.g., retrieved the wrong filing, or pulled chunks that confused the model.

    A: Naive model was right with JPM (financebench_id_00299), and due to incorrect retrieval RAG didn't answer at all.

    Q: Patterns by question_type - does RAG help more on domain-relevant than on novel-generated, or vice versa? Any hypothesis why?

    A: RAG helped more on the novel-generated questions when retrieval hit the specific filing evidence. It hurt or failed on several domain-relevant questions because broad finance terms like "working capital" or "gross margin" retrieved semantically similar chunks from the wrong pages or even wrong companies. 
    I believe a plausible hypothesis may be that semantic search over generic companies is too broad and non-particular. So it frequently retrieves wrong chunks. 

    """)
    task5_qa_output
    return


@app.cell(hide_code=True)
def _(mo):
    task6_output = mo.md(f"""
    ## Task 6 - Evaluation

    Measures:
    - correctness: DeepSeek judge, binary verdict plus one-sentence justification;
    - faithfulness: Ragas Faithfulness `.score()` on the first 20 sorted examples;
    - retrieval page-hit@k for `k = 1, 3, 5`.
    """)
    task6_output
    return


@app.cell
def _(
    JUDGE_MODEL,
    NEBIUS_API_KEY,
    NEBIUS_BASE_URL,
    RUN_FAITHFULNESS,
    call_chat_model,
    evidence_page_numbers,
    retrieve_chunks,
):
    from typing import Literal

    from pydantic import BaseModel, Field, ValidationError
    from tenacity import retry, stop_after_attempt, wait_exponential

    CORRECTNESS_JUDGE_PROMPT = """You judge whether a generated answer is correct compared with the ground truth.

    Return only JSON with:
    - verdict: "correct" or "incorrect"
    - justification: one sentence.

    Example:
    {
        "verdict": "correct",
        "justification": "The answer is correct."
    }
    """


    class CorrectnessJudgeResult(BaseModel):
        verdict: Literal["correct", "incorrect"]
        justification: str = Field(min_length=1)


    def judge_correctness(question: str, ground_truth: str, candidate_answer: str) -> dict:
        user_prompt = f"""Question:
    {question}

    Ground truth:
    {ground_truth}

    Candidate answer:
    {candidate_answer}
    """
        last_raw = ""

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            reraise=True,
        )
        def call_and_parse_judge() -> CorrectnessJudgeResult:
            nonlocal last_raw
            last_raw = call_chat_model(
                model=JUDGE_MODEL,
                api_key=NEBIUS_API_KEY,
                base_url=NEBIUS_BASE_URL,
                messages=[
                    {"role": "system", "content": CORRECTNESS_JUDGE_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=200,
            )
            return CorrectnessJudgeResult.model_validate_json(last_raw)

        try:
            parsed = call_and_parse_judge()
        except ValidationError:
            parsed = CorrectnessJudgeResult(
                verdict="incorrect",
                justification=f"Judge returned invalid JSON: {last_raw[:500]}",
            )
        except Exception as exc:
            parsed = CorrectnessJudgeResult(
                verdict="incorrect",
                justification=f"Judge call failed: {repr(exc)[:500]}",
            )
        return {
            "correctness": parsed.verdict,
            "correctness_justification": parsed.justification,
        }


    def score_faithfulness_with_ragas(
        *,
        question: str,
        answer: str,
        retrieved_contexts: list[str],
    ) -> float | None:
        if not RUN_FAITHFULNESS:
            return None
        from openai import AsyncOpenAI
        from ragas.llms import llm_factory
        from ragas.metrics.collections import Faithfulness

        client = AsyncOpenAI(api_key=NEBIUS_API_KEY, base_url=NEBIUS_BASE_URL)
        llm = llm_factory(JUDGE_MODEL, client=client)
        scorer = Faithfulness(llm=llm)
        result = scorer.score(
            user_input=question,
            response=answer,
            retrieved_contexts=retrieved_contexts,
        )
        return float(result.value)


    def compute_page_hit_at_k(row: dict, *, vectorstore, k: int) -> int:
        expected_pages = set(evidence_page_numbers(row.get("evidence")))
        if not expected_pages or vectorstore is None:
            return 0
        chunks = retrieve_chunks(vectorstore, row["question"], k=k)
        retrieved_pages = {
            int(chunk["page_number"])
            for chunk in chunks
            if chunk.get("doc_name") == row.get("doc_name")
            and str(chunk.get("page_number", "")).isdigit()
        }
        return int(bool(expected_pages & retrieved_pages))



    return (
        compute_page_hit_at_k,
        judge_correctness,
        score_faithfulness_with_ragas,
    )


@app.cell
def _(
    FAITHFULNESS_LIMIT,
    PAGE_HIT_K_VALUES,
    baseline_vectorstore,
    compute_page_hit_at_k,
    financebench_df,
    tqdm,
):
    task6_eval_base_df = financebench_df.sort_values("financebench_id").copy()
    if not task6_eval_base_df.empty:
        for _page_hit_k in PAGE_HIT_K_VALUES:
            task6_eval_base_df[f"page_hit_at_{_page_hit_k}"] = [
                compute_page_hit_at_k(row, vectorstore=baseline_vectorstore, k=_page_hit_k)
                for row in tqdm(
                    task6_eval_base_df.to_dict("records"),
                    desc=f"page_hit@{_page_hit_k}",
                    unit="row",
                )
            ]

    faithfulness_sample_df = task6_eval_base_df.head(FAITHFULNESS_LIMIT).copy()
    return (task6_eval_base_df,)


@app.cell
def _(
    ASSIGNMENT2_EVAL_BASE,
    FAITHFULNESS_LIMIT,
    FORCE_RERUN,
    GENERATION_MODEL,
    JUDGE_MODEL,
    OUTPUT_FORMAT,
    PAGE_HIT_K_VALUES,
    RETRY_ERRORS,
    RUN_EXPENSIVE_EVALS,
    TASK6_CORRECTNESS_CACHE_PATH,
    TASK6_FAITHFULNESS_CACHE_PATH,
    answer_with_rag,
    export_table,
    json,
    judge_correctness,
    package_rows,
    run_cached_rows,
    score_faithfulness_with_ragas,
    task6_eval_base_df,
):
    def run_task6_correctness_one(row: dict) -> dict:
        rag_result = answer_with_rag(str(row["question"]))
        judged = judge_correctness(
            question=str(row["question"]),
            ground_truth=str(row["answer"]),
            candidate_answer=rag_result["answer"],
        )
        return {
            "question": row["question"],
            "RAG_answer": rag_result["answer"],
            "retrieved_chunks_json": json.dumps(
                rag_result["retrieved_chunks"], ensure_ascii=True
            ),
            "ground_truth": row["answer"],
            **judged,
        }


    task6_correctness_cache_df = run_cached_rows(
        task="task6_correctness",
        rows=package_rows(task6_eval_base_df.to_dict("records")),
        cache_path=TASK6_CORRECTNESS_CACHE_PATH,
        config_payload={
            "generation_model": GENERATION_MODEL,
            "judge_model": JUDGE_MODEL,
            "k": 4,
        },
        input_payload_fn=lambda row: {
            "financebench_id": row["financebench_id"],
            "question": row["question"],
            "answer": row["answer"],
        },
        run_one_fn=run_task6_correctness_one,
        force_rerun=FORCE_RERUN,
        retry_errors=RETRY_ERRORS,
        allow_run=RUN_EXPENSIVE_EVALS,
    )

    if task6_correctness_cache_df.empty:
        assignment2_evaluation_df = task6_eval_base_df.copy()
        assignment2_evaluation_df["correctness"] = ""
        assignment2_evaluation_df["faithfulness"] = ""
    else:
        for _column in [
            "correctness",
            "correctness_justification",
            "RAG_answer",
            "retrieved_chunks_json",
        ]:
            if _column not in task6_correctness_cache_df.columns:
                task6_correctness_cache_df[_column] = ""
        assignment2_evaluation_df = task6_eval_base_df.merge(
            task6_correctness_cache_df[
                [
                    "financebench_id",
                    "correctness",
                    "correctness_justification",
                    "RAG_answer",
                    "retrieved_chunks_json",
                ]
            ],
            on="financebench_id",
            how="left",
        )
        assignment2_evaluation_df["faithfulness"] = ""

    def run_task6_faithfulness_one(row: dict) -> dict:
        retrieved_chunks = []
        if row.get("retrieved_chunks_json"):
            try:
                retrieved_chunks = json.loads(row["retrieved_chunks_json"])
            except json.JSONDecodeError:
                retrieved_chunks = []
        faithfulness = score_faithfulness_with_ragas(
            question=str(row["question"]),
            answer=str(row.get("RAG_answer", "")),
            retrieved_contexts=[
                str(chunk.get("content", "")) for chunk in retrieved_chunks
            ],
        )
        return {"faithfulness": faithfulness}


    faithfulness_input_df = (
        assignment2_evaluation_df.sort_values("financebench_id")
        .head(FAITHFULNESS_LIMIT)
        .copy()
    )
    task6_faithfulness_cache_df = run_cached_rows(
        task="task6_faithfulness",
        rows=package_rows(faithfulness_input_df.to_dict("records")),
        cache_path=TASK6_FAITHFULNESS_CACHE_PATH,
        config_payload={
            "generation_model": GENERATION_MODEL,
            "judge_model": JUDGE_MODEL,
            "faithfulness_limit": FAITHFULNESS_LIMIT,
            "metric": "ragas_faithfulness_score",
        },
        input_payload_fn=lambda row: {
            "financebench_id": row["financebench_id"],
            "question": row["question"],
            "RAG_answer": row.get("RAG_answer", ""),
            "retrieved_chunks_json": row.get("retrieved_chunks_json", ""),
        },
        run_one_fn=run_task6_faithfulness_one,
        force_rerun=FORCE_RERUN,
        retry_errors=RETRY_ERRORS,
        allow_run=RUN_EXPENSIVE_EVALS,
    )
    if (
        not task6_faithfulness_cache_df.empty
        and "faithfulness" in task6_faithfulness_cache_df.columns
    ):
        assignment2_evaluation_df = assignment2_evaluation_df.drop(
            columns=["faithfulness"], errors="ignore"
        ).merge(
            task6_faithfulness_cache_df[["financebench_id", "faithfulness"]],
            on="financebench_id",
            how="left",
        )
    elif "faithfulness" not in assignment2_evaluation_df.columns:
        assignment2_evaluation_df["faithfulness"] = ""

    eval_columns = [
        "financebench_id",
        "question",
        "correctness",
        "faithfulness",
        *[f"page_hit_at_{_page_hit_k}" for _page_hit_k in PAGE_HIT_K_VALUES],
    ]
    task6_export_paths = export_table(
        assignment2_evaluation_df,
        ASSIGNMENT2_EVAL_BASE,
        eval_columns,
        OUTPUT_FORMAT,
    )
    return assignment2_evaluation_df, task6_export_paths


@app.cell
def _(PAGE_HIT_K_VALUES, assignment2_evaluation_df, pd):
    def aggregate_evaluation(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        aggregate_row = {}
        if "correctness" in df.columns:
            aggregate_row["average_correctness"] = (
                df["correctness"].eq("correct").mean()
                if df["correctness"].notna().any()
                else None
            )
        if "faithfulness" in df.columns:
            aggregate_row["average_faithfulness"] = pd.to_numeric(
                df["faithfulness"], errors="coerce"
            ).mean()
        for _page_hit_k in PAGE_HIT_K_VALUES:
            column = f"page_hit_at_{_page_hit_k}"
            if column in df.columns:
                aggregate_row[column] = pd.to_numeric(
                    df[column], errors="coerce"
                ).mean()
        return pd.DataFrame([aggregate_row])


    task6_aggregate_df = aggregate_evaluation(assignment2_evaluation_df)
    return (task6_aggregate_df,)


@app.cell
def _(PAGE_HIT_K_VALUES, round_metric_columns, task6_aggregate_df):
    task6_aggregate_display_df = round_metric_columns(
        task6_aggregate_df,
        [
            "average_correctness",
            "average_faithfulness",
            *[f"page_hit_at_{_page_hit_k}" for _page_hit_k in PAGE_HIT_K_VALUES],
        ],
    )
    return (task6_aggregate_display_df,)


@app.cell
def _(display_table, mo, task6_aggregate_display_df):
    task6_aggregate_output = display_table(task6_aggregate_display_df, mo)
    task6_aggregate_output
    return


@app.cell(hide_code=True)
def _(mo, task6_aggregate_display_df):
    task6_qa_output = mo.md(f"""
    Q: What are the aggregate correctness, faithfulness, and retrieval page-hit results?

    A: 
    Aggregate **correctness**: {task6_aggregate_display_df["average_correctness"].iloc[0]}

    Aggregate **faithfulness**: {task6_aggregate_display_df["average_faithfulness"].iloc[0]}

    Aggregate **retrieval page-hit** results:
    {task6_aggregate_display_df.iloc[0][["page_hit_at_1", "page_hit_at_3", "page_hit_at_5"]].to_dict()}
    """)
    task6_qa_output
    return


@app.cell(hide_code=True)
def _(mo):
    task7_output = mo.md(f"""
    ## Task 7 - Improvement Cycles

    Baseline plus experiments are represented by `RagExperimentConfig`.

    Suggested first experiments:
    1. `prompt_v2`: stricter citation and no-answer wording.
    2. `k_8`: retrieve 8 chunks into the generator.
    3. `chunk_500`: rebuild FAISS with 500-character chunks.
    4. `reranker`: retrieve top 20, rerank to top 4 with `BAAI/bge-reranker-base`.
    """)
    task7_output
    return


@app.cell
def _(
    BASELINE_CHUNK_OVERLAP,
    BASELINE_CHUNK_SIZE,
    BASELINE_VECTORSTORE_NAME,
    GENERATION_MODEL,
    RagExperimentConfig,
    pd,
):
    baseline_experiment_config = RagExperimentConfig(
        experiment="baseline",
        change="Task 6 baseline: chunk_size=1000, overlap=150, k=4",
        vectorstore_name=BASELINE_VECTORSTORE_NAME,
        chunk_size=BASELINE_CHUNK_SIZE,
        chunk_overlap=BASELINE_CHUNK_OVERLAP,
        generator_model=GENERATION_MODEL,
        generation_prompt_version="v1",
        k_for_generation=4,
        hypothesis="Baseline establishes the current retrieval and generation quality before changes.",
    )

    RAG_SYSTEM_PROMPT_V2 = """You answer finance questions using only the provided context.

    Rules:
    1. Answer only from the retrieved context, not from memory.
    2. If the retrieved context does not contain the answer, say exactly: "The provided context does not contain the answer."
    3. Keep the answer concise.
    4. Cite every factual claim using [doc_name p.page_number].
    5. Do not cite retrieved sources that do not support the answer.
    """

    TASK7_EXPERIMENT_OVERRIDES = {}

    task7_experiment_configs = [
        baseline_experiment_config,
        RagExperimentConfig(
            experiment="prompt_v2",
            change="Stricter generation prompt with explicit citation discipline",
            vectorstore_name=BASELINE_VECTORSTORE_NAME,
            chunk_size=BASELINE_CHUNK_SIZE,
            chunk_overlap=BASELINE_CHUNK_OVERLAP,
            generator_model=GENERATION_MODEL,
            generation_prompt_version="v2",
            k_for_generation=4,
            hypothesis="A stricter prompt should reduce unsupported answers and improve citation discipline.",
        ),
        RagExperimentConfig(
            experiment="k_8",
            change="Increase chunks passed to generator from 4 to 8",
            vectorstore_name=BASELINE_VECTORSTORE_NAME,
            chunk_size=BASELINE_CHUNK_SIZE,
            chunk_overlap=BASELINE_CHUNK_OVERLAP,
            generator_model=GENERATION_MODEL,
            generation_prompt_version="v1",
            k_for_generation=8,
            hypothesis="More chunks should increase the chance that the evidence page reaches the generator.",
        ),
        RagExperimentConfig(
            experiment="chunk_500",
            change="Rebuild FAISS with chunk_size=500 and overlap=75",
            vectorstore_name="faiss_chunk500",
            chunk_size=500,
            chunk_overlap=75,
            generator_model=GENERATION_MODEL,
            generation_prompt_version="v1",
            k_for_generation=4,
            hypothesis="Smaller chunks may retrieve more targeted balance-sheet and table evidence.",
        ),
        RagExperimentConfig(
            experiment="reranker",
            change="Retrieve top 20 with FAISS, rerank to 4 with bge-reranker-base",
            vectorstore_name=BASELINE_VECTORSTORE_NAME,
            chunk_size=BASELINE_CHUNK_SIZE,
            chunk_overlap=BASELINE_CHUNK_OVERLAP,
            generator_model=GENERATION_MODEL,
            generation_prompt_version="v1",
            k_for_generation=4,
            use_reranker=True,
            rerank_fetch_k=20,
            hypothesis="A cross-encoder reranker should demote semantically related but unsupported chunks.",
        ),
    ]

    task7_experiment_config_df = pd.DataFrame(
        [config.to_payload() for config in task7_experiment_configs]
    )

    IMPROVEMENT_INDEX = 1
    selected_improvement_config = task7_experiment_configs[IMPROVEMENT_INDEX]
    return (
        RAG_SYSTEM_PROMPT_V2,
        TASK7_EXPERIMENT_OVERRIDES,
        selected_improvement_config,
        task7_experiment_config_df,
        task7_experiment_configs,
    )


@app.cell
def _(
    ASSIGNMENT2_VECTORSTORE_DIR,
    EMBEDDING_MODEL,
    FAITHFULNESS_LIMIT,
    FORCE_RERUN,
    JUDGE_MODEL,
    PAGE_HIT_K_VALUES,
    Path,
    RAG_SYSTEM_PROMPT_V1,
    RAG_SYSTEM_PROMPT_V2,
    RERANKER_MODEL,
    RETRY_ERRORS,
    RUN_EXPENSIVE_EVALS,
    TASK7_EXPERIMENT_CACHE_PATH,
    TASK7_EXPERIMENT_OVERRIDES,
    answer_from_chunks,
    baseline_vectorstore,
    build_or_load_vectorstore,
    evidence_page_numbers,
    financebench_df,
    judge_correctness,
    package_rows,
    pd,
    referenced_documents_df,
    run_cached_rows,
    score_faithfulness_with_ragas,
    source_fingerprint,
    task6_aggregate_df,
    task7_experiment_config_df,
    task7_experiment_configs,
):
    def task7_manifest(config, embedding_model: str) -> dict:
        document_fingerprints = []
        for row in referenced_documents_df.to_dict("records"):
            document_fingerprints.append(
                {
                    "doc_name": row["doc_name"],
                    "company": row.get("company", ""),
                    "doc_period": row.get("doc_period", ""),
                    "fingerprint": source_fingerprint(Path(row["pdf_path"])),
                }
            )
        return {
            "embedding_model": embedding_model,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "documents": document_fingerprints,
        }


    def vectorstore_for_experiment(config, embedding_model: str):
        if config.vectorstore_name == "faiss_chunk1000" and embedding_model == EMBEDDING_MODEL:
            return baseline_vectorstore
        vectorstore_path = ASSIGNMENT2_VECTORSTORE_DIR / config.vectorstore_name
        return build_or_load_vectorstore(
            documents_df=referenced_documents_df,
            vectorstore_path=vectorstore_path,
            manifest_path=vectorstore_path / "manifest.json",
            manifest_payload=task7_manifest(config, embedding_model),
            embedding_model=embedding_model,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            force_rebuild=FORCE_RERUN,
        )


    def page_hit_for_chunks(row: dict, chunks: list[dict]) -> int:
        expected_pages = set(evidence_page_numbers(row.get("evidence")))
        retrieved_pages = {
            int(chunk["page_number"])
            for chunk in chunks
            if chunk.get("doc_name") == row.get("doc_name")
            and str(chunk.get("page_number", "")).isdigit()
        }
        return int(bool(expected_pages & retrieved_pages))


    def retrieve_for_experiment(
        config,
        vectorstore,
        query: str,
        reranker=None,
        generation_k: int | None = None,
    ) -> list[dict]:
        result_k = generation_k or config.k_for_generation
        if config.use_reranker:
            candidate_chunks = vectorstore.similarity_search(
                query, k=config.rerank_fetch_k
            )
            candidate_chunks = [
                {
                    "doc_name": doc.metadata.get("doc_name", ""),
                    "page_number": doc.metadata.get("page_number", ""),
                    "company": doc.metadata.get("company", ""),
                    "doc_period": doc.metadata.get("doc_period", ""),
                    "content": doc.page_content,
                }
                for doc in candidate_chunks
            ]
            if not candidate_chunks or reranker is None:
                return candidate_chunks[:result_k]
            pairs = [(query, chunk["content"]) for chunk in candidate_chunks]
            scores = reranker.predict(pairs)
            ranked = sorted(
                zip(candidate_chunks, scores),
                key=lambda item: float(item[1]),
                reverse=True,
            )
            return [chunk for chunk, _score in ranked[:result_k]]
        docs = vectorstore.similarity_search(query, k=result_k)
        return [
            {
                "doc_name": doc.metadata.get("doc_name", ""),
                "page_number": doc.metadata.get("page_number", ""),
                "company": doc.metadata.get("company", ""),
                "doc_period": doc.metadata.get("doc_period", ""),
                "content": doc.page_content,
            }
            for doc in docs
        ]


    def aggregate_task7_results(task7_attempt_rows_df: pd.DataFrame) -> pd.DataFrame:
        result_rows = []
        if not task6_aggregate_df.empty:
            baseline_metrics = task6_aggregate_df.iloc[0].to_dict()
            result_rows.append(
                {
                    "experiment": "baseline",
                    "change": "Task 6 baseline: chunk_size=1000, overlap=150, k=4",
                    "correctness": baseline_metrics.get("average_correctness", ""),
                    "faithfulness": baseline_metrics.get("average_faithfulness", ""),
                    **{
                        f"page_hit_at_{k}": baseline_metrics.get(f"page_hit_at_{k}", "")
                        for k in PAGE_HIT_K_VALUES
                    },
                }
            )
        for config in task7_experiment_configs[1:]:
            if task7_attempt_rows_df.empty or "experiment" not in task7_attempt_rows_df:
                experiment_rows = pd.DataFrame()
            else:
                experiment_rows = task7_attempt_rows_df[
                    task7_attempt_rows_df["experiment"] == config.experiment
                ]
            result_row = {
                "experiment": config.experiment,
                "change": config.change,
                "correctness": "",
                "faithfulness": "",
                **{f"page_hit_at_{k}": "" for k in PAGE_HIT_K_VALUES},
            }
            if not experiment_rows.empty and "correctness" in experiment_rows.columns:
                result_row["correctness"] = experiment_rows["correctness"].eq(
                    "correct"
                ).mean()
            if not experiment_rows.empty and "faithfulness" in experiment_rows.columns:
                result_row["faithfulness"] = pd.to_numeric(
                    experiment_rows["faithfulness"], errors="coerce"
                ).mean()
            if not experiment_rows.empty:
                for k in PAGE_HIT_K_VALUES:
                    column = f"page_hit_at_{k}"
                    if column in experiment_rows.columns:
                        result_row[column] = pd.to_numeric(
                            experiment_rows[column], errors="coerce"
                        ).mean()
            result_rows.append(result_row)
        return pd.DataFrame(result_rows).merge(
            task7_experiment_config_df[
                ["experiment", "hypothesis"]
            ],
            on="experiment",
            how="left",
        )


    task7_attempt_frames = []
    for config in task7_experiment_configs[1:]:
        overrides = TASK7_EXPERIMENT_OVERRIDES.get(config.experiment, {})
        experiment_embedding_model = overrides.get("embedding_model", EMBEDDING_MODEL)
        if RUN_EXPENSIVE_EVALS and config.use_reranker:
            from sentence_transformers import CrossEncoder

            reranker = CrossEncoder(RERANKER_MODEL)
        else:
            reranker = None
        vectorstore = (
            vectorstore_for_experiment(config, experiment_embedding_model)
            if RUN_EXPENSIVE_EVALS
            else None
        )
        prompt = (
            RAG_SYSTEM_PROMPT_V2
            if config.generation_prompt_version == "v2"
            else RAG_SYSTEM_PROMPT_V1
        )
        experiment_rows = [
            {
                **row,
                "_faithfulness_eval": row_index < FAITHFULNESS_LIMIT,
            }
            for row_index, row in enumerate(
                financebench_df.sort_values("financebench_id").to_dict("records")
            )
        ]

        def run_task7_experiment_one(row: dict) -> dict:
            query = str(row["question"])
            chunks = retrieve_for_experiment(config, vectorstore, query, reranker)
            rag_result = answer_from_chunks(
                query=query,
                chunks=chunks,
                model=config.generator_model,
                system_prompt=prompt,
                include_programmatic_sources=True,
            )
            judged = judge_correctness(
                question=query,
                ground_truth=str(row["answer"]),
                candidate_answer=rag_result["answer"],
            )
            faithfulness = (
                score_faithfulness_with_ragas(
                    question=query,
                    answer=rag_result["answer"],
                    retrieved_contexts=[chunk["content"] for chunk in chunks],
                )
                if row.get("_faithfulness_eval")
                else None
            )
            return {
                "experiment": config.experiment,
                "question": row["question"],
                "RAG_answer": rag_result["answer"],
                "ground_truth": row["answer"],
                "faithfulness": faithfulness,
                **{
                    f"page_hit_at_{k}": page_hit_for_chunks(
                        row,
                        retrieve_for_experiment(
                            config,
                            vectorstore,
                            query,
                            reranker if config.use_reranker else None,
                            generation_k=k,
                        ),
                    )
                    for k in PAGE_HIT_K_VALUES
                },
                **judged,
            }

        experiment_df = run_cached_rows(
            task="task7_experiment",
            rows=package_rows(experiment_rows),
            cache_path=TASK7_EXPERIMENT_CACHE_PATH,
            config_payload={
                **config.to_payload(),
                "embedding_model": experiment_embedding_model,
                "judge_model": JUDGE_MODEL,
                "prompt": prompt,
                **overrides,
            },
            input_payload_fn=lambda row: {
                "financebench_id": row["financebench_id"],
                "question": row["question"],
                "answer": row["answer"],
            },
            run_one_fn=run_task7_experiment_one,
            force_rerun=FORCE_RERUN,
            retry_errors=RETRY_ERRORS,
            allow_run=RUN_EXPENSIVE_EVALS,
        )
        task7_attempt_frames.append(experiment_df)

    task7_attempt_rows_df = (
        pd.concat(task7_attempt_frames, ignore_index=True)
        if task7_attempt_frames
        else pd.DataFrame()
    )
    task7_results_df = aggregate_task7_results(task7_attempt_rows_df)
    return task7_attempt_rows_df, task7_results_df


@app.cell
def _(
    ASSIGNMENT2_VECTORSTORE_DIR,
    BASELINE_CHUNK_OVERLAP,
    BASELINE_CHUNK_SIZE,
    FAITHFULNESS_LIMIT,
    FORCE_RERUN,
    GENERATION_MODEL,
    JUDGE_MODEL,
    PAGE_HIT_K_VALUES,
    Path,
    RAG_SYSTEM_PROMPT_V1,
    RETRY_ERRORS,
    RUN_EXPENSIVE_EVALS,
    TASK7_EXPERIMENT_CACHE_PATH,
    answer_from_chunks,
    build_or_load_vectorstore,
    evidence_page_numbers,
    financebench_df,
    judge_correctness,
    package_rows,
    pd,
    referenced_documents_df,
    run_cached_rows,
    score_faithfulness_with_ragas,
    source_fingerprint,
):
    task7_embedding_experiment = {
        "experiment": "embedding_gemini_2",
        "change": "Rebuild FAISS with gemini-embedding-2 embeddings",
        "hypothesis": "Gemini embeddings may improve page-hit by using a stronger external embedding model.",
        "embedding_model": "gemini/gemini-embedding-2",
        "vectorstore_name": "faiss_gemini_embedding2_chunk1000",
        "chunk_size": BASELINE_CHUNK_SIZE,
        "chunk_overlap": BASELINE_CHUNK_OVERLAP,
        "k_for_generation": 4,
    }

    def task7_embedding_manifest() -> dict:
        document_fingerprints = []
        for row in referenced_documents_df.to_dict("records"):
            document_fingerprints.append(
                {
                    "doc_name": row["doc_name"],
                    "company": row.get("company", ""),
                    "doc_period": row.get("doc_period", ""),
                    "fingerprint": source_fingerprint(Path(row["pdf_path"])),
                }
            )
        return {
            "embedding_model": task7_embedding_experiment["embedding_model"],
            "chunk_size": task7_embedding_experiment["chunk_size"],
            "chunk_overlap": task7_embedding_experiment["chunk_overlap"],
            "documents": document_fingerprints,
        }

    def task7_embedding_retrieve(vectorstore, query: str, k: int) -> list[dict]:
        docs = vectorstore.similarity_search(query, k=k)
        return [
            {
                "doc_name": doc.metadata.get("doc_name", ""),
                "page_number": doc.metadata.get("page_number", ""),
                "company": doc.metadata.get("company", ""),
                "doc_period": doc.metadata.get("doc_period", ""),
                "content": doc.page_content,
            }
            for doc in docs
        ]

    def task7_embedding_page_hit(row: dict, chunks: list[dict]) -> int:
        expected_pages = set(evidence_page_numbers(row.get("evidence")))
        retrieved_pages = {
            int(chunk["page_number"])
            for chunk in chunks
            if chunk.get("doc_name") == row.get("doc_name")
            and str(chunk.get("page_number", "")).isdigit()
        }
        return int(bool(expected_pages & retrieved_pages))

    task7_embedding_vectorstore_path = (
        ASSIGNMENT2_VECTORSTORE_DIR / task7_embedding_experiment["vectorstore_name"]
    )
    task7_embedding_vectorstore = (
        build_or_load_vectorstore(
            documents_df=referenced_documents_df,
            vectorstore_path=task7_embedding_vectorstore_path,
            manifest_path=task7_embedding_vectorstore_path / "manifest.json",
            manifest_payload=task7_embedding_manifest(),
            embedding_model=task7_embedding_experiment["embedding_model"],
            chunk_size=task7_embedding_experiment["chunk_size"],
            chunk_overlap=task7_embedding_experiment["chunk_overlap"],
            force_rebuild=FORCE_RERUN,
        )
        if RUN_EXPENSIVE_EVALS
        else None
    )

    task7_embedding_rows = [
        {
            **row,
            "_faithfulness_eval": row_index < FAITHFULNESS_LIMIT,
        }
        for row_index, row in enumerate(
            financebench_df.sort_values("financebench_id").to_dict("records")
        )
    ]

    def run_task7_embedding_one(row: dict) -> dict:
        query = str(row["question"])
        chunks = task7_embedding_retrieve(
            task7_embedding_vectorstore,
            query,
            task7_embedding_experiment["k_for_generation"],
        )
        rag_result = answer_from_chunks(
            query=query,
            chunks=chunks,
            model=GENERATION_MODEL,
            system_prompt=RAG_SYSTEM_PROMPT_V1,
            include_programmatic_sources=True,
        )
        judged = judge_correctness(
            question=query,
            ground_truth=str(row["answer"]),
            candidate_answer=rag_result["answer"],
        )
        faithfulness = (
            score_faithfulness_with_ragas(
                question=query,
                answer=rag_result["answer"],
                retrieved_contexts=[chunk["content"] for chunk in chunks],
            )
            if row.get("_faithfulness_eval")
            else None
        )
        return {
            "experiment": task7_embedding_experiment["experiment"],
            "question": row["question"],
            "RAG_answer": rag_result["answer"],
            "ground_truth": row["answer"],
            "faithfulness": faithfulness,
            **{
                f"page_hit_at_{k}": task7_embedding_page_hit(
                    row,
                    task7_embedding_retrieve(task7_embedding_vectorstore, query, k),
                )
                for k in PAGE_HIT_K_VALUES
            },
            **judged,
        }

    task7_embedding_attempt_rows_df = run_cached_rows(
        task="task7_embedding_experiment",
        rows=package_rows(task7_embedding_rows),
        cache_path=TASK7_EXPERIMENT_CACHE_PATH,
        config_payload={
            **task7_embedding_experiment,
            "generator_model": GENERATION_MODEL,
            "judge_model": JUDGE_MODEL,
            "prompt": RAG_SYSTEM_PROMPT_V1,
        },
        input_payload_fn=lambda row: {
            "financebench_id": row["financebench_id"],
            "question": row["question"],
            "answer": row["answer"],
        },
        run_one_fn=run_task7_embedding_one,
        force_rerun=FORCE_RERUN,
        retry_errors=RETRY_ERRORS,
        allow_run=RUN_EXPENSIVE_EVALS,
        max_workers=1,
    )

    task7_embedding_result_row = {
        "experiment": task7_embedding_experiment["experiment"],
        "change": task7_embedding_experiment["change"],
        "correctness": "",
        "faithfulness": "",
        **{f"page_hit_at_{k}": "" for k in PAGE_HIT_K_VALUES},
        "hypothesis": task7_embedding_experiment["hypothesis"],
    }
    if not task7_embedding_attempt_rows_df.empty:
        if "correctness" in task7_embedding_attempt_rows_df.columns:
            task7_embedding_result_row["correctness"] = (
                task7_embedding_attempt_rows_df["correctness"].eq("correct").mean()
            )
        if "faithfulness" in task7_embedding_attempt_rows_df.columns:
            task7_embedding_result_row["faithfulness"] = pd.to_numeric(
                task7_embedding_attempt_rows_df["faithfulness"],
                errors="coerce",
            ).mean()
        for k in PAGE_HIT_K_VALUES:
            _column = f"page_hit_at_{k}"
            if _column in task7_embedding_attempt_rows_df.columns:
                task7_embedding_result_row[_column] = pd.to_numeric(
                    task7_embedding_attempt_rows_df[_column],
                    errors="coerce",
                ).mean()

    task7_embedding_results_df = pd.DataFrame([task7_embedding_result_row])
    return (task7_embedding_results_df,)


@app.cell
def _(
    ASSIGNMENT2_VECTORSTORE_DIR,
    BASELINE_CHUNK_OVERLAP,
    BASELINE_CHUNK_SIZE,
    FAITHFULNESS_LIMIT,
    FORCE_RERUN,
    GENERATION_MODEL,
    JUDGE_MODEL,
    PAGE_HIT_K_VALUES,
    Path,
    RAG_SYSTEM_PROMPT_V1,
    RETRY_ERRORS,
    RUN_EXPENSIVE_EVALS,
    TASK7_EXPERIMENT_CACHE_PATH,
    answer_from_chunks,
    build_or_load_vectorstore,
    evidence_page_numbers,
    financebench_df,
    judge_correctness,
    package_rows,
    pd,
    referenced_documents_df,
    run_cached_rows,
    score_faithfulness_with_ragas,
    source_fingerprint,
):
    task7_embedding_k8_experiment = {
        "experiment": "embedding_gemini_2_k8",
        "change": "Use gemini-embedding-2 embeddings with k=8",
        "hypothesis": "Combining Gemini embeddings with more retrieved chunks may improve answer coverage when relevant evidence is near the top results.",
        "embedding_model": "gemini/gemini-embedding-2",
        "vectorstore_name": "faiss_gemini_embedding2_chunk1000",
        "chunk_size": BASELINE_CHUNK_SIZE,
        "chunk_overlap": BASELINE_CHUNK_OVERLAP,
        "k_for_generation": 8,
    }

    def task7_embedding_k8_manifest() -> dict:
        document_fingerprints = []
        for row in referenced_documents_df.to_dict("records"):
            document_fingerprints.append(
                {
                    "doc_name": row["doc_name"],
                    "company": row.get("company", ""),
                    "doc_period": row.get("doc_period", ""),
                    "fingerprint": source_fingerprint(Path(row["pdf_path"])),
                }
            )
        return {
            "embedding_model": task7_embedding_k8_experiment["embedding_model"],
            "chunk_size": task7_embedding_k8_experiment["chunk_size"],
            "chunk_overlap": task7_embedding_k8_experiment["chunk_overlap"],
            "documents": document_fingerprints,
        }

    def task7_embedding_k8_retrieve(vectorstore, query: str, k: int) -> list[dict]:
        docs = vectorstore.similarity_search(query, k=k)
        return [
            {
                "doc_name": doc.metadata.get("doc_name", ""),
                "page_number": doc.metadata.get("page_number", ""),
                "company": doc.metadata.get("company", ""),
                "doc_period": doc.metadata.get("doc_period", ""),
                "content": doc.page_content,
            }
            for doc in docs
        ]

    def task7_embedding_k8_page_hit(row: dict, chunks: list[dict]) -> int:
        expected_pages = set(evidence_page_numbers(row.get("evidence")))
        retrieved_pages = {
            int(chunk["page_number"])
            for chunk in chunks
            if chunk.get("doc_name") == row.get("doc_name")
            and str(chunk.get("page_number", "")).isdigit()
        }
        return int(bool(expected_pages & retrieved_pages))

    task7_embedding_k8_vectorstore_path = (
        ASSIGNMENT2_VECTORSTORE_DIR / task7_embedding_k8_experiment["vectorstore_name"]
    )
    task7_embedding_k8_vectorstore = (
        build_or_load_vectorstore(
            documents_df=referenced_documents_df,
            vectorstore_path=task7_embedding_k8_vectorstore_path,
            manifest_path=task7_embedding_k8_vectorstore_path / "manifest.json",
            manifest_payload=task7_embedding_k8_manifest(),
            embedding_model=task7_embedding_k8_experiment["embedding_model"],
            chunk_size=task7_embedding_k8_experiment["chunk_size"],
            chunk_overlap=task7_embedding_k8_experiment["chunk_overlap"],
            force_rebuild=FORCE_RERUN,
        )
        if RUN_EXPENSIVE_EVALS
        else None
    )

    task7_embedding_k8_rows = [
        {
            **row,
            "_faithfulness_eval": row_index < FAITHFULNESS_LIMIT,
        }
        for row_index, row in enumerate(
            financebench_df.sort_values("financebench_id").to_dict("records")
        )
    ]

    def run_task7_embedding_k8_one(row: dict) -> dict:
        query = str(row["question"])
        chunks = task7_embedding_k8_retrieve(
            task7_embedding_k8_vectorstore,
            query,
            task7_embedding_k8_experiment["k_for_generation"],
        )
        rag_result = answer_from_chunks(
            query=query,
            chunks=chunks,
            model=GENERATION_MODEL,
            system_prompt=RAG_SYSTEM_PROMPT_V1,
            include_programmatic_sources=True,
        )
        judged = judge_correctness(
            question=query,
            ground_truth=str(row["answer"]),
            candidate_answer=rag_result["answer"],
        )
        faithfulness = (
            score_faithfulness_with_ragas(
                question=query,
                answer=rag_result["answer"],
                retrieved_contexts=[chunk["content"] for chunk in chunks],
            )
            if row.get("_faithfulness_eval")
            else None
        )
        return {
            "experiment": task7_embedding_k8_experiment["experiment"],
            "question": row["question"],
            "RAG_answer": rag_result["answer"],
            "ground_truth": row["answer"],
            "faithfulness": faithfulness,
            **{
                f"page_hit_at_{k}": task7_embedding_k8_page_hit(
                    row,
                    task7_embedding_k8_retrieve(
                        task7_embedding_k8_vectorstore,
                        query,
                        k,
                    ),
                )
                for k in PAGE_HIT_K_VALUES
            },
            **judged,
        }

    task7_embedding_k8_attempt_rows_df = run_cached_rows(
        task="task7_embedding_k8_experiment",
        rows=package_rows(task7_embedding_k8_rows),
        cache_path=TASK7_EXPERIMENT_CACHE_PATH,
        config_payload={
            **task7_embedding_k8_experiment,
            "generator_model": GENERATION_MODEL,
            "judge_model": JUDGE_MODEL,
            "prompt": RAG_SYSTEM_PROMPT_V1,
        },
        input_payload_fn=lambda row: {
            "financebench_id": row["financebench_id"],
            "question": row["question"],
            "answer": row["answer"],
        },
        run_one_fn=run_task7_embedding_k8_one,
        force_rerun=FORCE_RERUN,
        retry_errors=RETRY_ERRORS,
        allow_run=RUN_EXPENSIVE_EVALS,
        max_workers=5,
    )

    task7_embedding_k8_result_row = {
        "experiment": task7_embedding_k8_experiment["experiment"],
        "change": task7_embedding_k8_experiment["change"],
        "correctness": "",
        "faithfulness": "",
        **{f"page_hit_at_{k}": "" for k in PAGE_HIT_K_VALUES},
        "hypothesis": task7_embedding_k8_experiment["hypothesis"],
    }
    if not task7_embedding_k8_attempt_rows_df.empty:
        if "correctness" in task7_embedding_k8_attempt_rows_df.columns:
            task7_embedding_k8_result_row["correctness"] = (
                task7_embedding_k8_attempt_rows_df["correctness"].eq("correct").mean()
            )
        if "faithfulness" in task7_embedding_k8_attempt_rows_df.columns:
            task7_embedding_k8_result_row["faithfulness"] = pd.to_numeric(
                task7_embedding_k8_attempt_rows_df["faithfulness"],
                errors="coerce",
            ).mean()
        for _k in PAGE_HIT_K_VALUES:
            _column = f"page_hit_at_{_k}"
            if _column in task7_embedding_k8_attempt_rows_df.columns:
                task7_embedding_k8_result_row[_column] = pd.to_numeric(
                    task7_embedding_k8_attempt_rows_df[_column],
                    errors="coerce",
                ).mean()

    task7_embedding_k8_results_df = pd.DataFrame([task7_embedding_k8_result_row])
    return (task7_embedding_k8_results_df,)


@app.cell
def _(
    ASSIGNMENT2_VECTORSTORE_DIR,
    BASELINE_CHUNK_OVERLAP,
    BASELINE_CHUNK_SIZE,
    FAITHFULNESS_LIMIT,
    FORCE_RERUN,
    GENERATION_MODEL,
    JUDGE_MODEL,
    PAGE_HIT_K_VALUES,
    Path,
    RAG_SYSTEM_PROMPT_V1,
    RETRY_ERRORS,
    RUN_EXPENSIVE_EVALS,
    TASK7_EXPERIMENT_CACHE_PATH,
    answer_from_chunks,
    build_or_load_vectorstore,
    evidence_page_numbers,
    financebench_df,
    judge_correctness,
    package_rows,
    pd,
    referenced_documents_df,
    run_cached_rows,
    score_faithfulness_with_ragas,
    source_fingerprint,
):
    task7_hybrid_bm25_gemini_k8_experiment = {
        "experiment": "hybrid_bm25_gemini_2_k8",
        "change": "Fuse BM25 lexical retrieval with gemini-embedding-2 semantic retrieval, k=8",
        "hypothesis": "Hybrid lexical and semantic retrieval should improve exact company, year, and financial metric matching while preserving semantic recall.",
        "embedding_model": "gemini/gemini-embedding-2",
        "vectorstore_name": "faiss_gemini_embedding2_chunk1000",
        "chunk_size": BASELINE_CHUNK_SIZE,
        "chunk_overlap": BASELINE_CHUNK_OVERLAP,
        "k_for_generation": 8,
        "semantic_fetch_k": 20,
        "bm25_fetch_k": 20,
    }

    def ensure_rank_bm25_installed() -> None:
        import importlib.util

        if importlib.util.find_spec("rank_bm25") is None:
            import subprocess
            import sys

            print("[task7_hybrid_bm25_gemini_2_k8] Installing rank-bm25")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "rank-bm25"])

    def task7_hybrid_bm25_manifest() -> dict:
        document_fingerprints = []
        for row in referenced_documents_df.to_dict("records"):
            document_fingerprints.append(
                {
                    "doc_name": row["doc_name"],
                    "company": row.get("company", ""),
                    "doc_period": row.get("doc_period", ""),
                    "fingerprint": source_fingerprint(Path(row["pdf_path"])),
                }
            )
        return {
            "embedding_model": task7_hybrid_bm25_gemini_k8_experiment["embedding_model"],
            "chunk_size": task7_hybrid_bm25_gemini_k8_experiment["chunk_size"],
            "chunk_overlap": task7_hybrid_bm25_gemini_k8_experiment["chunk_overlap"],
            "documents": document_fingerprints,
        }

    def task7_hybrid_doc_to_chunk(doc) -> dict:
        return {
            "doc_name": doc.metadata.get("doc_name", ""),
            "page_number": doc.metadata.get("page_number", ""),
            "company": doc.metadata.get("company", ""),
            "doc_period": doc.metadata.get("doc_period", ""),
            "content": doc.page_content,
        }

    def task7_hybrid_chunk_key(chunk: dict) -> tuple:
        return (
            chunk.get("doc_name", ""),
            str(chunk.get("page_number", "")),
            chunk.get("content", ""),
        )

    def task7_hybrid_rrf(chunk_lists: list[list[dict]], final_k: int) -> list[dict]:
        scores = {}
        chunks_by_key = {}
        for chunk_list in chunk_lists:
            for rank, chunk in enumerate(chunk_list, start=1):
                key = task7_hybrid_chunk_key(chunk)
                chunks_by_key[key] = chunk
                scores[key] = scores.get(key, 0.0) + 1.0 / (60 + rank)
        ranked_keys = sorted(scores, key=scores.get, reverse=True)
        return [chunks_by_key[key] for key in ranked_keys[:final_k]]

    def task7_hybrid_page_hit(row: dict, chunks: list[dict]) -> int:
        expected_pages = set(evidence_page_numbers(row.get("evidence")))
        retrieved_pages = {
            int(chunk["page_number"])
            for chunk in chunks
            if chunk.get("doc_name") == row.get("doc_name")
            and str(chunk.get("page_number", "")).isdigit()
        }
        return int(bool(expected_pages & retrieved_pages))

    ensure_rank_bm25_installed()
    from langchain_community.retrievers import BM25Retriever

    task7_hybrid_vectorstore_path = (
        ASSIGNMENT2_VECTORSTORE_DIR
        / task7_hybrid_bm25_gemini_k8_experiment["vectorstore_name"]
    )
    task7_hybrid_vectorstore = (
        build_or_load_vectorstore(
            documents_df=referenced_documents_df,
            vectorstore_path=task7_hybrid_vectorstore_path,
            manifest_path=task7_hybrid_vectorstore_path / "manifest.json",
            manifest_payload=task7_hybrid_bm25_manifest(),
            embedding_model=task7_hybrid_bm25_gemini_k8_experiment["embedding_model"],
            chunk_size=task7_hybrid_bm25_gemini_k8_experiment["chunk_size"],
            chunk_overlap=task7_hybrid_bm25_gemini_k8_experiment["chunk_overlap"],
            force_rebuild=FORCE_RERUN,
        )
        if RUN_EXPENSIVE_EVALS
        else None
    )

    task7_hybrid_documents = (
        list(task7_hybrid_vectorstore.docstore._dict.values())
        if RUN_EXPENSIVE_EVALS
        else []
    )
    task7_hybrid_bm25_retriever = (
        BM25Retriever.from_documents(task7_hybrid_documents)
        if task7_hybrid_documents
        else None
    )
    if task7_hybrid_bm25_retriever is not None:
        task7_hybrid_bm25_retriever.k = task7_hybrid_bm25_gemini_k8_experiment["bm25_fetch_k"]

    def task7_hybrid_retrieve(query: str, final_k: int) -> list[dict]:
        semantic_docs = task7_hybrid_vectorstore.similarity_search(
            query,
            k=task7_hybrid_bm25_gemini_k8_experiment["semantic_fetch_k"],
        )
        semantic_chunks = [task7_hybrid_doc_to_chunk(doc) for doc in semantic_docs]
        lexical_docs = (
            task7_hybrid_bm25_retriever.invoke(query)
            if task7_hybrid_bm25_retriever is not None
            else []
        )
        lexical_chunks = [task7_hybrid_doc_to_chunk(doc) for doc in lexical_docs]
        return task7_hybrid_rrf([semantic_chunks, lexical_chunks], final_k)

    task7_hybrid_rows = [
        {
            **row,
            "_faithfulness_eval": row_index < FAITHFULNESS_LIMIT,
        }
        for row_index, row in enumerate(
            financebench_df.sort_values("financebench_id").to_dict("records")
        )
    ]

    def run_task7_hybrid_bm25_gemini_k8_one(row: dict) -> dict:
        query = str(row["question"])
        fused_chunks = task7_hybrid_retrieve(
            query,
            max(
                task7_hybrid_bm25_gemini_k8_experiment["k_for_generation"],
                max(PAGE_HIT_K_VALUES),
            ),
        )
        chunks = fused_chunks[: task7_hybrid_bm25_gemini_k8_experiment["k_for_generation"]]
        rag_result = answer_from_chunks(
            query=query,
            chunks=chunks,
            model=GENERATION_MODEL,
            system_prompt=RAG_SYSTEM_PROMPT_V1,
            include_programmatic_sources=True,
        )
        judged = judge_correctness(
            question=query,
            ground_truth=str(row["answer"]),
            candidate_answer=rag_result["answer"],
        )
        faithfulness = (
            score_faithfulness_with_ragas(
                question=query,
                answer=rag_result["answer"],
                retrieved_contexts=[chunk["content"] for chunk in chunks],
            )
            if row.get("_faithfulness_eval")
            else None
        )
        return {
            "experiment": task7_hybrid_bm25_gemini_k8_experiment["experiment"],
            "question": row["question"],
            "RAG_answer": rag_result["answer"],
            "ground_truth": row["answer"],
            "faithfulness": faithfulness,
            **{
                f"page_hit_at_{_k}": task7_hybrid_page_hit(
                    row,
                    fused_chunks[:_k],
                )
                for _k in PAGE_HIT_K_VALUES
            },
            **judged,
        }

    task7_hybrid_bm25_gemini_k8_attempt_rows_df = run_cached_rows(
        task="task7_hybrid_bm25_gemini_k8_experiment",
        rows=package_rows(task7_hybrid_rows),
        cache_path=TASK7_EXPERIMENT_CACHE_PATH,
        config_payload={
            **task7_hybrid_bm25_gemini_k8_experiment,
            "generator_model": GENERATION_MODEL,
            "judge_model": JUDGE_MODEL,
            "prompt": RAG_SYSTEM_PROMPT_V1,
            "fusion": "reciprocal_rank_fusion",
        },
        input_payload_fn=lambda row: {
            "financebench_id": row["financebench_id"],
            "question": row["question"],
            "answer": row["answer"],
        },
        run_one_fn=run_task7_hybrid_bm25_gemini_k8_one,
        force_rerun=FORCE_RERUN,
        retry_errors=RETRY_ERRORS,
        allow_run=RUN_EXPENSIVE_EVALS,
        max_workers=5,
    )

    task7_hybrid_bm25_gemini_k8_result_row = {
        "experiment": task7_hybrid_bm25_gemini_k8_experiment["experiment"],
        "change": task7_hybrid_bm25_gemini_k8_experiment["change"],
        "correctness": "",
        "faithfulness": "",
        **{f"page_hit_at_{_k}": "" for _k in PAGE_HIT_K_VALUES},
        "hypothesis": task7_hybrid_bm25_gemini_k8_experiment["hypothesis"],
    }
    if not task7_hybrid_bm25_gemini_k8_attempt_rows_df.empty:
        if "correctness" in task7_hybrid_bm25_gemini_k8_attempt_rows_df.columns:
            task7_hybrid_bm25_gemini_k8_result_row["correctness"] = (
                task7_hybrid_bm25_gemini_k8_attempt_rows_df["correctness"].eq("correct").mean()
            )
        if "faithfulness" in task7_hybrid_bm25_gemini_k8_attempt_rows_df.columns:
            task7_hybrid_bm25_gemini_k8_result_row["faithfulness"] = pd.to_numeric(
                task7_hybrid_bm25_gemini_k8_attempt_rows_df["faithfulness"],
                errors="coerce",
            ).mean()
        for _k in PAGE_HIT_K_VALUES:
            _column = f"page_hit_at_{_k}"
            if _column in task7_hybrid_bm25_gemini_k8_attempt_rows_df.columns:
                task7_hybrid_bm25_gemini_k8_result_row[_column] = pd.to_numeric(
                    task7_hybrid_bm25_gemini_k8_attempt_rows_df[_column],
                    errors="coerce",
                ).mean()

    task7_hybrid_bm25_gemini_k8_results_df = pd.DataFrame(
        [task7_hybrid_bm25_gemini_k8_result_row]
    )
    return (task7_hybrid_bm25_gemini_k8_results_df,)


@app.cell
def _(
    ASSIGNMENT2_CACHE_DIR,
    ASSIGNMENT2_VECTORSTORE_DIR,
    BASELINE_CHUNK_OVERLAP,
    BASELINE_CHUNK_SIZE,
    FORCE_RERUN,
    NVIDIA_API_KEY,
    NVIDIA_BASE_URL,
    Path,
    QUERY_ROUTER_MODEL,
    RUN_EXPENSIVE_EVALS,
    build_or_load_vectorstore,
    json,
    pd,
    referenced_documents_df,
    source_fingerprint,
    tqdm,
):
    import ast as _ast
    import os as _os
    import re

    import numpy as np
    from pydantic import BaseModel as _BaseModel
    from pydantic import Field as _Field
    from pydantic import ValidationError as _ValidationError
    from pydantic import field_validator as _field_validator
    from tenacity import retry as _retry
    from tenacity import stop_after_attempt as _stop_after_attempt
    from tenacity import wait_exponential as _wait_exponential

    task7_graph_query_router_max_tokens = int(
        _os.getenv("ASSIGNMENT2_QUERY_ROUTER_MAX_TOKENS", "512")
    )

    task7_graph_vectorstore_config = {
        "embedding_model": "gemini/gemini-embedding-2",
        "vectorstore_name": "faiss_gemini_embedding2_chunk1000",
        "chunk_size": BASELINE_CHUNK_SIZE,
        "chunk_overlap": BASELINE_CHUNK_OVERLAP,
    }

    def task7_graph_vectorstore_manifest() -> dict:
        task7_graph_doc_fingerprints = []
        for task7_graph_doc_row in referenced_documents_df.to_dict("records"):
            task7_graph_doc_fingerprints.append(
                {
                    "doc_name": task7_graph_doc_row["doc_name"],
                    "company": task7_graph_doc_row.get("company", ""),
                    "doc_period": task7_graph_doc_row.get("doc_period", ""),
                    "fingerprint": source_fingerprint(Path(task7_graph_doc_row["pdf_path"])),
                }
            )
        return {
            "embedding_model": task7_graph_vectorstore_config["embedding_model"],
            "chunk_size": task7_graph_vectorstore_config["chunk_size"],
            "chunk_overlap": task7_graph_vectorstore_config["chunk_overlap"],
            "documents": task7_graph_doc_fingerprints,
        }

    task7_graph_vectorstore_path = (
        ASSIGNMENT2_VECTORSTORE_DIR
        / task7_graph_vectorstore_config["vectorstore_name"]
    )
    task7_graph_vectorstore = (
        build_or_load_vectorstore(
            documents_df=referenced_documents_df,
            vectorstore_path=task7_graph_vectorstore_path,
            manifest_path=task7_graph_vectorstore_path / "manifest.json",
            manifest_payload=task7_graph_vectorstore_manifest(),
            embedding_model=task7_graph_vectorstore_config["embedding_model"],
            chunk_size=task7_graph_vectorstore_config["chunk_size"],
            chunk_overlap=task7_graph_vectorstore_config["chunk_overlap"],
            force_rebuild=FORCE_RERUN,
        )
        if RUN_EXPENSIVE_EVALS
        else None
    )

    def task7_graph_document_category(task7_graph_doc_name: str) -> str:
        task7_graph_upper_name = str(task7_graph_doc_name).upper()
        if "10Q" in task7_graph_upper_name or "10-Q" in task7_graph_upper_name:
            return "10Q"
        if "10K" in task7_graph_upper_name or "10-K" in task7_graph_upper_name:
            return "10K"
        if "8K" in task7_graph_upper_name or "8-K" in task7_graph_upper_name:
            return "8K"
        if "EARN" in task7_graph_upper_name:
            return "earnings"
        return ""

    def task7_graph_document_year(
        task7_graph_doc_name: str,
        task7_graph_doc_period: object,
    ) -> str:
        if str(task7_graph_doc_period).strip():
            return str(task7_graph_doc_period).strip()
        task7_graph_year_match = re.search(r"(20\d{2})", str(task7_graph_doc_name))
        return task7_graph_year_match.group(1) if task7_graph_year_match else ""

    def task7_graph_doc_to_chunk(
        task7_graph_doc,
        *,
        task7_graph_chunk_id: str = "",
        task7_graph_faiss_index: int | None = None,
        task7_graph_score: float | None = None,
        task7_graph_rank: int | None = None,
    ) -> dict:
        return {
            "chunk_id": task7_graph_chunk_id,
            "faiss_index": task7_graph_faiss_index,
            "rank": task7_graph_rank,
            "score": task7_graph_score,
            "doc_name": task7_graph_doc.metadata.get("doc_name", ""),
            "page_number": task7_graph_doc.metadata.get("page_number", ""),
            "company": task7_graph_doc.metadata.get("company", ""),
            "doc_period": task7_graph_doc.metadata.get("doc_period", ""),
            "year": task7_graph_document_year(
                task7_graph_doc.metadata.get("doc_name", ""),
                task7_graph_doc.metadata.get("doc_period", ""),
            ),
            "category": task7_graph_document_category(
                task7_graph_doc.metadata.get("doc_name", "")
            ),
            "content": task7_graph_doc.page_content,
        }

    def task7_graph_build_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
        if task7_graph_vectorstore is None:
            return pd.DataFrame(), pd.DataFrame()

        task7_graph_docstore = task7_graph_vectorstore.docstore._dict
        task7_graph_index_map = task7_graph_vectorstore.index_to_docstore_id
        task7_graph_node_rows = []
        for task7_graph_faiss_index, task7_graph_docstore_id in task7_graph_index_map.items():
            task7_graph_doc = task7_graph_docstore[task7_graph_docstore_id]
            task7_graph_doc_name = task7_graph_doc.metadata.get("doc_name", "")
            task7_graph_doc_period = task7_graph_doc.metadata.get("doc_period", "")
            task7_graph_chunk_id = f"chunk_{task7_graph_faiss_index}"
            task7_graph_node_rows.append(
                {
                    "chunk_id": task7_graph_chunk_id,
                    "faiss_index": int(task7_graph_faiss_index),
                    "docstore_id": task7_graph_docstore_id,
                    "doc_name": task7_graph_doc_name,
                    "company": task7_graph_doc.metadata.get("company", ""),
                    "doc_period": task7_graph_doc_period,
                    "year": task7_graph_document_year(
                        task7_graph_doc_name,
                        task7_graph_doc_period,
                    ),
                    "category": task7_graph_document_category(task7_graph_doc_name),
                    "page_number": task7_graph_doc.metadata.get("page_number", ""),
                    "content": task7_graph_doc.page_content,
                }
            )

        task7_graph_nodes_df = pd.DataFrame(task7_graph_node_rows)
        task7_graph_edge_rows = []
        if not task7_graph_nodes_df.empty:
            for task7_graph_group_columns, task7_graph_edge_type in [
                (["doc_name"], "same_document"),
                (["doc_name", "page_number"], "same_page"),
                (["company"], "same_company"),
                (["company", "year"], "same_company_year"),
                (["category"], "same_filing_type"),
            ]:
                task7_graph_grouped = task7_graph_nodes_df.groupby(
                    task7_graph_group_columns,
                    dropna=False,
                )
                for _, task7_graph_group_df in task7_graph_grouped:
                    task7_graph_group_ids = task7_graph_group_df["chunk_id"].tolist()
                    if len(task7_graph_group_ids) < 2:
                        continue
                    for task7_graph_from_id, task7_graph_to_id in zip(
                        task7_graph_group_ids[:-1],
                        task7_graph_group_ids[1:],
                        strict=False,
                    ):
                        task7_graph_edge_rows.append(
                            {
                                "from_id": task7_graph_from_id,
                                "to_id": task7_graph_to_id,
                                "connection_type": task7_graph_edge_type,
                                "strength": 0.5,
                                "metadata": json.dumps(
                                    {
                                        "group_columns": task7_graph_group_columns,
                                    },
                                    ensure_ascii=True,
                                ),
                            }
                        )
        return task7_graph_nodes_df, pd.DataFrame(task7_graph_edge_rows)

    task7_graph_chunk_nodes_df, task7_graph_chunk_edges_df = task7_graph_build_tables()

    task7_graph_document_ontology_cache_path = (
        ASSIGNMENT2_CACHE_DIR / "task7_graph_document_ontology.jsonl"
    )

    def task7_graph_json_from_text(task7_graph_text: str) -> dict:
        task7_graph_stripped_text = task7_graph_text.strip()
        task7_graph_fenced_match = re.search(
            r"```(?:json)?\s*(.*?)```",
            task7_graph_stripped_text,
            re.S,
        )
        if task7_graph_fenced_match:
            task7_graph_stripped_text = task7_graph_fenced_match.group(1).strip()
        task7_graph_object_match = re.search(r"\{.*\}", task7_graph_stripped_text, re.S)
        if task7_graph_object_match:
            task7_graph_stripped_text = task7_graph_object_match.group(0)
        try:
            return json.loads(task7_graph_stripped_text)
        except json.JSONDecodeError:
            try:
                return _ast.literal_eval(task7_graph_stripped_text)
            except (SyntaxError, ValueError):
                task7_graph_loose_result = task7_graph_loose_metadata_from_text(
                    task7_graph_text
                )
                if task7_graph_loose_result:
                    return task7_graph_loose_result
                raise

    def task7_graph_loose_metadata_from_text(task7_graph_text: str) -> dict:
        task7_graph_loose_result = {}
        task7_graph_category_match = re.search(
            r"\bcategory\b\s*[:=]\s*[\"'`]?([A-Za-z0-9_-]+)",
            task7_graph_text,
            re.I,
        )
        if task7_graph_category_match:
            task7_graph_loose_result["category"] = task7_graph_category_match.group(1)
        task7_graph_year_match = re.search(
            r"\byear\b\s*[:=]\s*[\"'`]?(20\d{2})",
            task7_graph_text,
            re.I,
        )
        if task7_graph_year_match:
            task7_graph_loose_result["year"] = task7_graph_year_match.group(1)
        task7_graph_company_match = re.search(
            r"\bcompany\b\s*[:=]\s*[\"'`]?([^,\n}]+)",
            task7_graph_text,
            re.I,
        )
        if task7_graph_company_match:
            task7_graph_loose_result["company"] = task7_graph_company_match.group(1).strip(" \"'`")
        task7_graph_topics_match = re.search(
            r"\b(?:primary_topics|metric_terms)\b\s*[:=]\s*\[([^\]]*)\]",
            task7_graph_text,
            re.I | re.S,
        )
        if task7_graph_topics_match:
            task7_graph_loose_result["primary_topics"] = [
                task7_graph_topic.strip(" \"'`\n\t")
                for task7_graph_topic in task7_graph_topics_match.group(1).split(",")
                if task7_graph_topic.strip(" \"'`\n\t")
            ]
            task7_graph_loose_result["metric_terms"] = task7_graph_loose_result[
                "primary_topics"
            ]
        return task7_graph_loose_result

    def task7_graph_call_and_parse(
        *,
        task7_graph_messages: list[dict],
        task7_graph_model_cls,
        task7_graph_extra_payload: dict | None = None,
    ):
        import litellm

        @_retry(
            stop=_stop_after_attempt(3),
            wait=_wait_exponential(multiplier=1, min=1, max=8),
            reraise=True,
        )
        def task7_graph_call_once():
            task7_graph_response = litellm.completion(
                model=f"nvidia_nim/{QUERY_ROUTER_MODEL}",
                api_key=NVIDIA_API_KEY,
                api_base=NVIDIA_BASE_URL,
                messages=task7_graph_messages,
                temperature=0.0,
                max_tokens=task7_graph_query_router_max_tokens,
                timeout=60,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            task7_graph_raw_content = task7_graph_response.choices[0].message.content
            task7_graph_parsed_payload = task7_graph_json_from_text(
                task7_graph_raw_content
            ) or task7_graph_loose_metadata_from_text(task7_graph_raw_content)
            if task7_graph_extra_payload:
                task7_graph_parsed_payload = {
                    **task7_graph_parsed_payload,
                    **task7_graph_extra_payload,
                }
            return task7_graph_model_cls.model_validate(task7_graph_parsed_payload)

        return task7_graph_call_once()

    class Task7GraphDocumentOntology(_BaseModel):
        doc_name: str
        category: str = ""
        year: str = ""
        primary_topics: list[str] = _Field(default_factory=list)

        @_field_validator("year", mode="before")
        @classmethod
        def coerce_year(cls, task7_graph_year_value):
            return "" if task7_graph_year_value is None else str(task7_graph_year_value)

    def task7_graph_load_document_ontology_cache() -> dict[str, dict]:
        if not task7_graph_document_ontology_cache_path.exists():
            return {}
        task7_graph_cache_rows = {}
        for task7_graph_cache_line in task7_graph_document_ontology_cache_path.read_text(
            encoding="utf-8"
        ).splitlines():
            if not task7_graph_cache_line.strip():
                continue
            try:
                task7_graph_cache_row = json.loads(task7_graph_cache_line)
            except json.JSONDecodeError:
                continue
            task7_graph_doc_name = str(task7_graph_cache_row.get("doc_name", ""))
            if task7_graph_doc_name:
                task7_graph_cache_rows[task7_graph_doc_name] = task7_graph_cache_row
        return task7_graph_cache_rows

    def task7_graph_fallback_document_ontology(
        task7_graph_doc_name: str,
        task7_graph_doc_df: pd.DataFrame,
    ) -> dict:
        task7_graph_doc_period = (
            str(task7_graph_doc_df["doc_period"].dropna().iloc[0])
            if not task7_graph_doc_df["doc_period"].dropna().empty
            else ""
        )
        task7_graph_topics = sorted(
            {
                task7_graph_topic_match.group(0).lower()
                for task7_graph_content in task7_graph_doc_df["content"].head(8).astype(str)
                for task7_graph_topic_match in re.finditer(
                    r"\b(cash|revenue|sales|margin|ebitda|ebitdar|assets|liabilities|debt|inventory|working capital|capex|dividend|income|earnings)\b",
                    task7_graph_content,
                    re.I,
                )
            }
        )[:12]
        return {
            "doc_name": task7_graph_doc_name,
            "category": task7_graph_document_category(task7_graph_doc_name),
            "year": task7_graph_document_year(
                task7_graph_doc_name,
                task7_graph_doc_period,
            ),
            "primary_topics": task7_graph_topics,
        }

    def task7_graph_infer_document_ontology(
        task7_graph_doc_name: str,
        task7_graph_doc_df: pd.DataFrame,
    ) -> dict:
        task7_graph_fallback = task7_graph_fallback_document_ontology(
            task7_graph_doc_name,
            task7_graph_doc_df,
        )
        if not NVIDIA_API_KEY:
            return task7_graph_fallback
        try:
            task7_graph_doc_sample = "\n\n".join(
                task7_graph_doc_df["content"].head(3).astype(str).tolist()
            )[:1800]
            task7_graph_company = (
                str(task7_graph_doc_df["company"].dropna().iloc[0])
                if not task7_graph_doc_df["company"].dropna().empty
                else ""
            )
            task7_graph_llm_result = task7_graph_call_and_parse(
                task7_graph_messages=[
                    {
                        "role": "system",
                        "content": (
                            "Classify this financial filing for retrieval metadata. "
                            "Return JSON with doc_name, category, year, primary_topics. "
                            "category must be one of 10K, 10Q, 8K, earnings, or empty string. "
                            "primary_topics should be short finance terms found or strongly implied."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"doc_name: {task7_graph_doc_name}\n"
                            f"company: {task7_graph_company}\n"
                            f"sample:\n{task7_graph_doc_sample}"
                        ),
                    },
                ],
                task7_graph_model_cls=Task7GraphDocumentOntology,
                task7_graph_extra_payload={"doc_name": task7_graph_doc_name},
            ).model_dump()
        except Exception as task7_graph_ontology_exc:
            print(
                "[task7_graph_ontology] Falling back to deterministic ontology: "
                f"{repr(task7_graph_ontology_exc)[:200]}"
            )
            return task7_graph_fallback

        for task7_graph_ontology_key in ["category", "year"]:
            if not task7_graph_llm_result.get(task7_graph_ontology_key):
                task7_graph_llm_result[task7_graph_ontology_key] = task7_graph_fallback.get(
                    task7_graph_ontology_key,
                    "",
                )
        task7_graph_llm_result["primary_topics"] = sorted(
            set(task7_graph_llm_result.get("primary_topics", []))
            | set(task7_graph_fallback.get("primary_topics", []))
        )[:16]
        return task7_graph_llm_result

    def task7_graph_build_document_ontology_table() -> pd.DataFrame:
        if task7_graph_chunk_nodes_df.empty:
            return pd.DataFrame()
        task7_graph_cached_ontology = task7_graph_load_document_ontology_cache()
        task7_graph_ontology_rows = []
        task7_graph_doc_groups = list(task7_graph_chunk_nodes_df.groupby(
            "doc_name",
            dropna=False,
        ))
        for task7_graph_doc_name, task7_graph_doc_df in tqdm(
            task7_graph_doc_groups,
            desc="[task7_graph_ontology]",
            unit="doc",
        ):
            task7_graph_doc_name = str(task7_graph_doc_name)
            task7_graph_ontology_row = task7_graph_cached_ontology.get(task7_graph_doc_name)
            if task7_graph_ontology_row is None and RUN_EXPENSIVE_EVALS:
                task7_graph_ontology_row = task7_graph_infer_document_ontology(
                    task7_graph_doc_name,
                    task7_graph_doc_df,
                )
                with task7_graph_document_ontology_cache_path.open(
                    "a",
                    encoding="utf-8",
                ) as task7_graph_cache_file:
                    task7_graph_cache_file.write(
                        json.dumps(task7_graph_ontology_row, ensure_ascii=True) + "\n"
                    )
            if task7_graph_ontology_row is None:
                task7_graph_ontology_row = task7_graph_fallback_document_ontology(
                    task7_graph_doc_name,
                    task7_graph_doc_df,
                )
            task7_graph_ontology_rows.append(task7_graph_ontology_row)
        return pd.DataFrame(task7_graph_ontology_rows)

    task7_graph_document_ontology_df = task7_graph_build_document_ontology_table()
    if not task7_graph_document_ontology_df.empty and not task7_graph_chunk_nodes_df.empty:
        task7_graph_ontology_lookup = task7_graph_document_ontology_df.set_index(
            "doc_name"
        )
        task7_graph_chunk_nodes_df["category"] = task7_graph_chunk_nodes_df[
            "doc_name"
        ].map(task7_graph_ontology_lookup["category"]).fillna(
            task7_graph_chunk_nodes_df["category"]
        )
        task7_graph_chunk_nodes_df["year"] = task7_graph_chunk_nodes_df["doc_name"].map(
            task7_graph_ontology_lookup["year"]
        ).fillna(task7_graph_chunk_nodes_df["year"])

    class Task7GraphQueryRoute(_BaseModel):
        company: str = ""
        year: str = ""
        category: str = ""
        metric_terms: list[str] = _Field(default_factory=list)
        confidence: float = 0.0

        @_field_validator("year", mode="before")
        @classmethod
        def coerce_year(cls, task7_graph_year_value):
            return "" if task7_graph_year_value is None else str(task7_graph_year_value)

    def task7_graph_regex_route(task7_graph_question: str) -> Task7GraphQueryRoute:
        task7_graph_question_lower = task7_graph_question.lower()
        task7_graph_companies = sorted(
            {
                str(task7_graph_company)
                for task7_graph_company in referenced_documents_df["company"].dropna().unique()
                if str(task7_graph_company).strip()
            },
            key=len,
            reverse=True,
        )
        task7_graph_company = ""
        for task7_graph_candidate_company in task7_graph_companies:
            if task7_graph_candidate_company.lower() in task7_graph_question_lower:
                task7_graph_company = task7_graph_candidate_company
                break
        task7_graph_year_match = re.search(r"(?:FY|fiscal year|year)?\s*(20\d{2})", task7_graph_question, re.I)
        task7_graph_category = ""
        if re.search(r"\bQ[1-4]\b|quarter", task7_graph_question, re.I):
            task7_graph_category = "10Q"
        elif re.search(r"\bFY\b|fiscal year|annual|10-k|10k", task7_graph_question, re.I):
            task7_graph_category = "10K"
        task7_graph_metric_terms = [
            task7_graph_match.group(0).lower()
            for task7_graph_match in re.finditer(
                r"\b(cash|revenue|sales|margin|ebitda|ebitdar|assets|liabilities|debt|inventory|working capital|capex|dividend|income|earnings)\b",
                task7_graph_question,
                re.I,
            )
        ]
        return Task7GraphQueryRoute(
            company=task7_graph_company,
            year=task7_graph_year_match.group(1) if task7_graph_year_match else "",
            category=task7_graph_category,
            metric_terms=sorted(set(task7_graph_metric_terms)),
            confidence=0.45 if task7_graph_company or task7_graph_year_match else 0.0,
        )

    def task7_graph_route_query(task7_graph_question: str) -> dict:
        task7_graph_regex_result = task7_graph_regex_route(task7_graph_question)
        if not NVIDIA_API_KEY:
            return task7_graph_regex_result.model_dump()
        try:
            task7_graph_llm_result = task7_graph_call_and_parse(
                task7_graph_messages=[
                    {
                        "role": "system",
                        "content": (
                            "Extract retrieval filters from the finance question. "
                            "Use only information stated in the question. "
                            "Return JSON with company, year, category, metric_terms, confidence. "
                            "category must be one of 10K, 10Q, 8K, earnings, or empty string."
                        ),
                    },
                    {"role": "user", "content": task7_graph_question},
                ],
                task7_graph_model_cls=Task7GraphQueryRoute,
            )
        except (_ValidationError, Exception) as task7_graph_route_exc:
            print(f"[task7_graph_route] Falling back to regex route: {repr(task7_graph_route_exc)[:200]}")
            return task7_graph_regex_result.model_dump()

        task7_graph_merged = task7_graph_llm_result.model_dump()
        task7_graph_regex_dump = task7_graph_regex_result.model_dump()
        for task7_graph_route_key in ["company", "year", "category"]:
            if not task7_graph_merged.get(task7_graph_route_key):
                task7_graph_merged[task7_graph_route_key] = task7_graph_regex_dump.get(
                    task7_graph_route_key,
                    "",
                )
        task7_graph_metric_terms = sorted(
            set(task7_graph_merged.get("metric_terms", []))
            | set(task7_graph_regex_dump.get("metric_terms", []))
        )
        task7_graph_merged["metric_terms"] = task7_graph_metric_terms
        return task7_graph_merged

    def task7_graph_filter_indices(
        task7_graph_route: dict,
        *,
        task7_graph_min_candidates: int = 40,
    ) -> set[int]:
        if task7_graph_chunk_nodes_df.empty:
            return set()
        task7_graph_candidate_df = task7_graph_chunk_nodes_df
        task7_graph_filters = [
            ("company", task7_graph_route.get("company", "")),
            ("year", task7_graph_route.get("year", "")),
            ("category", task7_graph_route.get("category", "")),
        ]
        for task7_graph_filter_count in range(len(task7_graph_filters), -1, -1):
            task7_graph_filtered_df = task7_graph_candidate_df
            for task7_graph_filter_column, task7_graph_filter_value in task7_graph_filters[
                :task7_graph_filter_count
            ]:
                if str(task7_graph_filter_value).strip():
                    task7_graph_filtered_df = task7_graph_filtered_df[
                        task7_graph_filtered_df[task7_graph_filter_column]
                        .astype(str)
                        .str.lower()
                        == str(task7_graph_filter_value).lower()
                    ]
            if len(task7_graph_filtered_df) >= task7_graph_min_candidates:
                return set(task7_graph_filtered_df["faiss_index"].astype(int).tolist())
        return set(task7_graph_chunk_nodes_df["faiss_index"].astype(int).tolist())

    def task7_graph_semantic_rank(
        task7_graph_query: str,
        *,
        task7_graph_candidate_indices: set[int] | None = None,
        task7_graph_fetch_k: int = 20,
    ) -> list[dict]:
        if task7_graph_vectorstore is None:
            return []
        if not task7_graph_candidate_indices:
            task7_graph_docs = task7_graph_vectorstore.similarity_search(
                task7_graph_query,
                k=task7_graph_fetch_k,
            )
            return [
                task7_graph_doc_to_chunk(
                    task7_graph_doc,
                    task7_graph_rank=task7_graph_doc_rank,
                )
                for task7_graph_doc_rank, task7_graph_doc in enumerate(
                    task7_graph_docs,
                    start=1,
                )
            ]

        task7_graph_query_vector = np.array(
            task7_graph_vectorstore.embedding_function.embed_query(task7_graph_query),
            dtype=np.float32,
        )
        task7_graph_scored_indices = []
        for task7_graph_candidate_index in task7_graph_candidate_indices:
            task7_graph_chunk_vector = np.array(
                task7_graph_vectorstore.index.reconstruct(int(task7_graph_candidate_index)),
                dtype=np.float32,
            )
            task7_graph_distance = float(
                np.linalg.norm(task7_graph_query_vector - task7_graph_chunk_vector)
            )
            task7_graph_scored_indices.append(
                (task7_graph_distance, int(task7_graph_candidate_index))
            )
        task7_graph_scored_indices.sort(key=lambda task7_graph_item: task7_graph_item[0])

        task7_graph_docstore = task7_graph_vectorstore.docstore._dict
        task7_graph_ranked_chunks = []
        for task7_graph_rank, (
            task7_graph_distance,
            task7_graph_faiss_index,
        ) in enumerate(task7_graph_scored_indices[:task7_graph_fetch_k], start=1):
            task7_graph_docstore_id = task7_graph_vectorstore.index_to_docstore_id[
                task7_graph_faiss_index
            ]
            task7_graph_doc = task7_graph_docstore[task7_graph_docstore_id]
            task7_graph_ranked_chunks.append(
                task7_graph_doc_to_chunk(
                    task7_graph_doc,
                    task7_graph_chunk_id=f"chunk_{task7_graph_faiss_index}",
                    task7_graph_faiss_index=task7_graph_faiss_index,
                    task7_graph_score=-task7_graph_distance,
                    task7_graph_rank=task7_graph_rank,
                )
            )
        return task7_graph_ranked_chunks

    def task7_graph_expand_neighbors(
        task7_graph_seed_chunks: list[dict],
        *,
        task7_graph_final_k: int,
        task7_graph_page_radius: int = 1,
    ) -> list[dict]:
        if task7_graph_chunk_nodes_df.empty:
            return task7_graph_seed_chunks[:task7_graph_final_k]
        task7_graph_scored = {}
        for task7_graph_seed_rank, task7_graph_seed_chunk in enumerate(
            task7_graph_seed_chunks,
            start=1,
        ):
            task7_graph_doc_name = task7_graph_seed_chunk.get("doc_name", "")
            try:
                task7_graph_page = int(task7_graph_seed_chunk.get("page_number", -10000))
            except (TypeError, ValueError):
                task7_graph_page = -10000
            task7_graph_neighbor_df = task7_graph_chunk_nodes_df[
                (task7_graph_chunk_nodes_df["doc_name"] == task7_graph_doc_name)
                & (
                    task7_graph_chunk_nodes_df["page_number"]
                    .astype(str)
                    .str.fullmatch(r"\d+")
                )
            ].copy()
            if task7_graph_neighbor_df.empty:
                continue
            task7_graph_neighbor_df["page_int"] = task7_graph_neighbor_df[
                "page_number"
            ].astype(int)
            task7_graph_neighbor_df = task7_graph_neighbor_df[
                (
                    task7_graph_neighbor_df["page_int"]
                    .sub(task7_graph_page)
                    .abs()
                    <= task7_graph_page_radius
                )
            ]
            task7_graph_base_score = 1.0 / (task7_graph_seed_rank + 1)
            for task7_graph_neighbor_row in task7_graph_neighbor_df.to_dict("records"):
                task7_graph_neighbor_index = int(task7_graph_neighbor_row["faiss_index"])
                task7_graph_page_delta = abs(
                    int(task7_graph_neighbor_row["page_int"]) - task7_graph_page
                )
                task7_graph_neighbor_score = task7_graph_base_score * (
                    1.0 if task7_graph_page_delta == 0 else 0.55
                )
                task7_graph_existing = task7_graph_scored.get(task7_graph_neighbor_index)
                if (
                    task7_graph_existing is None
                    or task7_graph_neighbor_score > task7_graph_existing[0]
                ):
                    task7_graph_scored[task7_graph_neighbor_index] = (
                        task7_graph_neighbor_score,
                        task7_graph_neighbor_row,
                    )
        task7_graph_ranked_neighbors = sorted(
            task7_graph_scored.values(),
            key=lambda task7_graph_item: task7_graph_item[0],
            reverse=True,
        )
        return [
            {
                "chunk_id": task7_graph_neighbor_row["chunk_id"],
                "faiss_index": int(task7_graph_neighbor_row["faiss_index"]),
                "rank": task7_graph_neighbor_rank,
                "score": task7_graph_neighbor_score,
                "doc_name": task7_graph_neighbor_row["doc_name"],
                "page_number": task7_graph_neighbor_row["page_number"],
                "company": task7_graph_neighbor_row["company"],
                "doc_period": task7_graph_neighbor_row["doc_period"],
                "year": task7_graph_neighbor_row["year"],
                "category": task7_graph_neighbor_row["category"],
                "content": task7_graph_neighbor_row["content"],
            }
            for task7_graph_neighbor_rank, (
                task7_graph_neighbor_score,
                task7_graph_neighbor_row,
            ) in enumerate(task7_graph_ranked_neighbors[:task7_graph_final_k], start=1)
        ]

    def task7_graph_cluster_boost(
        task7_graph_query: str,
        task7_graph_route: dict,
        *,
        task7_graph_final_k: int,
    ) -> list[dict]:
        task7_graph_candidates = task7_graph_filter_indices(
            task7_graph_route,
            task7_graph_min_candidates=40,
        )
        task7_graph_seed_chunks = task7_graph_semantic_rank(
            task7_graph_query,
            task7_graph_candidate_indices=task7_graph_candidates,
            task7_graph_fetch_k=30,
        )
        return task7_graph_expand_neighbors(
            task7_graph_seed_chunks,
            task7_graph_final_k=task7_graph_final_k,
            task7_graph_page_radius=1,
        )

    def task7_graph_soft_metadata_boost(
        task7_graph_seed_chunks: list[dict],
        task7_graph_route: dict,
        *,
        task7_graph_final_k: int,
    ) -> list[dict]:
        task7_graph_boosted_chunks = []
        for task7_graph_rank, task7_graph_chunk in enumerate(
            task7_graph_seed_chunks,
            start=1,
        ):
            task7_graph_boost = 0.0
            if (
                str(task7_graph_route.get("company", "")).strip()
                and str(task7_graph_chunk.get("company", "")).lower()
                == str(task7_graph_route.get("company", "")).lower()
            ):
                task7_graph_boost += 0.20
            if (
                str(task7_graph_route.get("year", "")).strip()
                and str(task7_graph_chunk.get("year", "")).lower()
                == str(task7_graph_route.get("year", "")).lower()
            ):
                task7_graph_boost += 0.10
            if (
                str(task7_graph_route.get("category", "")).strip()
                and str(task7_graph_chunk.get("category", "")).lower()
                == str(task7_graph_route.get("category", "")).lower()
            ):
                task7_graph_boost += 0.08
            task7_graph_boosted_chunks.append(
                {
                    **task7_graph_chunk,
                    "rank": task7_graph_rank,
                    "score": (1.0 / (task7_graph_rank + 1)) + task7_graph_boost,
                }
            )
        return sorted(
            task7_graph_boosted_chunks,
            key=lambda task7_graph_chunk: task7_graph_chunk.get("score", 0.0),
            reverse=True,
        )[:task7_graph_final_k]

    return (
        task7_graph_cluster_boost,
        task7_graph_expand_neighbors,
        task7_graph_filter_indices,
        task7_graph_route_query,
        task7_graph_semantic_rank,
        task7_graph_soft_metadata_boost,
        task7_graph_vectorstore_config,
    )


@app.cell
def _(
    FAITHFULNESS_LIMIT,
    FORCE_RERUN,
    GENERATION_MODEL,
    JUDGE_MODEL,
    PAGE_HIT_K_VALUES,
    RAG_SYSTEM_PROMPT_V1,
    RETRY_ERRORS,
    RUN_EXPENSIVE_EVALS,
    TASK7_EXPERIMENT_CACHE_PATH,
    answer_from_chunks,
    evidence_page_numbers,
    financebench_df,
    judge_correctness,
    package_rows,
    pd,
    run_cached_rows,
    score_faithfulness_with_ragas,
    task7_graph_cluster_boost,
    task7_graph_expand_neighbors,
    task7_graph_filter_indices,
    task7_graph_route_query,
    task7_graph_semantic_rank,
    task7_graph_soft_metadata_boost,
    task7_graph_vectorstore_config,
):
    task7_graph_experiments = [
        {
            "experiment": "metadata_filter_gemini_2_k8",
            "change": "Infer company/year/filing type from the query, filter Gemini candidates, then generate with k=8",
            "hypothesis": "Query-derived metadata filters should reduce wrong-company and wrong-filing retrieval without using benchmark evidence.",
            "retrieval_mode": "metadata_filter",
            "k_for_generation": 8,
            "semantic_fetch_k": 30,
        },
        {
            "experiment": "neighbor_context_gemini_2_k8",
            "change": "Retrieve Gemini top-20, expand same-document page neighbors, then generate with k=8",
            "hypothesis": "Neighbor expansion should recover nearby table context when the closest chunk lands on an adjacent page or split boundary.",
            "retrieval_mode": "neighbor_context",
            "k_for_generation": 8,
            "semantic_fetch_k": 20,
        },
        {
            "experiment": "ontology_cluster_boost_gemini_2_k8",
            "change": "Combine query metadata filters with same-document/page-neighbor cluster boosting, then generate with k=8",
            "hypothesis": "Combining metadata filtering and local chunk connections should improve evidence coverage more than either signal alone.",
            "retrieval_mode": "metadata_neighbor_boost",
            "k_for_generation": 8,
            "semantic_fetch_k": 30,
        },
        {
            "experiment": "soft_metadata_boost_gemini_2_k8",
            "change": "Retrieve Gemini top-30, softly boost query-matching metadata, then generate with k=8",
            "hypothesis": "Soft metadata boosts should preserve Gemini semantic recall while nudging same-company, same-year, and same-filing chunks upward.",
            "retrieval_mode": "soft_metadata_boost",
            "k_for_generation": 8,
            "semantic_fetch_k": 30,
        },
    ]

    def task7_graph_page_hit(row: dict, task7_graph_chunks: list[dict]) -> int:
        task7_graph_expected_pages = set(evidence_page_numbers(row.get("evidence")))
        task7_graph_retrieved_pages = {
            int(task7_graph_chunk["page_number"])
            for task7_graph_chunk in task7_graph_chunks
            if task7_graph_chunk.get("doc_name") == row.get("doc_name")
            and str(task7_graph_chunk.get("page_number", "")).isdigit()
        }
        return int(bool(task7_graph_expected_pages & task7_graph_retrieved_pages))

    def task7_graph_retrieve(
        task7_graph_experiment: dict,
        task7_graph_query: str,
        *,
        task7_graph_fetch_k: int,
    ) -> list[dict]:
        task7_graph_route = task7_graph_route_query(task7_graph_query)
        if task7_graph_experiment["retrieval_mode"] == "metadata_filter":
            task7_graph_candidate_indices = task7_graph_filter_indices(
                task7_graph_route,
                task7_graph_min_candidates=40,
            )
            return task7_graph_semantic_rank(
                task7_graph_query,
                task7_graph_candidate_indices=task7_graph_candidate_indices,
                task7_graph_fetch_k=task7_graph_fetch_k,
            )
        if task7_graph_experiment["retrieval_mode"] == "neighbor_context":
            task7_graph_seed_chunks = task7_graph_semantic_rank(
                task7_graph_query,
                task7_graph_candidate_indices=None,
                task7_graph_fetch_k=task7_graph_experiment["semantic_fetch_k"],
            )
            return task7_graph_expand_neighbors(
                task7_graph_seed_chunks,
                task7_graph_final_k=task7_graph_fetch_k,
                task7_graph_page_radius=1,
            )
        if task7_graph_experiment["retrieval_mode"] == "soft_metadata_boost":
            task7_graph_seed_chunks = task7_graph_semantic_rank(
                task7_graph_query,
                task7_graph_candidate_indices=None,
                task7_graph_fetch_k=task7_graph_experiment["semantic_fetch_k"],
            )
            return task7_graph_soft_metadata_boost(
                task7_graph_seed_chunks,
                task7_graph_route,
                task7_graph_final_k=task7_graph_fetch_k,
            )
        return task7_graph_cluster_boost(
            task7_graph_query,
            task7_graph_route,
            task7_graph_final_k=task7_graph_fetch_k,
        )

    task7_graph_rows = [
        {
            **task7_graph_row,
            "_faithfulness_eval": task7_graph_row_index < FAITHFULNESS_LIMIT,
        }
        for task7_graph_row_index, task7_graph_row in enumerate(
            financebench_df.sort_values("financebench_id").to_dict("records")
        )
    ]

    def task7_graph_run_experiment(task7_graph_experiment: dict) -> pd.DataFrame:
        def run_task7_graph_one(task7_graph_row: dict) -> dict:
            task7_graph_query = str(task7_graph_row["question"])
            task7_graph_retrieved_chunks = task7_graph_retrieve(
                task7_graph_experiment,
                task7_graph_query,
                task7_graph_fetch_k=max(
                    task7_graph_experiment["k_for_generation"],
                    max(PAGE_HIT_K_VALUES),
                ),
            )
            task7_graph_generation_chunks = task7_graph_retrieved_chunks[
                : task7_graph_experiment["k_for_generation"]
            ]
            task7_graph_rag_result = answer_from_chunks(
                query=task7_graph_query,
                chunks=task7_graph_generation_chunks,
                model=GENERATION_MODEL,
                system_prompt=RAG_SYSTEM_PROMPT_V1,
                include_programmatic_sources=True,
            )
            task7_graph_judged = judge_correctness(
                question=task7_graph_query,
                ground_truth=str(task7_graph_row["answer"]),
                candidate_answer=task7_graph_rag_result["answer"],
            )
            task7_graph_faithfulness = (
                score_faithfulness_with_ragas(
                    question=task7_graph_query,
                    answer=task7_graph_rag_result["answer"],
                    retrieved_contexts=[
                        task7_graph_chunk["content"]
                        for task7_graph_chunk in task7_graph_generation_chunks
                    ],
                )
                if task7_graph_row.get("_faithfulness_eval")
                else None
            )
            return {
                "experiment": task7_graph_experiment["experiment"],
                "question": task7_graph_row["question"],
                "RAG_answer": task7_graph_rag_result["answer"],
                "ground_truth": task7_graph_row["answer"],
                "faithfulness": task7_graph_faithfulness,
                **{
                    f"page_hit_at_{task7_graph_k}": task7_graph_page_hit(
                        task7_graph_row,
                        task7_graph_retrieved_chunks[:task7_graph_k],
                    )
                    for task7_graph_k in PAGE_HIT_K_VALUES
                },
                **task7_graph_judged,
            }

        task7_graph_attempt_rows_df = run_cached_rows(
            task=f"task7_{task7_graph_experiment['experiment']}_experiment",
            rows=package_rows(task7_graph_rows),
            cache_path=TASK7_EXPERIMENT_CACHE_PATH,
            config_payload={
                **task7_graph_experiment,
                **task7_graph_vectorstore_config,
                "generator_model": GENERATION_MODEL,
                "judge_model": JUDGE_MODEL,
                "prompt": RAG_SYSTEM_PROMPT_V1,
                "max_workers": 5,
            },
            input_payload_fn=lambda task7_graph_row: {
                "financebench_id": task7_graph_row["financebench_id"],
                "question": task7_graph_row["question"],
                "answer": task7_graph_row["answer"],
            },
            run_one_fn=run_task7_graph_one,
            force_rerun=FORCE_RERUN,
            retry_errors=RETRY_ERRORS,
            allow_run=RUN_EXPENSIVE_EVALS,
            max_workers=5,
        )

        task7_graph_result_row = {
            "experiment": task7_graph_experiment["experiment"],
            "change": task7_graph_experiment["change"],
            "correctness": "",
            "faithfulness": "",
            **{f"page_hit_at_{task7_graph_k}": "" for task7_graph_k in PAGE_HIT_K_VALUES},
            "hypothesis": task7_graph_experiment["hypothesis"],
        }
        if not task7_graph_attempt_rows_df.empty:
            if "correctness" in task7_graph_attempt_rows_df.columns:
                task7_graph_result_row["correctness"] = (
                    task7_graph_attempt_rows_df["correctness"].eq("correct").mean()
                )
            if "faithfulness" in task7_graph_attempt_rows_df.columns:
                task7_graph_result_row["faithfulness"] = pd.to_numeric(
                    task7_graph_attempt_rows_df["faithfulness"],
                    errors="coerce",
                ).mean()
            for task7_graph_k in PAGE_HIT_K_VALUES:
                task7_graph_column = f"page_hit_at_{task7_graph_k}"
                if task7_graph_column in task7_graph_attempt_rows_df.columns:
                    task7_graph_result_row[task7_graph_column] = pd.to_numeric(
                        task7_graph_attempt_rows_df[task7_graph_column],
                        errors="coerce",
                    ).mean()
        return pd.DataFrame([task7_graph_result_row])

    task7_metadata_filter_gemini_k8_results_df = task7_graph_run_experiment(
        task7_graph_experiments[0]
    )
    task7_neighbor_context_gemini_k8_results_df = task7_graph_run_experiment(
        task7_graph_experiments[1]
    )
    task7_ontology_cluster_boost_gemini_k8_results_df = task7_graph_run_experiment(
        task7_graph_experiments[2]
    )
    task7_soft_metadata_boost_gemini_k8_results_df = task7_graph_run_experiment(
        task7_graph_experiments[3]
    )
    return (
        task7_metadata_filter_gemini_k8_results_df,
        task7_neighbor_context_gemini_k8_results_df,
        task7_ontology_cluster_boost_gemini_k8_results_df,
        task7_soft_metadata_boost_gemini_k8_results_df,
    )


@app.cell
def _(
    pd,
    task7_embedding_k8_results_df,
    task7_embedding_results_df,
    task7_hybrid_bm25_gemini_k8_results_df,
    task7_metadata_filter_gemini_k8_results_df,
    task7_neighbor_context_gemini_k8_results_df,
    task7_ontology_cluster_boost_gemini_k8_results_df,
    task7_results_df,
    task7_soft_metadata_boost_gemini_k8_results_df,
):
    task7_results_combined_df = pd.concat(
        [
            task7_results_df,
            task7_embedding_results_df,
            task7_embedding_k8_results_df,
            task7_hybrid_bm25_gemini_k8_results_df,
            task7_metadata_filter_gemini_k8_results_df,
            task7_neighbor_context_gemini_k8_results_df,
            task7_ontology_cluster_boost_gemini_k8_results_df,
            task7_soft_metadata_boost_gemini_k8_results_df,
        ],
        ignore_index=True,
    )
    return (task7_results_combined_df,)


@app.cell
def _(PAGE_HIT_K_VALUES, round_metric_columns, task7_results_combined_df):
    task7_results_display_df = round_metric_columns(
        task7_results_combined_df,
        [
            "correctness",
            "faithfulness",
            *[f"page_hit_at_{_page_hit_k}" for _page_hit_k in PAGE_HIT_K_VALUES],
        ],
    )
    return (task7_results_display_df,)


@app.cell
def _(
    ASSIGNMENT2_IMPROVEMENT_BASE,
    OUTPUT_FORMAT,
    PAGE_HIT_K_VALUES,
    export_table,
    task7_results_display_df,
):
    task7_export_paths = export_table(
        task7_results_display_df,
        ASSIGNMENT2_IMPROVEMENT_BASE,
        [
            "experiment",
            "change",
            "correctness",
            "faithfulness",
            *[f"page_hit_at_{_page_hit_k}" for _page_hit_k in PAGE_HIT_K_VALUES],
        ],
        OUTPUT_FORMAT,
    )
    return (task7_export_paths,)


@app.cell
def _(display_table, mo, task7_results_display_df):
    task7_results_output = display_table(task7_results_display_df, mo)
    task7_results_output
    return


@app.cell
def _(display_table, mo, selected_improvement_config, task7_attempt_rows_df):
    if task7_attempt_rows_df.empty or "experiment" not in task7_attempt_rows_df:
        selected_improvement_rows_df = task7_attempt_rows_df
    else:
        selected_improvement_rows_df = task7_attempt_rows_df[
            task7_attempt_rows_df["experiment"]
            == selected_improvement_config.experiment
        ]
    selected_improvement_output = mo.vstack(
        [
            mo.md(
                f"### Cycle - {selected_improvement_config.experiment}: {selected_improvement_config.change}"
            ),
            display_table(selected_improvement_rows_df, mo),
        ]
    )
    selected_improvement_output
    return


@app.cell
def _(mo, pd, task7_results_display_df):
    try:
        import matplotlib.pyplot as plt

        metric_columns = [
            column
            for column in ["correctness", "faithfulness", "page_hit_at_5"]
            if column in task7_results_display_df.columns
        ]
        if (
            task7_results_display_df.empty
            or "experiment" not in task7_results_display_df.columns
            or not metric_columns
        ):
            task7_visualization_output = mo.md("No Task 7 metrics available to plot.")
        else:
            _plot_df = task7_results_display_df[["experiment", *metric_columns]].copy()
            for _column in metric_columns:
                _plot_df[_column] = pd.to_numeric(_plot_df[_column], errors="coerce")
            _ax = _plot_df.set_index("experiment")[metric_columns].plot(kind="bar")
            _ax.set_ylim(0, 1)
            _ax.set_ylabel("score")
            _ax.set_title("Task 7 Improvement Cycle Metrics")
            _ax.legend(loc="best")
            task7_visualization_output = mo.mpl.interactive(_ax.figure)
    except Exception as task7_plot_exc:
        task7_visualization_output = mo.md(
            f"Task 7 visualization skipped: `{repr(task7_plot_exc)}`"
        )
    task7_visualization_output
    return


@app.cell
def _(mo, pd, task7_results_display_df):
    def make_task7_heatmap_output(task7_results_display_df, mo, pd):
        import matplotlib.pyplot as plt

        metric_columns = [
            column
            for column in [
                "correctness",
                "faithfulness",
                "page_hit_at_1",
                "page_hit_at_3",
                "page_hit_at_5",
            ]
            if column in task7_results_display_df.columns
        ]

        if (
            task7_results_display_df.empty
            or "experiment" not in task7_results_display_df.columns
            or not metric_columns
        ):
            return mo.md("No Task 7 metrics available to plot.")

        heatmap_df = task7_results_display_df[["experiment", *metric_columns]].copy()

        for column in metric_columns:
            heatmap_df[column] = pd.to_numeric(
                heatmap_df[column],
                errors="coerce",
            )

        heatmap_df = heatmap_df.set_index("experiment")[metric_columns]

        fig, ax = plt.subplots(figsize=(8, 3.5))
        image = ax.imshow(
            heatmap_df,
            aspect="auto",
            vmin=0,
            vmax=1,
            cmap="Blues",
        )

        ax.set_xticks(range(len(metric_columns)))
        ax.set_xticklabels(metric_columns, rotation=30, ha="right")
        ax.set_yticks(range(len(heatmap_df.index)))
        ax.set_yticklabels(heatmap_df.index)

        for row_index, experiment in enumerate(heatmap_df.index):
            for col_index, metric in enumerate(metric_columns):
                value = heatmap_df.loc[experiment, metric]
                if pd.notna(value):
                    ax.text(
                        col_index,
                        row_index,
                        f"{value:.3f}",
                        ha="center",
                        va="center",
                        color="black",
                    )

        ax.set_title("Task 7 Improvement Cycle Metrics")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()

        return mo.mpl.interactive(fig)


    task7_heatmap_output = make_task7_heatmap_output(
        task7_results_display_df=task7_results_display_df,
        mo=mo,
        pd=pd,
    )

    task7_heatmap_output
    return


@app.cell(hide_code=True)
def _(mo, task7_results_display_df):
    if task7_results_display_df.empty:
        task7_experiment_lines = "No Task 7 experiment results are available yet."
    else:
        task7_experiment_lines = "\n".join(
            (
                f"- `{row['experiment']}` hypothesis: {row.get('hypothesis', '')}"
            )
            for row in task7_results_display_df.to_dict("records")
        )
    task7_qa_output = mo.md(f"""
    ### Answers - Task 7

    	Q: What is the hypothesis and interpretation for each experiment?

    	A: Hypotheses:
    {task7_experiment_lines}

    	Interpretations:
    - `prompt_v2`: Almost no change, made faithfulness worse. Quite possibly could be tuned more, but I don't deem this to be the top spot to improve. 
    - `k_8`: Surprisingly good results (Okay, not so surprising given we're just increasing the probability correct chunks entering the context. ); would definitely increase input tokens and increase cost. I haven't tracked it in this assignment - just forgot to be honest. 
    - `chunk_500`: Worse results quite possibly because chunk size is not the determinant here. It is already quite small to capture the document part meaning. 
    - `reranker`: Quite peculiar, better result with a lower hit rate. 
    - `embedding_gemini_2`: As expected, it produced a meaningful improvement in both hit rate and subsequent faithfulness + correctness. 
    - `embedding_gemini_2_k8`: Again as expected those improvements proved additive, so they're worth combining. 
    - `hybrid_bm25_gemini_2_k8`: Worse results likely because of equally weighting both approaches → Full text search is likely too noisy for it as it doesn't have any metadata or any ontology. Definitely asking for a neurosymbolic approach. 

    Q: Where does the pipeline fail most — retrieval, generation, or both? What would you try with one more week?

    A: The current pipeline fails most at retrieval. The generator is actually quite conservative, uh which is a surprise given it's not a big model. I would try several approaches:
    - Possibly adding a kind of a metadata or frontmatter summary, per document and per document section ideally. 
    - Maybe I might put all the documents in a graph so they are connected and we are searching by meaning + hierarchyq as well.
    """)
    task7_qa_output
    return


@app.cell
def _(task7_attempt_rows_df):
    baseline_vs_chunk500 = (
        task7_attempt_rows_df[
            task7_attempt_rows_df["experiment"].isin(["prompt_v2", "chunk_500"])
        ]
        .pivot(index="financebench_id", columns="experiment", values="correctness")
        .dropna()
    )

    disagreement_rate = (
        baseline_vs_chunk500["prompt_v2"]
        != baseline_vs_chunk500["chunk_500"]
    ).mean()

    disagreement_rate
    return (disagreement_rate,)


@app.cell(hide_code=True)
def _(disagreement_rate, mo):
    bonus_output = mo.md(f"""
    ## Bonus - Multi-Scale Chunking

    Optional extension:
    - build FAISS indices for chunk sizes such as 300, 1000, and 2000;
    - keep embedding model, splitter type, and overlap policy fixed;
    - compare page-hit@5 per question and summarize whether the best chunk size is query-dependent.

    Q: If you did the bonus, what does the summary table show? What is the disagreement rate and what does it mean?

    A: I actually did chunk-500 sizing. Disagreement rate is {disagreement_rate}. Given lower chunks are hurting both hit rate and correctness - I'd reiterate again that likely they're just losing on big context table or information chunks. I'm not sure overlap would be helpful here, as we could just increase the chunk size?
    """)
    bonus_output
    return


@app.cell(hide_code=True)
def _(
    mo,
    task1_export_paths,
    task5_export_paths,
    task6_export_paths,
    task7_export_paths,
):
    all_export_paths = [
        *task1_export_paths,
        *task5_export_paths,
        *task6_export_paths,
        *task7_export_paths,
    ]
    export_path_lines = "\n".join(f"- `{path}`" for path in all_export_paths)
    deliverables_output = mo.md(
        f"""
        ## Deliverables

        Generated table paths:

    {export_path_lines}

        Final export command:

    ```bash
    uv run marimo export ipynb w2-ai-product/ai-product-2.py -o w2-ai-product/ai-product-2.ipynb --include-outputs -f
    ```
        """
    )
    deliverables_output
    return


if __name__ == "__main__":
    app.run()
