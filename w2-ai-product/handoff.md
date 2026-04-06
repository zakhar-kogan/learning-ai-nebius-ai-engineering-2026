# Handoff Note — LLM Evaluation Assignment

The repository structure and implementation scaffold are generated according to `implementation_plan.md` (v4): all 6 notebooks exist, the shared logic in `src/` is implemented, and the system prompts are created.

This is **not** a finished submission yet. The actual execution, manual grading, prompt iteration, and export steps still need to be run by you.

## 0. Anything Left to Complete? (Yes — the execution phase)

Here is exactly what you need to do to finish the assignment:

### Step 1: Add your API Keys
Copy `.env.example` to `.env` inside `w2-ai-product/` and populate your API keys:
```env
NEBIUS_API_KEY=your_key_here
NVIDIA_API_KEY=your_key_here  # Optional, if you want to use it for dev
```

### Step 2: Run Task 1, 2, and 3
Run the notebooks in your browser using `marimo`:
```bash
marimo edit notebooks/01_rubric.py
marimo edit notebooks/02_generate.py
marimo edit notebooks/03_human_eval.py
```
**CRITICAL PAUSE:** In `03_human_eval.py`, after the first cell runs, it will save programmatic ratings to `outputs/assignment_01.xlsx`.
- You MUST stop here, open `outputs/assignment_01.xlsx` in Excel/Sheets, and **manually score 10-15 products** for the 5 criteria (`good`, `ok`, or `bad`).
- After scoring, save the Excel file and run the rest of `03_human_eval.py` to generate your baseline analysis.
- Fill in the Markdown cell at the bottom of the notebook summarizing your findings.

### Step 3: Iterate on Task 4 (The Improvement Cycle)
```bash
marimo edit notebooks/04_improve.py
```
- Based on your manual baseline, try to beat it. I've populated two sample experiments (larger model, and 2-shot prompting).
- You can tweak the hyper-parameters, prompt versions, or models.
- If an experiment sucks, keep its log in the markdown table at the top, but delete its code block. Keep only the successful code.
- You can compare your MLflow traces via: `mlflow ui --backend-store-uri sqlite:///outputs/experiments.db`

### Step 4: Run the Judge (Task 5 & 6)
```bash
marimo edit notebooks/05_judge.py
marimo edit notebooks/06_analysis.py
```
- In Task 5, verify the 5 "Sanity Check" outputs. If the Judge is behaving weirdly, edit the `prompts/judge_all.txt` file until it behaves correctly. Then run the full dataset.
- In Task 6, run the final analysis to compare your human scores against the Judge's scores. 
- You still need to write the final essay in the markdown section of `06_analysis.py` analyzing the trade-offs of Human vs. LLM judging.

### Step 5: Export and Submit
Once all notebooks run properly, export them to HTML for your submission:
```bash
marimo export html notebooks/01_rubric.py -o outputs/html/task_01.html
marimo export html notebooks/02_generate.py -o outputs/html/task_02.html
marimo export html notebooks/03_human_eval.py -o outputs/html/task_03.html
marimo export html notebooks/04_improve.py -o outputs/html/task_04.html
marimo export html notebooks/05_judge.py -o outputs/html/task_05.html
marimo export html notebooks/06_analysis.py -o outputs/html/task_06.html
```
Submit the HTML files (or just submit the `.py` marimo notebooks directly, depending on the grader's exact preference) alongside the final `outputs/assignment_01.xlsx` file.

## 1. Context File Created
I also generated `agents.md` in the `w2-ai-product` directory, which gives any future AI assistant the architectural rules of this exact assignment (such as why `explanation` comes before `verdict` in the Pydantic schema).
