# Assignment working notes

This file captures the current executed state of the assignment so the final write-up can cite concrete numbers instead of open prompts.

## Current execution status
- Task 1: completed
- Task 2: completed
- Task 3: completed
- Task 4: completed
- Task 5: completed
- Task 6: completed

## Deliverables currently present
- Workbook: `outputs/assignment_01.xlsx`
- MLflow DB: `outputs/experiments.db`
- Task 4 workbook: `outputs/task_04_experiments.xlsx`
- Task 5 sanity CSV: `outputs/task_05_judge_sanity.csv`
- Task 6 judged CSV: `outputs/task_06_judged_experiments.csv`
- HTML exports: `outputs/html/task_01.html` through `outputs/html/task_06.html`

## Baseline setup used in Task 2
- Provider: `nebius`
- Model: `meta-llama/Meta-Llama-3.1-8B-Instruct`
- Prompt version: `generation_v1.txt`
- Temperature: `0.3`
- Max tokens: `200`
- Dataset size at runtime: `50` products

## Baseline generation observations from Task 2
- Mean latency: `4040 ms`
- Total cost: `$0.000517`
- Workbook written to: `outputs/assignment_01.xlsx`
- MLflow DB written to: `outputs/experiments.db`

## Manual evaluation sample from Task 3
- Rows manually scored: `16`
- Pass count: `9`
- Fail count: `7`
- Baseline pass rate on scored sample: `56%` (`9 / 16`)

### Criterion counts on the scored sample
- Fluency: `13 good`, `3 ok`, `0 bad`
- Grammar: `15 good`, `1 ok`, `0 bad`
- Tone: `15 good`, `0 ok`, `1 bad`
- Length: `16 good`, `0 ok`, `0 bad`
- Grounding: `0 good`, `12 ok`, `4 bad`
- Latency: `0 good`, `12 ok`, `4 bad`
- Cost: `16 good`, `0 ok`, `0 bad`

### Ranked by good-rate on the scored sample
1. Length — `100% good`
2. Tone — `94% good`
3. Grammar — `94% good`
4. Fluency — `81% good`
5. Grounding — `0% good`

## Baseline interpretation
- The baseline is already strong on grammar, tone, and length.
- Cost is comfortably within the `good` bucket for all baseline rows.
- The main quality risk is grounding, not fluency or grammar.
- Some baseline failures are also driven by latency falling into the `bad` bucket.
- Task 4 therefore needed to prioritize groundedness while preserving the already-strong style metrics.

## Concrete examples worth citing later
- `Apple iPhone 15 Pro` — failed grounding despite otherwise strong writing; useful example of polished but insufficiently grounded copy.
- `Stanley Quencher H2.0 40 oz` — useful hallucination example because the description mentions a battery even though the product is a tumbler.
- `Google Pixel 8 Pro` — passed; good example of a baseline description staying close to the source data.
- `Sony WH-1000XM5 Headphones` — passed; useful example of acceptable baseline quality.

## Task 4 experiment setup
- Experiment 1: `exp1_llama8b_prompt_v3_voice`
  What changed: voice rewrite with stronger anti-list and anti-hallucination instructions.
- Experiment 2: `exp2_qwen30b_prompt_v3_voice`
  What changed: stronger model, same improved voice prompt.
- Experiment 3: `exp3_llama8b_prompt_v4_scaffold`
  What changed: hidden narrative scaffold to improve flow.
- Experiment 4: `exp4_llama8b_prompt_v4_best_of_2`
  What changed: two candidates plus LLM selection.

### Task 4 prompt-example decisions
- Positive examples injected from `POSITIVE_EXAMPLES`: `Hydro Flask 32 oz Wide Mouth`, `Apple iPhone 15 Pro`
- Active contrastive bad example in the base prompt: `Stanley Quencher H2.0 40 oz`
- Separate observed failure example for analysis/logging: `PlayStation 5 Slim`

## Task 4 manual comparison results

### Pass rate on the common 16-row manual sample
- Baseline: `56.25%`
- Exp 1: `75.00%`
- Exp 2: `50.00%`
- Exp 3: `81.25%`
- Exp 4: `31.25%`

### Manual style and grounding takeaways
- Best fluency improvement: `Exp 3` with `16/16 good` fluency.
- Best tone result: `Exp 1`, `Exp 2`, `Exp 3`, and `Exp 4` all reached `16/16 good` tone.
- Best combined tone + fluency result: `Exp 3`.
- Best grounding mean on the scored sample: `Exp 4` at `0.4375`, but still `0 good` and operationally too expensive/slow.
- Worst grounding regression: `Exp 2`, with `8 bad` grounding ratings out of `16`.

### Operational metrics across all 50 rows
- Baseline: mean latency `4039.6 ms`, total cost `$0.000517`
- Exp 1: mean latency `3200.6 ms`, total cost `$0.001074`
- Exp 2: mean latency `1044.3 ms`, total cost `$0.001110`
- Exp 3: mean latency `3052.3 ms`, total cost `$0.001144`
- Exp 4: mean latency `5802.7 ms`, total cost `$0.002711`

### Operational rubric takeaways
- `Exp 2` is the fastest option by a large margin: all `50/50` rows are `good` on latency.
- `Exp 4` is operationally poor: `37/50` rows are `bad` on latency and its total cost is highest.
- All experiments stayed in the `good` cost bucket per row under the rubric, so latency was the real operational differentiator.

### Task 4 conclusion to reuse in the final report
- `Exp 3` is the strongest quality result overall because it produced the best manual pass rate and the strongest combined tone + fluency performance.
- `Exp 1` is the simplest credible improvement over baseline if the report wants to emphasize a minimal prompt-only change.
- `Exp 2` looks attractive operationally, but the human grounding scores are worse than baseline, so it should not be presented as the best qualitative improvement.
- `Exp 4` is not worth the added complexity because the selector pipeline does not justify its latency overhead.

## Task 5 judge notes
- The 5-row sanity check completed and was saved to `outputs/task_05_judge_sanity.csv`.
- No prompt rewrite was needed after the current sanity run.
- The judge setup is strongest on structure and consistency, but Task 6 shows that criterion quality differs sharply by category.

## Task 6 human-vs-judge results

### Criterion agreement on human-labeled rows
- Fluency: all-at-once `91.03%`, per-criterion `92.31%`
- Grammar: all-at-once `98.72%`, per-criterion `98.72%`
- Tone: all-at-once `98.72%`, per-criterion `97.44%`
- Length: all-at-once `61.54%`, per-criterion `75.64%`
- Grounding: all-at-once `1.28%`, per-criterion `3.85%`

### Judge vs human pass rates by experiment
- Baseline: human `56.25%`, all-at-once judge `83.72%`, per-criterion judge `84.09%`
- Exp 1: human `75.00%`, all-at-once judge `88.64%`, per-criterion judge `92.00%`
- Exp 2: human `50.00%`, all-at-once judge `97.56%`, per-criterion judge `97.96%`
- Exp 3: human `81.25%`, all-at-once judge `95.74%`, per-criterion judge `93.75%`
- Exp 4: human `31.25%`, all-at-once judge `24.49%`, per-criterion judge `26.53%`

### Task 6 analysis takeaways
- The judge agrees most with the human on grammar and tone, and it is also strong on fluency.
- The judge is materially weaker on length, though per-criterion judging helps there.
- Grounding is the clear failure mode for automated judging in this repo: agreement remains near zero even after switching to per-criterion judging.
- The judge is generally more lenient than the human on Baseline, Exp 1, Exp 2, and Exp 3.
- `Exp 4` is the one case where both human and judge agree that the experiment performs poorly overall.

### How to use the agreement table in the final analysis
- Use it to justify where LLM-as-a-judge is trustworthy: grammar, tone, fluency.
- Use it to explain where human calibration is still necessary: especially grounding, and to a lesser extent length.
- Use the delta between all-at-once and per-criterion agreement to show that decomposing the task helps somewhat for length, but does not solve grounding.

## Suggested final-report narrative
1. Define the rubric and make pass/fail explicit.
2. Show that the baseline already performs well on grammar, tone, and length, but not on grounding.
3. Compare prompt/model interventions against the same manually scored sample.
4. Present `Exp 3` as the strongest quality improvement, with `Exp 1` as the simpler prompt-only alternative.
5. Show that LLM-as-a-judge scales well and is highly consistent on style-oriented criteria.
6. Conclude that grounding still benefits from human calibration because the automated judge is far less aligned there.

## MLflow notes
Use this exact command from inside `w2-ai-product/`:

```bash
mlflow ui --backend-store-uri sqlite:///outputs/experiments.db
```

If you run plain `mlflow ui` without that backend URI, MLflow will usually open the default local store instead, which is why you may only see an empty `Default` experiment.
