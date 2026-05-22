# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

This project is a Hybrid RAG reproduction project. Follow both the general coding guidelines and the project-specific experiment rules below.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

---

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

```text
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

# Project-Specific Instructions: Hybrid RAG Replication

## 5. Core Experiment Rules

This project compares RAG variants under controlled conditions.

Target methods:
- Vector RAG
- Graph RAG
- Hybrid RAG
- Vector + Graph RAG baseline, only if explicitly implemented

Important:
- Hybrid RAG is not simple concatenation of vector results and graph results.
- Hybrid RAG must use vector similarity to select graph start nodes, then perform graph-based traversal or expansion.
- Do not silently change dataset, query set, chunk size, top_k, traversal depth, embedding model, LLM, temperature, or evaluator settings.
- If any experiment setting changes, record it.

Every result should be traceable to:
- retrieved context
- chunk ID
- graph node or edge ID, if applicable
- source document metadata

## 6. Graph Construction Rules

Graph construction must be evidence-based and traceable.

Each chunk should preserve:
- `doc_id`
- `chunk_id`
- `source_path`
- `page`, if available
- `section_path`, if available
- `text`
- `token_count`

Each extracted triple should preserve:
- `subject`
- `relation`
- `object`
- `source_chunk_id`
- `evidence_text`
- `confidence`, if available

Rules:
- Do not create triples without evidence text.
- Do not invent entities or relations not supported by the source chunk.
- Do not merge similar entities unless normalization rules are explicitly defined.
- Before full graph construction, inspect sample chunks and sample triples.
- If chunks contain empty text, broken encoding, or missing critical metadata, stop and report the issue.

## 7. Evaluation Rules

Evaluation must be reproducible and comparable.

Before reporting performance:
- Use the same evaluation dataset for all compared methods.
- Use the same question set, reference answers, and reference contexts when applicable.
- Use temperature=0 unless generation variability is being tested.
- Save raw predictions, retrieved contexts, experiment settings, and metric summaries.

Recommended metrics:
- `context_precision`
- `context_recall`
- `faithfulness`
- `answer_relevancy`
- `answer_correctness`
- `answer_similarity`

If a metric returns `NaN`:
- Do not treat it as zero.
- Do not ignore it.
- Check for empty retrieved contexts, missing references, or wrong evaluator input format.
- Record the cause before drawing conclusions.

Do not compare methods if one method failed to retrieve valid contexts unless the failure itself is the result being analyzed.

## 8. Commands

Primary local environment:

```text
Windows PowerShell
```

Activate virtual environment:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run all tests:

```powershell
python -m pytest -q
```

Run a specific test file:

```powershell
python -m pytest test_graph_rag.py -q
```

Important:
- `collected 0 items` is not a successful test.
- If pytest collects 0 items, check test file names, test function names, and pytest discovery rules.
- Before full graph construction, run a small smoke test first.

---

## 9. What Not To Do

- Do not introduce new baselines without approval.
- Do not change experiment settings silently.
- Do not delete existing experiment outputs unless asked.
- Do not report only successful metrics while hiding failed runs.
- Do not invent graph relations without evidence text.
- Do not optimize for speed before verifying correctness.
- Do not refactor unrelated files during experiment debugging.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer silent experiment-setting changes, clearer graph construction, and more reproducible evaluation results.