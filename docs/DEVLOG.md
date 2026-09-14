# HopSense — Development Log

Multi-hop retrieval + reasoning system. This log tracks what was built,
why, and what problems drove each next step.

---

## Milestone 1: Project Setup (Sept 12, 2026)

**Goal:** Get the skeleton working. Prove OpenAI + LangChain + FAISS + 
HotpotQA all talk to each other.

- Setup: [`src/load_data.py`](../src/load_data.py) downloads 100 HotpotQA questions
- Output: [`outputs/01_load_data_output.txt`](outputs/01_load_data_output.txt)
- Result: ✅ HotpotQA loaded, sample question confirms multi-hop nature

---

## Milestone 2: Question Decomposition

**Goal:** Break complex questions into atomic sub-questions.

- Code: [`src/decomposer.py`](../src/decomposer.py)
- Approach: GPT-4o-mini with Pydantic structured output. Decides 
  multi-hop vs single-hop, produces 2-4 sub-questions.
- Output: [`outputs/02_decomposer_output.txt`](outputs/02_decomposer_output.txt)
- Result: ✅ Correctly identifies single-hop ("What year was Einstein 
  born?") vs multi-hop questions. Sub-questions are clean and atomic.

---

## Milestone 3: Dense Retrieval (FAISS + sentence-transformers)

**Goal:** Given a sub-question, return top-K relevant passages.

- Code: [`src/retriever.py`](../src/retriever.py)
- Stack: sentence-transformers (all-MiniLM-L6-v2, 384-dim) + FAISS IndexFlatIP
- Output: [`outputs/03_retriever_output.txt`](outputs/03_retriever_output.txt)

### Problem discovered
**Naive retrieval misses one of the two entities in multi-hop questions.**

For "Were Scott Derrickson and Ed Wood of the same nationality?":
- Rank 1: Ed Wood (film) — about the movie, not the person
- Rank 2: Ed Wood — the person ✅
- Rank 3: Adam Collis — unrelated filmmaker
- **Scott Derrickson's page: NOT in top 3.**

The retriever anchors on whichever name is more common in the corpus 
and under-retrieves the other. Details: 
[`problems/01_naive_rag_misses_entity.md`](problems/01_naive_rag_misses_entity.md)

### Why this drives the next step
Confirms the core motivation for HopSense — retrieve per sub-question 
instead of retrieving once on the full question.

---

## Milestone 4: Full Pipeline (Naive vs HopSense) ✅

**Goal:** Compare naive RAG against decompose → retrieve-per-hop → synthesize.

- Code: [`src/pipeline.py`](../src/pipeline.py)
- Output: [`outputs/04_pipeline_naive_vs_hopsense.txt`](outputs/04_pipeline_naive_vs_hopsense.txt)

### Result on first test question

Q: "Were Scott Derrickson and Ed Wood of the same nationality?"
Gold: yes

| System | Answer | Sources | Confidence |
|---|---|---|---|
| Naive RAG | "Yes." | (none surfaced) | (none reported) |
| HopSense | "Yes, both Scott Derrickson and Ed Wood were American." | Scott Derrickson, Ed Wood pages | 1.00 overall |

### Key observation
Both correct on this easy case. The difference is **auditability**:
HopSense exposes which passage supported each hop, per-hop confidence,
and overall confidence via weakest-link propagation. Naive RAG gives
a bare answer with no way to interrogate why.

### What this doesn't prove yet
1 correct question ≠ system works. Need to run 100+ questions to
measure accuracy delta, retrieval recall delta, and confidence
calibration. That's Milestone 5.
---
<!-- ## Next milestones
- [ ] Run on 100 HotpotQA questions, log accuracy
- [ ] Build EvalArena metrics: recall@k, faithfulness, hallucination
- [ ] Add contradiction detection between hops
- [ ] Swap OpenAI → Llama 3 8B via Ollama, compare -->
---

## Milestone 5: Batch Evaluation on 100 HotpotQA Questions ✅

**Goal:** Move beyond single-question demos. Get real, statistically meaningful numbers on naive RAG vs HopSense across 100 questions.

- Code: [`src/batch_eval.py`](../src/batch_eval.py), [`src/analyze_results.py`](../src/analyze_results.py)
- Data: 100 HotpotQA validation examples (from [`data/hotpot_sample.json`](../data/hotpot_sample.json))
- Raw results: [`results/batch_results_100q.json`](../results/batch_results_100q.json)
- Outputs: 
  - [`outputs/06_batch_eval_100q.txt`](outputs/06_batch_eval_100q.txt) — batch run log
  - [`outputs/07_analysis_100q.txt`](outputs/07_analysis_100q.txt) — analysis breakdown

### Headline result

| | Exact Match | Contains Gold | Avg Time |
|---|---|---|---|
| Naive RAG | 38% | 40% | 1.1 s |
| **HopSense** | **51%** | **58%** | 4.8 s |

**+13 EM, +18 contains-gold at ~4× latency.** 96/100 questions correctly identified as multi-hop.

### Head-to-head breakdown

| Category | Count | % |
|---|---|---|
| Both correct | 34 | 34% |
| HopSense only wins | 24 | 24% |
| Naive only wins | 6 | 6% |
| Both wrong | 36 | 36% |

HopSense wins 4× more often than naive. But 36% both-wrong shows the ceiling — some HotpotQA questions are hard for reasons neither system addresses (weak decomposition, retrieval gaps, or knowledge missing from the distractor context).

### Three problems surfaced by this milestone

Full details in `docs/problems/`:

1. **[07.1 — Bridge vs comparison split](problems/07.1_hopsense_helps_bridge_hurts_comparison.md)**
   HopSense: bridge +24 pp, comparison -5 pp. Decomposition helps chained retrieval, hurts when both entities are already in the query.

2. **[07.2 — Calibration broken](problems/07.2_calibration_broken.md)**
   72% of answers labelled "confidence ≥ 0.90" — but 32% of those are wrong. LLM self-reported confidence is unreliable and overconfident. Most dangerous failure mode.

3. **[07.3 — Over-decomposition and dependent sub-questions](problems/07.3_over_decomposition_and_dependent_subq.md)**
   Decomposer sometimes adds a wrapper sub-question that duplicates the original, or produces sub-questions that reference each other ("the actress identified in the first question") — untenable when each sub-question is retrieved in isolation.

### Prioritisation for next milestones

| Fix | Effort | Impact | Order |
|---|---|---|---|
| 07.3 stricter decomposer prompt | 20 min | Cleans baseline for other measurements | **Next** |
| 07.2 self-consistency confidence | 2 hr | Fixes most dangerous failure mode | After 07.3 |
| 07.1 route comparisons to naive | 30 min | Small overall gain (+5 pp on 21% of questions) | Later |

### What this doesn't yet include

- No F1 (token-overlap) metric — HotpotQA's standard metric. Adding this next.
- No retrieval-quality metrics (recall@k, MRR) — that's EvalArena.
- No per-hop latency/cost breakdown.
- No baseline vs open-weight LLM comparison (deferred to Phase 3).

---

<!-- ## Next milestones

- [ ] **M6:** Fix 07.3 — stricter decomposer prompt, rerun 100q, measure delta
- [ ] **M7:** Fix 07.2 — self-consistency confidence, measure calibration curve  
- [ ] **M8:** EvalArena Phase 1 — F1, recall@k, MRR, faithfulness metrics
- [ ] **M9:** EvalArena Phase 2 — LLM-as-judge for hallucination
- [ ] **M10:** Swap OpenAI → Llama 3 8B via Ollama, benchmark tradeoff -->\

---

---

## Milestone 6: Attempted fix for problem 07.3 (stricter decomposer prompt) — ROLLED BACK

**Hypothesis:** A stricter decomposer prompt (naming failure modes 
explicitly, forbidding back-references and wrappers) will fix over-
decomposition and dependent-sub-question errors.

**Change:** Updated `DECOMPOSER_PROMPT` in `src/decomposer.py` with 
explicit rules against wrapper sub-questions and back-references.

**Result:** HopSense contains-gold dropped from 58% to 54%. Failure 
cases from M5 (Bobbi Bacha, Alfred Balk) still fail. The model still 
produces back-referencing sub-questions verbatim.

**Outputs:**
- [`outputs/08_batch_eval_100q_after_m6.txt`](outputs/08_batch_eval_100q_after_m6.txt)
- [`outputs/09_analysis_100q_after_m6.txt`](outputs/09_analysis_100q_after_m6.txt)
- Raw JSON: [`results/batch_results_100q_after_m6.json`](../results/batch_results_100q_after_m6.json)

**Decision:** Rolled back the prompt change. Real fix is chained 
execution (deferred to a later milestone). Full analysis appended to 
[`problems/07.3_over_decomposition_and_dependent_subq.md`](problems/07.3_over_decomposition_and_dependent_subq.md).

**Takeaway:** Prompt engineering hits diminishing returns quickly on 
this class of failure. Some multi-hop errors look architectural — 
independent decomposition can't cleanly handle genuinely sequential 
questions.

