"""
HopSense pipeline: decompose -> retrieve per sub-question -> synthesize.

Naive baseline: single retrieval on original question.
HopSense:       decomposed multi-hop retrieval + synthesis.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from decomposer import decompose, DecomposedQuestion
from retriever import Retriever, Passage, load_passages_from_hotpot

load_dotenv()


class HopAnswer(BaseModel):
    """Answer for a single sub-question."""
    sub_question: str
    answer: str
    supporting_titles: List[str] = Field(description="Which passage titles supported this")
    confidence: float = Field(description="0-1 confidence based on evidence quality")


class HopSenseResult(BaseModel):
    """Final result of HopSense pipeline."""
    original_question: str
    is_multi_hop: bool
    sub_questions: List[str]
    hop_answers: List[HopAnswer]
    final_answer: str
    overall_confidence: float


# ============ Sub-question answering ============
SUBQ_PROMPT = """Answer the sub-question using ONLY the passages below.
If the passages don't contain the answer, say "UNKNOWN".

Passages:
{passages}

Sub-question: {sub_question}

Return:
- answer: short factual answer (or "UNKNOWN")
- supporting_titles: list of passage titles that support your answer
- confidence: 0.0 to 1.0 based on how clearly the passages answer this"""


def answer_sub_question(sub_q: str, passages: List[Passage]) -> HopAnswer:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_structured = llm.with_structured_output(HopAnswer)
    
    passages_text = "\n\n".join(
        f"[{p.title}]\n{p.text}" for p in passages
    )
    
    prompt = ChatPromptTemplate.from_template(SUBQ_PROMPT)
    chain = prompt | llm_structured
    
    result = chain.invoke({
        "passages": passages_text,
        "sub_question": sub_q,
    })
    # Overwrite in case the LLM changed sub_question wording
    result.sub_question = sub_q
    return result


# ============ Final synthesis ============
SYNTHESIS_PROMPT = """You are given an original question and answers to its sub-questions.
Combine the sub-answers into a final answer.

Original question: {original_question}

Sub-question answers:
{hop_summary}

Rules for final_answer:
- Give ONLY the atomic answer, nothing more.
- Yes/no questions: answer "yes" or "no" (lowercase, no period).
- Entity questions: just the entity name (e.g. "American", not "They are American").
- Number/date questions: just the number or date.
- Do NOT explain, restate the question, or add "Both are..." style prefixes.

Also provide:
- overall_confidence: 0.0 to 1.0, considering all hop confidences
  (a chain is only as strong as its weakest link)"""


# Replaced for problem 2 with above prompt, which is more explicit about the final answer format.
# SYNTHESIS_PROMPT = """You are given an original question and answers to its sub-questions.
# Combine the sub-answers into a final answer to the original question.

# Original question: {original_question}

# Sub-question answers:
# {hop_summary}

# Provide:
# - final_answer: your answer to the original question (short)
# - overall_confidence: 0.0 to 1.0, considering the confidence of each hop 
#   (a chain is only as strong as its weakest link)"""


class Synthesis(BaseModel):
    final_answer: str
    overall_confidence: float


def synthesize(original_q: str, hop_answers: List[HopAnswer]) -> Synthesis:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_structured = llm.with_structured_output(Synthesis)
    
    hop_summary = "\n".join(
        f"- Q: {h.sub_question}\n  A: {h.answer} (confidence: {h.confidence:.2f})"
        for h in hop_answers
    )
    
    prompt = ChatPromptTemplate.from_template(SYNTHESIS_PROMPT)
    chain = prompt | llm_structured
    return chain.invoke({
        "original_question": original_q,
        "hop_summary": hop_summary,
    })


# ============ Full pipeline ============
def run_hopsense(question: str, retriever: Retriever, k: int = 3) -> HopSenseResult:
    """Run the full HopSense pipeline on one question."""
    
    # Step 1: Decompose
    decomp = decompose(question)
    
    # Step 2: For each sub-question, retrieve and answer
    if decomp.is_multi_hop:
        sub_qs = decomp.sub_questions
    else:
        sub_qs = [question]  # single-hop: just use original
    
    hop_answers = []
    for sq in sub_qs:
        passages = retriever.retrieve(sq, k=k)
        hop_ans = answer_sub_question(sq, passages)
        hop_answers.append(hop_ans)
    
    # Step 3: Synthesize
    synth = synthesize(question, hop_answers)
    
    return HopSenseResult(
        original_question=question,
        is_multi_hop=decomp.is_multi_hop,
        sub_questions=sub_qs,
        hop_answers=hop_answers,
        final_answer=synth.final_answer,
        overall_confidence=synth.overall_confidence,
    )


# ============ Naive baseline ============
NAIVE_PROMPT = """Answer the question using ONLY the passages below.
If the passages don't contain the answer, say "UNKNOWN".

Passages:
{passages}

Question: {question}

Give ONLY the atomic answer:
- Yes/no questions: "yes" or "no" (lowercase).
- Entity/date/number: just the entity, date, or number.
- No explanation, no restating the question."""

# replaced for problem 2 with above prompt, which is more explicit about the final answer format.
# NAIVE_PROMPT = """Answer the question using ONLY the passages below.
# If the passages don't contain the answer, say "UNKNOWN".

# Passages:
# {passages}

# Question: {question}

# Give a short, direct answer."""


def run_naive_rag(question: str, retriever: Retriever, k: int = 3) -> str:
    """Naive single-retrieval RAG (baseline)."""
    passages = retriever.retrieve(question, k=k)
    passages_text = "\n\n".join(f"[{p.title}]\n{p.text}" for p in passages)
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_template(NAIVE_PROMPT)
    chain = prompt | llm
    
    resp = chain.invoke({"passages": passages_text, "question": question})
    return resp.content.strip()


# ============ TEST ============
if __name__ == "__main__":
    import json
    
    with open("data/hotpot_sample.json") as f:
        data = json.load(f)
    
    example = data[0]
    question = example["question"]
    gold = example["answer"]
    
    print(f"{'='*70}")
    print(f"Question: {question}")
    print(f"Gold answer: {gold}")
    print(f"{'='*70}\n")
    
    # Build retriever on this example's passages
    passages = load_passages_from_hotpot(example)
    retriever = Retriever()
    retriever.build_index(passages)
    
    # --- Naive baseline ---
    print(f"\n{'-'*70}")
    print("NAIVE RAG (baseline)")
    print(f"{'─'*70}")
    naive_ans = run_naive_rag(question, retriever, k=3)
    print(f"Answer: {naive_ans}")
    
    # --- HopSense ---
    print(f"\n{'-'*70}")
    print("HOPSENSE (multi-hop)")
    print(f"{'─'*70}")
    result = run_hopsense(question, retriever, k=3)
    print(f"Multi-hop detected: {result.is_multi_hop}")
    print(f"Sub-questions:")
    for i, sq in enumerate(result.sub_questions, 1):
        print(f"  {i}. {sq}")
    print(f"\nHop answers:")
    for h in result.hop_answers:
        print(f"  Q: {h.sub_question}")
        print(f"  A: {h.answer} (conf={h.confidence:.2f}, sources={h.supporting_titles})")
    print(f"\nFinal answer: {result.final_answer}")
    print(f"Overall confidence: {result.overall_confidence:.2f}")