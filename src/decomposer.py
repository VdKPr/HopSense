"""
Decomposer: breaks a complex multi-hop question into simpler sub-questions.

Example:
    "Were Scott Derrickson and Ed Wood of the same nationality?"
    ->
    ["What is Scott Derrickson's nationality?",
     "What is Ed Wood's nationality?"]
"""

import os
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


class DecomposedQuestion(BaseModel):
    """Structured output for question decomposition."""
    reasoning: str = Field(
        description="Brief explanation of why the question needs decomposition"
    )
    sub_questions: List[str] = Field(
        description="List of simpler questions that together answer the original"
    )
    is_multi_hop: bool = Field(
        description="True if the question requires multiple retrieval steps"
    )

# DECOMPOSER_PROMPT = """You are an expert at breaking down complex questions into simpler ones.

# Given a question, decide if it needs multiple retrieval steps to answer.
# - Single-hop questions (e.g., "What year was Einstein born?") need no decomposition.
#   Return is_multi_hop=False and sub_questions=[the original question].
# - Multi-hop questions require retrieving information about multiple entities or 
#   chained lookups.

# For multi-hop questions, produce 2-3 atomic sub-questions.

# STRICT RULES for sub-questions:
# 1. Each sub-question must be answerable ON ITS OWN by looking up ONE fact in 
#    Wikipedia or a factual corpus. No sub-question may depend on the answer to 
#    another sub-question.
# 2. Each sub-question must name a SPECIFIC entity from the original question 
#    (e.g. a person, place, work, event). Do NOT write sub-questions like 
#    "the actress identified in question 1" or "this person" — the retriever 
#    is stateless and cannot resolve back-references.
# 3. Do NOT include a wrapper sub-question that just restates the original 
#    question (e.g. "What is the answer to the original question?"). 
#    The original will be answered by combining the sub-question answers.
# 4. Do NOT decompose beyond what is necessary. If two sub-questions are enough, 
#    do not produce three. Fewer, sharper sub-questions beat more, weaker ones.
# 5. Prefer sub-questions of the form: 
#    "What is the [attribute] of [specific entity]?" or 
#    "Who [did specific action]?"

# Return your reasoning, whether it's multi-hop, and the sub-questions.

# Question: {question}"""

# whiile tackling Ready for M6 — fix Problem 07.3, stricter decomposer prompt.
# But did not worked to revered back to the original and commented upper Decomposer prompt and added the below one to make it work.
DECOMPOSER_PROMPT = """You are an expert at breaking down complex questions into simpler ones.

Given a question, decide if it needs multiple retrieval steps to answer.
- Single-hop questions (e.g., "What year was Einstein born?") need no decomposition.
- Multi-hop questions require retrieving information about multiple entities or 
  reasoning across facts (e.g., comparing two people, chained lookups).

For multi-hop questions, break them into 2-4 atomic sub-questions.
Each sub-question should be answerable independently by looking up one fact.

Return your reasoning, whether it's multi-hop, and the sub-questions.

Question: {question}"""


def get_decomposer():
    """Returns a decomposer chain with structured output."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_structured = llm.with_structured_output(DecomposedQuestion)
    
    prompt = ChatPromptTemplate.from_template(DECOMPOSER_PROMPT)
    chain = prompt | llm_structured
    return chain


def decompose(question: str) -> DecomposedQuestion:
    """Decompose a question into sub-questions."""
    chain = get_decomposer()
    return chain.invoke({"question": question})


# ============ TEST ============
if __name__ == "__main__":
    test_questions = [
        "Were Scott Derrickson and Ed Wood of the same nationality?",
        "What year was Einstein born?",
        "Did the CEO of the company that acquired GitHub work at Microsoft before 2010?",
    ]
    
    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print(f"{'='*60}")
        result = decompose(q)
        print(f"Multi-hop: {result.is_multi_hop}")
        print(f"Reasoning: {result.reasoning}")
        print(f"Sub-questions:")
        for i, sq in enumerate(result.sub_questions, 1):
            print(f"  {i}. {sq}")