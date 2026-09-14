"""
Batch evaluation: run naive RAG vs HopSense on all HotpotQA questions
in data/hotpot_sample.json. Save per-question results as JSON.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
import time
from pathlib import Path
from typing import List, Dict, Any

from retriever import Retriever, load_passages_from_hotpot
from pipeline import run_naive_rag, run_hopsense


def normalize_answer(s: str) -> str:
    """Normalize an answer for exact-match comparison."""
    import re, string
    s = s.lower().strip()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    s = ' '.join(s.split())
    return s


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def contains_gold(pred: str, gold: str) -> bool:
    """Softer match: is gold answer contained in prediction?"""
    return normalize_answer(gold) in normalize_answer(pred)


def evaluate_all(data_path: str = "data/hotpot_sample.json",
                 output_path: str = "results/batch_results.json",
                 n_limit: int = None):
    
    with open(data_path) as f:
        data = json.load(f)
    
    if n_limit:
        data = data[:n_limit]
    
    print(f"Evaluating {len(data)} questions...")
    print(f"{'='*70}\n")
    
    results = []
    
    for i, example in enumerate(data):
        question = example["question"]
        gold = example["answer"]
        
        print(f"[{i+1}/{len(data)}] {question[:80]}...")
        
        # Build fresh retriever for each question 
        # (HotpotQA gives per-question distractor context)
        passages = load_passages_from_hotpot(example)
        retriever = Retriever()
        retriever.build_index(passages)
        
        # --- Naive RAG ---
        t0 = time.time()
        try:
            naive_ans = run_naive_rag(question, retriever, k=3)
            naive_error = None
        except Exception as e:
            naive_ans = ""
            naive_error = str(e)
        naive_time = time.time() - t0
        
        # --- HopSense ---
        t0 = time.time()
        try:
            hop_result = run_hopsense(question, retriever, k=3)
            hop_ans = hop_result.final_answer
            hop_conf = hop_result.overall_confidence
            hop_sub_qs = hop_result.sub_questions
            hop_per_hop = [
                {
                    "sub_q": h.sub_question,
                    "answer": h.answer,
                    "confidence": h.confidence,
                    "sources": h.supporting_titles,
                }
                for h in hop_result.hop_answers
            ]
            hop_multi = hop_result.is_multi_hop
            hop_error = None
        except Exception as e:
            hop_ans = ""
            hop_conf = 0.0
            hop_sub_qs = []
            hop_per_hop = []
            hop_multi = False
            hop_error = str(e)
        hop_time = time.time() - t0
        
        result = {
            "id": example.get("id", str(i)),
            "question": question,
            "gold_answer": gold,
            "type": example.get("type", ""),
            "level": example.get("level", ""),
            "naive": {
                "answer": naive_ans,
                "exact_match": exact_match(naive_ans, gold),
                "contains_gold": contains_gold(naive_ans, gold),
                "time_sec": round(naive_time, 2),
                "error": naive_error,
            },
            "hopsense": {
                "answer": hop_ans,
                "exact_match": exact_match(hop_ans, gold),
                "contains_gold": contains_gold(hop_ans, gold),
                "confidence": hop_conf,
                "is_multi_hop": hop_multi,
                "sub_questions": hop_sub_qs,
                "per_hop": hop_per_hop,
                "time_sec": round(hop_time, 2),
                "error": hop_error,
            },
        }
        results.append(result)
        
        # Quick per-question print
        naive_ok = "✓" if result["naive"]["contains_gold"] else "✗"
        hop_ok = "✓" if result["hopsense"]["contains_gold"] else "✗"
        print(f"    Gold: {gold}")
        print(f"    Naive [{naive_ok}]: {naive_ans[:80]}")
        print(f"    HopSense [{hop_ok}]: {hop_ans[:80]}  (conf={hop_conf:.2f})")
        print()
    
    # Save
    Path(output_path).parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary
    n = len(results)
    naive_em = sum(r["naive"]["exact_match"] for r in results)
    naive_cg = sum(r["naive"]["contains_gold"] for r in results)
    hop_em = sum(r["hopsense"]["exact_match"] for r in results)
    hop_cg = sum(r["hopsense"]["contains_gold"] for r in results)
    hop_multi_detected = sum(r["hopsense"]["is_multi_hop"] for r in results)
    
    avg_naive_time = sum(r["naive"]["time_sec"] for r in results) / n
    avg_hop_time = sum(r["hopsense"]["time_sec"] for r in results) / n
    
    print(f"\n{'='*70}")
    print(f"SUMMARY on {n} questions")
    print(f"{'='*70}")
    print(f"Multi-hop detected: {hop_multi_detected}/{n} ({100*hop_multi_detected/n:.0f}%)")
    print()
    print(f"                    Exact Match    Contains Gold    Avg Time")
    print(f"Naive RAG:          {naive_em:3d}/{n} ({100*naive_em/n:.0f}%)     {naive_cg:3d}/{n} ({100*naive_cg/n:.0f}%)      {avg_naive_time:.2f}s")
    print(f"HopSense:           {hop_em:3d}/{n} ({100*hop_em/n:.0f}%)     {hop_cg:3d}/{n} ({100*hop_cg/n:.0f}%)      {avg_hop_time:.2f}s")
    print()
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    # Start with 10 questions to make sure everything works
    # Then bump to 100 when confident.
    evaluate_all(n_limit=100)